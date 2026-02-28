import threading
import time
from enum import Enum
from functools import wraps
from multiprocessing import Process
from queue import Empty, Queue
from time import sleep
from typing import Union

import cv2
import faster_fifo
import filelock
import gymnasium as gym
from filelock import FileLock

from sample_factory.algo.utils.rl_utils import make_dones
from sample_factory.envs.env_utils import RewardShapingInterface, get_default_reward_shaping
from sample_factory.utils.utils import log
from sf.doom.doom_gym import doom_lock_file
from sf.doom.doom_render import concat_grid
from sf.doom.multiplayer.doom_multiagent import DEFAULT_UDP_PORT, find_available_port

_CRASHED = object() # await_tasks() checks for this to check if the whole game group is dead
class _GameGroupCrashError(Exception):
    """
    All vizdoom games have one UDP game. If one crashes, the others cannot recover independently, the entire
    group must be killed and reinitialised on a fresh port.
    This should be a separate Exception class
    """


def retry_doom(exception_class=Exception, num_attempts=3, sleep_time=1, should_reset=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(num_attempts):
                try:
                    return func(*args, **kwargs)
                except exception_class as e:
                    # This accesses the self instance variable
                    multiagent_wrapper_obj = args[0]
                    multiagent_wrapper_obj.initialized = False
                    multiagent_wrapper_obj.close()

                    # This is done to reset if it is in the step function
                    if should_reset:
                        multiagent_wrapper_obj.reset()

                    if i == num_attempts - 1:
                        raise
                    else:
                        log.error("Failed with error %r, trying again", e)
                        sleep(sleep_time)

        return wrapper

    return decorator


def safe_get(q, timeout=1e6, msg="Queue timeout"):
    """Using queue.get() with timeout is necessary, otherwise KeyboardInterrupt is not handled."""
    while True:
        try:
            return q.get(timeout=timeout)
        except Empty:
            log.warning(msg)


def udp_port_num(env_config):
    if env_config is None:
        return DEFAULT_UDP_PORT
    port_to_use = DEFAULT_UDP_PORT + 100 * env_config.worker_index + env_config.vector_index
    return port_to_use


class TaskType(Enum):
    INIT, TERMINATE, RESET, STEP, STEP_UPDATE, INFO, SET_ATTR = range(7)


def init_multiplayer_env(make_env_func, player_id, env_config, init_info=None):
    env = make_env_func(player_id=player_id)

    if env_config is not None and "worker_index" in env_config:
        env.unwrapped.worker_index = env_config.worker_index
    if env_config is not None and "vector_index" in env_config:
        env.unwrapped.vector_index = env_config.vector_index

    if init_info is None:
        port_to_use = udp_port_num(env_config)
        port = find_available_port(port_to_use, increment=1000)
        log.debug("Using port %d", port)
        init_info = dict(port=port)

    env.unwrapped.init_info = init_info

    env.unwrapped.seed(env.unwrapped.worker_index * 1000 + env.unwrapped.vector_index * 10 + player_id)
    return env


class MultiAgentEnvWorker:
    def __init__(self, player_id, make_env_func, env_config, use_multiprocessing=False, reset_on_init=True):
        self.player_id = player_id
        self.make_env_func = make_env_func
        self.env_config = env_config
        self.reset_on_init = reset_on_init
        if use_multiprocessing:
            self.process = Process(target=self.start, daemon=False)
            self.task_queue, self.result_queue = faster_fifo.Queue(), faster_fifo.Queue()
        else:
            self.process = threading.Thread(target=self.start)
            self.task_queue, self.result_queue = Queue(), Queue()

        self.process.start()

    def _init(self, init_info):
        log.info("Initializing env for player %d, init_info: %r...", self.player_id, init_info)
        env = init_multiplayer_env(self.make_env_func, self.player_id, self.env_config, init_info)
        if self.reset_on_init:
            env.reset()
        return env

    @staticmethod
    def _terminate(env):
        if env is None:
            return
        env.close()

    @staticmethod
    def _get_info(env):
        """Specific to custom VizDoom environments."""
        info = {}
        if hasattr(env.unwrapped, "get_info_all"):
            info = env.unwrapped.get_info_all()  # info for the new episode
        return info

    def _set_env_attr(self, env, player_id, attr_chain, value):
        """Allows us to set an arbitrary attribute of the environment, e.g. attr_chain can be unwrapped.foo.bar"""
        assert player_id == self.player_id, "Can only set attributes for the current player"

        attrs = attr_chain.split(".")
        curr_attr = env
        try:
            for attr_name in attrs[:-1]:
                curr_attr = getattr(curr_attr, attr_name)
        except AttributeError:
            log.error("Env does not have an attribute %s", attr_chain)

        attr_to_set = attrs[-1]
        setattr(curr_attr, attr_to_set, value)

    def start(self):
        env = None

        while True:
            data, task_type = safe_get(self.task_queue)

            if task_type == TaskType.INIT:
                env = self._init(data)
                self.result_queue.put(None)  # signal we're done
                continue

            if task_type == TaskType.TERMINATE:
                self._terminate(env)
                break

            # ViZDoom 1.3.0 multiplayer crashes (signal 11) during new_episode() and advance_action(). When one process in a multiplayer game dies, the peers cant recover on their own because the UDP game is gone
            # So we signal _CRASHED back to the main thread so it can kill the whole group and reinit new one on new port
            # Another fix is on vizdoom codebase as reported here:
            # https://github.com/Farama-Foundation/ViZDoom/issues/693
            try:
                results = None
                if task_type == TaskType.RESET:
                    results = env.reset(**data) if data else env.reset()
                elif task_type == TaskType.INFO:
                    results = self._get_info(env)
                elif task_type == TaskType.STEP or task_type == TaskType.STEP_UPDATE:
                    action = data
                    env.unwrapped.update_state = task_type == TaskType.STEP_UPDATE
                    results = env.step(action)
                elif task_type == TaskType.SET_ATTR:
                    player_id, attr_chain, value = data
                    self._set_env_attr(env, player_id, attr_chain, value)
                else:
                    raise Exception(f"Unknown task type {task_type}")

                self.result_queue.put(results)
            except Exception as exc:
                log.error(
                    "ViZDoom worker player_id=%d crashed during %s: %s",
                    self.player_id, task_type, exc,
                )
                self.result_queue.put(_CRASHED)
                # Cleanup
                try:
                    self._terminate(env)
                except Exception:
                    pass
                # Proc dead
                break # Exit the worker loop


class MultiAgentEnv(gym.Env, RewardShapingInterface):
    def __init__(self, num_agents, make_env_func, env_config, skip_frames, render_mode):
        gym.Env.__init__(self)
        RewardShapingInterface.__init__(self)

        self.num_agents = num_agents
        log.debug("Multi agent env, num agents: %d", self.num_agents)
        self.skip_frames = skip_frames  # number of frames to skip (1 = no skip)

        env = make_env_func(player_id=-1)  # temporary env just to query observation_space and stuff
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.default_reward_shaping = get_default_reward_shaping(env)
        env.close()

        self.current_reward_shaping = [self.default_reward_shaping for _ in range(self.num_agents)]

        self.make_env_func = make_env_func

        self.safe_init = env_config is not None and env_config.get("safe_init", False)

        if self.safe_init:
            sleep_seconds = env_config.worker_index * 1.0
            log.info("Sleeping %.3f seconds to avoid creating all envs at once", sleep_seconds)
            time.sleep(sleep_seconds)
            log.info("Done sleeping at %d", env_config.worker_index)

        self.env_config = env_config
        self.workers = None

        # only needed when rendering
        self.enable_rendering = False
        self.last_obs = None

        self.reset_on_init = True

        self.initialized = False

        self.render_mode = render_mode

    # def wipe_when_one_die(self, terminated, truncated, infos):
    #     """This function is quite specific to pitfall, temrinates when one agent dies to make it 'cooperative'
    #     Goal is to ensure all agents reach the end together.

    #     Note: DoomPitfallRewardShaping run before this method in step(). When wipe_when_one_die force-terminates surviving
    #     agents, their wrappers already returned with done=False, so true_objective and team_score_adjust are not applied for
    #     alive agents on the step.
    #     """
    #     lst_dead = [bool(info.get("DEAD", 0)) for info in infos]

    #     if not any(lst_dead):
    #         return terminated, truncated

    #     # Terminates when any agent dies to make sure all of them reaches the end together
    #     # This doesn't really enough for now as no credit assignment, might uncomment all() for more forgiving
    #     # if all(lst_dead):
    #     if any(lst_dead):
    #         terminated = [True] * self.num_agents
    #         truncated = [False] * self.num_agents
    #         for info in infos:
    #             extra_stats = info.setdefault("episode_extra_stats", {})
    #             extra_stats["one_agent_died"] = 1
    #     return terminated, truncated

    def get_default_reward_shaping(self):
        return self.default_reward_shaping

    def set_reward_shaping(self, reward_shaping: dict, agent_indices: Union[int, slice]):
        if isinstance(agent_indices, int):
            agent_indices = slice(agent_indices, agent_indices + 1)
        for agent_idx in range(agent_indices.start, agent_indices.stop):
            self.current_reward_shaping[agent_idx] = reward_shaping
            self.set_env_attr(
                agent_idx,
                "unwrapped.reward_shaping_interface.reward_shaping_scheme",
                reward_shaping,
            )

    def await_tasks(self, data, task_type, timeout=None):
        """
        Task result is always a tuple of lists, e.g.:
        (
            [0th_agent_obs, 1st_agent_obs, ... ],
            [0th_agent_reward, 1st_agent_reward, ... ],
            ...
        )

        If your "task" returns only one result per agent (e.g. reset() returns only the observation),
        the result will be a tuple of length 1. It is a responsibility of the caller to index appropriately.

        """
        if data is None:
            data = [None] * self.num_agents

        assert len(data) == self.num_agents, f"Expected {self.num_agents} items, got {len(data)}"

        for i, worker in enumerate(self.workers):
            worker.task_queue.put((data[i], task_type))

        result_lists = None
        # env.step() mostly in C++ and I/O, which releases GIL, so technically those workers run in parallel even though they are threads
        # So the main bottleneck here is main thread waiting for each worker's result in .get()
        # This is sequential, so might make it async so main thread waits for all at once
        # TODO: Try asynchronous collection with select() or asyncio
        for i, worker in enumerate(self.workers):
            results = safe_get(
                worker.result_queue,
                timeout=0.2 if timeout is None else timeout,
                msg=f"Takes a surprisingly long time to process task {task_type}, retry...",
            )

            # A crashed worker puts _CRASHED on its queue
            # The whole multiplayer game should be killed now
            if results is _CRASHED:
                log.error(f"Game group crash detected (worker {i}). Tearing down all {self.num_agents} workers and reinitializing.")
                raise _GameGroupCrashError(f"ViZDoom worker {i} crashed during {task_type}")

            if not isinstance(results, (tuple, list)):
                results = [results]

            if result_lists is None:
                result_lists = tuple([] for _ in results)

            for j, r in enumerate(results):
                result_lists[j].append(r)

        # This only improves notably for more agents
        # Comparison runs with 4 agents (check the tag: non_parallel vs parallel collection):
        # https://wandb.ai/khoi-eindhoven-university-of-technology/marl_vizdoom/table
        # Sync: fps=2.608,0653061224
        # Async: fps=3.305,1611185087
        #
        # Edit: This shouldnt matter, main thread needs all results before proceeding to next step anyways,
        # so the total time is always blocked by the slowest worker
        #
        # async def _collect():
        #     a = asyncio.get_running_loop()
        #     tasks = [
        #         a.run_in_executor(None, safe_get, worker.result_queue,
        #             0.2 if timeout is None else timeout,
        #             f"Takes a surprisingly long time to process task {task_type}, retry...",
        #         ) for worker in self.workers]
        #     return await asyncio.gather(*tasks)
        # results = asyncio.run(_collect())
        # for r in results:
        #     if not isinstance(r, (tuple, list)):
        #         r = [r]
        #     if result_lists is None:
        #         result_lists = tuple([] for _ in r)
        #     for j, r in enumerate(r):
        #         result_lists[j].append(r)

        return result_lists

    def _ensure_initialized(self):
        if self.initialized:
            return

        self.workers = [
            MultiAgentEnvWorker(i, self.make_env_func, self.env_config, reset_on_init=self.reset_on_init)
            for i in range(self.num_agents)
        ]

        init_attempt = 0
        while True:
            init_attempt += 1
            try:
                port_to_use = udp_port_num(self.env_config)
                port = find_available_port(port_to_use, increment=1000)
                log.debug("Using port %d", port)
                init_info = dict(port=port)

                lock_file = doom_lock_file(max_parallel=20)
                lock = FileLock(lock_file)
                with lock.acquire(timeout=10):
                    for i, worker in enumerate(self.workers):
                        worker.task_queue.put((init_info, TaskType.INIT))
                        if self.safe_init:
                            time.sleep(1.0)  # just in case
                        else:
                            time.sleep(0.05)

                    for i, worker in enumerate(self.workers):
                        worker.result_queue.get(timeout=70)

            except filelock.Timeout:
                continue
            except Exception as exc:
                raise RuntimeError(f"Critical error: worker stuck on initialization. Abort! {exc}")
            else:
                break

        log.debug("%d agent workers initialized for env %d!", len(self.workers), self.env_config.worker_index)
        self.initialized = True

    @retry_doom(exception_class=Exception, num_attempts=3, sleep_time=1, should_reset=False)
    def info(self):
        self._ensure_initialized()
        info = self.await_tasks(None, TaskType.INFO)
        if info is None:
            return None
        return info[0]

    @retry_doom(exception_class=Exception, num_attempts=3, sleep_time=1, should_reset=False)
    def reset(self, **kwargs):
        self._ensure_initialized()
        observation, info = self.await_tasks([kwargs] * self.num_agents, TaskType.RESET, timeout=2.0)
        return observation, info

    @retry_doom(exception_class=Exception, num_attempts=3, sleep_time=1, should_reset=True)
    def step(self, actions):
        self._ensure_initialized()

        for frame in range(self.skip_frames - 1):
            self.await_tasks(actions, TaskType.STEP)

        obs, rew, terminated, truncated, infos = self.await_tasks(actions, TaskType.STEP_UPDATE)
        dones = make_dones(terminated, truncated)

        for info in infos:
            info["num_frames"] = self.skip_frames

        if all(dones):
            obs, reset_infos = self.await_tasks([{}] * self.num_agents, TaskType.RESET, timeout=2.0)
            for i, reset_info in enumerate(reset_infos):
                infos[i]["reset_info"] = reset_info

        if self.enable_rendering:
            self.last_obs = obs

        return obs, rew, terminated, truncated, infos

    # noinspection PyUnusedLocal
    def render(self):
        self.enable_rendering = True

        if self.last_obs is None:
            return

        if self.render_mode is None:
            return
        elif self.render_mode == "human":
            # for o in self.last_obs: print(o)
            # For MUTLI-AGENT scenarios with ADDITIONAL_INPUT in extra_wrappers, obs is wrapped in a dictionary
            # with format like {"obs": array([[[...]]]), 'measurements': array([...])}
            # But then for multiagent scenarios like pitfall, we dont need extra game variables, so we dont pass that,
            # thus observations remains the plain np arrays, so here we have to check if it's dict or array here
            obs_display = [o["obs"] if isinstance(o, dict) else o for o in self.last_obs]
            obs_grid = concat_grid(obs_display, self.render_mode)
            cv2.imshow("vizdoom", obs_grid)
        elif self.render_mode == "rgb_array":
            obs_display = [o["obs"] if isinstance(o, dict) else o for o in self.last_obs]
            obs_grid = concat_grid(obs_display, self.render_mode)
            return obs_grid
        else:
            raise ValueError(f"{self.render_mode=} is not supported")

    def close(self):
        if self.workers is not None:
            # log.info('Stopping multiagent env %d...', self.env_config.worker_index)
            for worker in self.workers:
                try:
                    worker.task_queue.put((None, TaskType.TERMINATE))
                except Exception:
                    pass
                time.sleep(0.1)
            for worker in self.workers:
                worker.process.join(timeout=5)
                if worker.process.is_alive():
                    log.warning(
                        f"Worker player_id={worker.player_id} did not exit in time (stuck in vizdoom C++ code). "
                        "Thread will be abandoned and auto clean up on process exit."
                    )
            self.workers = None

    def set_env_attr(self, agent_idx, attr_chain, value):
        data = (agent_idx, attr_chain, value)
        worker = self.workers[agent_idx]
        worker.task_queue.put((data, TaskType.SET_ATTR))

        result = safe_get(worker.result_queue, timeout=0.1)
        assert result is None, f"Expected None, got {result}"
