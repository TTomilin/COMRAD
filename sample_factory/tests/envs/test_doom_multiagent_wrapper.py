from __future__ import annotations

from queue import Queue

import gymnasium as gym
from sample_factory.utils.attr_dict import AttrDict

import comrad.envs.multiagent.doom_multiagent_wrapper as doom_multiagent_wrapper


class _TinyEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, **kwargs):
        return 0, {}

    def step(self, action):
        return 0, 0.0, False, False, {}

    def close(self):
        return None


def test_multiagent_reset_retries_after_group_crash(monkeypatch):
    monkeypatch.setattr(doom_multiagent_wrapper, "get_default_reward_shaping", lambda env: {})

    env = doom_multiagent_wrapper.MultiAgentEnv(
        num_agents=2,
        make_env_func=lambda player_id: _TinyEnv(),
        env_config=AttrDict(worker_index=0, vector_index=0, safe_init=False),
        skip_frames=1,
        render_mode=None,
    )

    calls = {"await": 0, "close": 0}
    env.initialized = True

    monkeypatch.setattr(env, "_ensure_initialized", lambda: None)
    monkeypatch.setattr(doom_multiagent_wrapper, "sleep", lambda _: None)

    def fake_close():
        calls["close"] += 1

    def fake_await_tasks(data, task_type, timeout=None):
        calls["await"] += 1
        if calls["await"] == 1:
            raise doom_multiagent_wrapper._GameGroupCrashError("reset crash")
        return [0, 0], [{}, {}]

    monkeypatch.setattr(env, "close", fake_close)
    monkeypatch.setattr(env, "await_tasks", fake_await_tasks)

    obs, info = env.reset()

    assert calls["close"] == 1
    assert env.initialized is False
    assert obs == [0, 0]
    assert info == [{}, {}]


def test_multiagent_worker_reports_crashed_when_init_raises():
    worker = object.__new__(doom_multiagent_wrapper.MultiAgentEnvWorker)
    worker.player_id = 0
    worker.task_queue = Queue()
    worker.result_queue = Queue()
    worker._terminate = lambda env: None
    worker._init = lambda data: (_ for _ in ()).throw(RuntimeError("boom"))

    worker.task_queue.put(({"port": 40300}, doom_multiagent_wrapper.TaskType.INIT))
    worker.start()

    assert worker.result_queue.get_nowait() is doom_multiagent_wrapper._CRASHED


def test_multiagent_init_retries_after_init_crash(monkeypatch):
    monkeypatch.setattr(doom_multiagent_wrapper, "get_default_reward_shaping", lambda env: {})
    monkeypatch.setattr(doom_multiagent_wrapper, "sleep", lambda _: None)
    monkeypatch.setattr(doom_multiagent_wrapper.time, "sleep", lambda _: None)
    monkeypatch.setattr(doom_multiagent_wrapper, "find_available_port", lambda start_port, increment=1000: start_port)

    class _DummyLock:
        def acquire(self, timeout=10):
            class _Ctx:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()

    monkeypatch.setattr(doom_multiagent_wrapper, "FileLock", lambda path: _DummyLock())
    monkeypatch.setattr(doom_multiagent_wrapper, "doom_lock_file", lambda max_parallel: "/tmp/dummy.lock")

    env = doom_multiagent_wrapper.MultiAgentEnv(
        num_agents=2,
        make_env_func=lambda player_id: _TinyEnv(),
        env_config=AttrDict(worker_index=0, vector_index=0, safe_init=False),
        skip_frames=1,
        render_mode=None,
    )

    class _DummyProcess:
        def join(self, timeout=None):
            return None

        def is_alive(self):
            return False

    spawn_round = {"count": 0}
    class _DummyWorker:
        def __init__(self, player_id, make_env_func, env_config, use_multiprocessing=False, reset_on_init=True):
            self.player_id = player_id
            self.task_queue = Queue()
            self.result_queue = Queue()
            self.process = _DummyProcess()
            self.init_calls = 0
            self.round = spawn_round["count"]

        def queue_init_result(self):
            if self.round == 1 and self.player_id == 0 and self.init_calls == 0:
                self.init_calls += 1
                self.result_queue.put(doom_multiagent_wrapper._CRASHED)
            else:
                self.init_calls += 1
                self.result_queue.put(None)

    class _InitAwareQueue(Queue):
        def __init__(self, owner):
            super().__init__()
            self.owner = owner

        def put(self, item, block=True, timeout=None):
            super().put(item, block=block, timeout=timeout)
            data, task_type = item
            if task_type == doom_multiagent_wrapper.TaskType.INIT:
                self.owner.queue_init_result()
            elif task_type == doom_multiagent_wrapper.TaskType.TERMINATE:
                self.owner.result_queue.put(None)

    def _spawn_workers_with_init_queue():
        spawn_round["count"] += 1
        env.workers = []
        for i in range(env.num_agents):
            worker = _DummyWorker(i, env.make_env_func, env.env_config, reset_on_init=env.reset_on_init)
            worker.task_queue = _InitAwareQueue(worker)
            env.workers.append(worker)

    monkeypatch.setattr(env, "_spawn_workers", _spawn_workers_with_init_queue)

    env._ensure_initialized()

    assert env.initialized is True
    assert spawn_round["count"] == 2
