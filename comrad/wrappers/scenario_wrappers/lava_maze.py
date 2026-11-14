import gymnasium as gym
import numpy as np

from comrad.wrappers.scenario_wrappers.lava_maze_grid import (
    bfs_distance,
    decode_maze_grid,
    next_step_from_info,
    safe_int,
)
from comrad.wrappers.state_observation import ego_state_dim, ego_state_measurement_slice, measurement_scale

LAVA_MAZE_MEASUREMENTS_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    high=np.array([200., 1., 1., 1., 1., 1., 1.], dtype=np.float32),
    dtype=np.float32,
)


class LavaMazeRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        grid_size=8,
        goal_reward=10.0,
        death_penalty=-5.0,
        step_penalty=-0.005,
        distance_reward_scale=0.5,
        lava_burn_penalty_scale=0.02,
        signal_penalty=-0.025,
    ):
        super().__init__(env)
        self.grid_size = grid_size
        self.goal_reward = goal_reward
        self.death_penalty = death_penalty
        self.step_penalty = step_penalty
        self.distance_reward_scale = distance_reward_scale
        self.lava_burn_penalty_scale = lava_burn_penalty_scale
        self.signal_penalty = signal_penalty

        self.flash_window = 30
        self.sender_bonus_scale = 0.2
        self.no_progress_penalty = -0.005
        self.flash_timer = 0
        self.progress_occurred = False

        self.prev_vars = {}
        self.orig_env_reward = 0.0
        # Benchmark score J = mazes completed. Counted as goal relocations rather than read from the ACS level counter
        # (USER22), which the released WAD advances twice per completion (the goal moves one tic after the check).
        self.goals_reached = 0
        self._last_goal = None

    def _track_goal(self, info):
        goal = (
            self._safe_int((info or {}).get("USER13", -1), -1),
            self._safe_int((info or {}).get("USER18", -1), -1),
        )
        if goal[0] < 0 or goal[1] < 0:
            return
        if self._last_goal is not None and goal != self._last_goal:
            self.goals_reached += 1
        self._last_goal = goal

    def _decode_maze_grid(self, bits_0, bits_1, bits_2, grid_size=None):
        """
        Reconstructs the grid from the 3 ACS integer chunks.
        Returns a 2D list where 1 is safe, 0 is lava.
        """
        return decode_maze_grid(bits_0, bits_1, bits_2, grid_size or self.grid_size)

    def _safe_int(self, value, default):
        return safe_int(value, default)

    def _grid_size_from(self, info):
        return max(1, self._safe_int((info or {}).get("USER11", self.grid_size), self.grid_size))

    def _get_bfs_distance(self, grid, start_x, start_y, goal_x, goal_y, grid_size=None):
        """
        Calculates the shortest path distance using BFS (999 when in lava or unreachable).
        """
        return bfs_distance(grid, start_x, start_y, goal_x, goal_y, grid_size)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None or info is None:
            return obs, reward, terminated, truncated, info

        self._track_goal(info)

        if not self.prev_vars:
            self.sync_vars(info)
            if terminated or truncated:
                info["true_objective"] = self.goals_reached
            return obs, 0.0, terminated, truncated, info

        shaped_reward = self.step_penalty

        p1_x = self._safe_int(info.get("USER14", -1), -1)
        p1_y = self._safe_int(info.get("USER15", -1), -1)
        prev_p1_x = self._safe_int(self.prev_vars.get("USER14", -1), -1)
        prev_p1_y = self._safe_int(self.prev_vars.get("USER15", -1), -1)

        goal_x = self._safe_int(info.get("USER13", -1), -1)
        goal_y = self._safe_int(info.get("USER18", -1), -1)
        prev_goal_x = self._safe_int(self.prev_vars.get("USER13", -1), -1)
        prev_goal_y = self._safe_int(self.prev_vars.get("USER18", -1), -1)

        current_levels = self._safe_int(info.get("USER22", 0), 0)
        prev_levels = self._safe_int(self.prev_vars.get("USER22", 0), 0)

        current_signal = self._safe_int(info.get("USER16", 0), 0)

        current_flash_active = self._safe_int(info.get("USER17", 0), 0)
        prev_flash_active = self._safe_int(self.prev_vars.get("USER17", 0), 0)
        new_flash = current_flash_active == 1 and prev_flash_active == 0 and current_signal != 0

        if new_flash:
            shaped_reward += self.signal_penalty

        current_hp = self._safe_int(info.get("HEALTH", 0), 0)
        prev_hp = self._safe_int(self.prev_vars.get("HEALTH", 0), 0)
        grid_size = self._grid_size_from(info)

        # Sender bonus logic
        player_id = getattr(self.env.unwrapped, "player_id", -1)
        if player_id == 1:
            if new_flash:
                self.flash_timer = self.flash_window
                self.progress_occurred = False

            if self.flash_timer > 0:
                self.flash_timer -= 1
                if (p1_x != -1 and p1_y != -1 and goal_x != -1 and goal_y != -1 and
                    prev_p1_x != -1 and prev_p1_y != -1 and
                    goal_x == prev_goal_x and goal_y == prev_goal_y):

                    grid = self._decode_maze_grid(
                        self._safe_int(info.get("USER19", 0), 0),
                        self._safe_int(info.get("USER20", 0), 0),
                        self._safe_int(info.get("USER21", 0), 0),
                        grid_size,
                    )

                    current_dist = self._get_bfs_distance(grid, p1_x, p1_y, goal_x, goal_y, grid_size)
                    prev_dist = self._get_bfs_distance(grid, prev_p1_x, prev_p1_y, goal_x, goal_y, grid_size)

                    if current_dist != 999 and prev_dist != 999:
                        dist_diff = prev_dist - current_dist
                        if dist_diff > 0:
                            # Sender gets a fraction of the progress reward
                            sender_progress_bonus = dist_diff * self.distance_reward_scale * self.sender_bonus_scale
                            shaped_reward += sender_progress_bonus
                            self.progress_occurred = True

                if self.flash_timer == 0 and not self.progress_occurred:
                    shaped_reward += self.no_progress_penalty

        hp_loss = max(0, prev_hp - current_hp)
        if hp_loss > 0:
            shaped_reward -= hp_loss * self.lava_burn_penalty_scale

        if current_levels > prev_levels:
            shaped_reward += self.goal_reward

        if current_hp <= 0 and prev_hp > 0:
            shaped_reward += self.death_penalty

        # Shortest-Path Distance Reward
        if (p1_x != -1 and p1_y != -1 and goal_x != -1 and goal_y != -1 and
            prev_p1_x != -1 and prev_p1_y != -1):
            # Ensure the maze hasn't reset this exact frame
            if goal_x == prev_goal_x and goal_y == prev_goal_y:

                grid = self._decode_maze_grid(
                    self._safe_int(info.get("USER19", 0), 0),
                    self._safe_int(info.get("USER20", 0), 0),
                    self._safe_int(info.get("USER21", 0), 0),
                    grid_size,
                )

                current_dist = self._get_bfs_distance(grid, p1_x, p1_y, goal_x, goal_y, grid_size)
                prev_dist = self._get_bfs_distance(grid, prev_p1_x, prev_p1_y, goal_x, goal_y, grid_size)

                if current_dist != 999 and prev_dist != 999:
                    dist_diff = prev_dist - current_dist
                    shaped_reward += dist_diff * self.distance_reward_scale

        individual_reward = reward + shaped_reward
        self.orig_env_reward += reward

        if terminated or truncated:
            info["true_objective"] = self.goals_reached

        self.sync_vars(info)
        return obs, individual_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.sync_vars(info)
        self.flash_timer = 0
        self.progress_occurred = False
        self.orig_env_reward = 0.0
        # tracking starts at the first step: reset info can still carry the previous episode's goal
        self.goals_reached = 0
        self._last_goal = None
        return obs, info

    def sync_vars(self, info):
        info = info or {}
        self.prev_vars = {
            "USER11": self._safe_int(info.get("USER11", self.grid_size), self.grid_size),
            "USER13": self._safe_int(info.get("USER13", -1), -1),
            "USER14": self._safe_int(info.get("USER14", -1), -1),
            "USER15": self._safe_int(info.get("USER15", -1), -1),
            "USER16": self._safe_int(info.get("USER16", 0), 0),
            "USER17": self._safe_int(info.get("USER17", 0), 0),
            "USER18": self._safe_int(info.get("USER18", -1), -1),
            "USER19": self._safe_int(info.get("USER19", 0), 0),
            "USER20": self._safe_int(info.get("USER20", 0), 0),
            "USER21": self._safe_int(info.get("USER21", 0), 0),
            "USER22": self._safe_int(info.get("USER22", 0), 0),
            "HEALTH": self._safe_int(info.get("HEALTH", 100), 100),
        }

class LavaMazeAdditionalInput(gym.Wrapper):
    """
    health
    color_signal: 0=none, 1=green, 2=red, 3=yellow, 4=blue (USER16)
    flash_active: 0/1 (USER17)
    """

    def __init__(self, env):
        super().__init__(env)
        current_obs_space = self.observation_space

        self.observation_space = gym.spaces.Dict({
            "obs": current_obs_space,
            "measurements": LAVA_MAZE_MEASUREMENTS_SPACE,
        })

        self.measurements_vec = np.zeros(LAVA_MAZE_MEASUREMENTS_SPACE.shape, dtype=np.float32)

    def _parse_info(self, obs, info):
        obs_dict = {"obs": obs, "measurements": self.measurements_vec}

        if info is None:
            # for blocking vision and using additional vectors only
            # if getattr(self.env.unwrapped, "player_id", -1) == 0:
            #     obs = np.zeros_like(obs)
            # return {"obs": obs, "measurements": self.measurements_vec.copy()}
            return obs_dict

        measurements = obs_dict["measurements"]
        measurements[0] = max(0.0, info.get("HEALTH", 0.0))
        flash_active = bool(info.get("USER17", 0))
        measurements[1:6] = 0.0
        color = max(0, min(4, int(info.get("USER16", 0))))
        if not flash_active or color == 0:
            measurements[1] = 1.0
        else:
            measurements[1 + color] = 1.0

        measurements[6] = float(bool(info.get("USER17", 0)))

        # if getattr(self.env.unwrapped, "player_id", -1) == 0:
        #     obs = np.zeros_like(obs)

        return obs_dict

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.measurements_vec.fill(0.0)

        if info is None:
            info = self.env.unwrapped.get_info()

        obs_dict = self._parse_info(obs, info)
        return obs_dict, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        if obs is None:
            return obs, rew, terminated, truncated, info

        obs_dict = self._parse_info(obs, info)
        return obs_dict, rew, terminated, truncated, info


def _set_direction_one_hot(measurements, direction):
    """measurements[1:6] = one-hot over (none, +x/green, -x/red, +y/yellow, -y/blue), measurements[6] = active flag."""
    direction = max(0, min(4, int(direction)))
    measurements[1:6] = 0.0
    measurements[1 + direction] = 1.0
    measurements[6] = float(direction > 0)
    return measurements


class LavaMazeOracleHintInput(LavaMazeAdditionalInput):
    """
    Oracle variant (communication removed): the navigator's measurement vector carries the BFS next-step direction
    toward the goal, computed from the same telemetry as LavaMazeRewardShaping. The observer gets health only.
    Same 7-dim layout as LavaMazeAdditionalInput: [HEALTH, none, +x, -x, +y, -y, hint_active].
    """

    def _parse_info(self, obs, info):
        obs_dict = {"obs": obs, "measurements": self.measurements_vec}
        if info is None:
            return obs_dict

        measurements = obs_dict["measurements"]
        measurements[0] = max(0.0, info.get("HEALTH", 0.0))
        if getattr(self.env.unwrapped, "player_id", -1) == 0:
            _set_direction_one_hot(measurements, next_step_from_info(info))
        else:
            measurements[1:7] = 0.0
        return obs_dict


class LavaMazeOracleMessageWrapper(gym.Wrapper):
    """
    Oracle variant (lossless channel), applied around MultiAgentEnv. The observer's weapon-select head
    (action head `message_head`, values 0..4) is a discrete message; its value at step t is written one-hot into the
    navigator's measurements[1:6] at step t+1 (no weapon-fire latency, flash timer or signal penalty).
    Requires LavaMazeAdditionalInput on the per-player stack; the observer's ATTACK should be masked so the ACS
    flash never fires (LAVA_MAZE_ORACLE_MSG_ACTION_MASK).

    The channel is the `measurements` key (pixel/blind modality) or, with --obs_modality=state, the measurement block
    StateObservationWrapper folded into the `ego` state vector. Without either the constructor raises.
    """

    def __init__(self, env, navigator_index=0, observer_index=1, message_head=3):
        super().__init__(env)
        self.num_agents = getattr(env, "num_agents", 2)
        self.is_multiagent = True
        self.navigator_index = navigator_index
        self.observer_index = observer_index
        self.message_head = message_head
        self._channel_key, self._channel_slice, self._channel_scale = self._message_channel(env.observation_space)

    @staticmethod
    def _message_channel(observation_space):
        spaces = observation_space.spaces if isinstance(observation_space, gym.spaces.Dict) else {}
        meas_dim = LAVA_MAZE_MEASUREMENTS_SPACE.shape[0]
        if "measurements" in spaces and spaces["measurements"].shape == LAVA_MAZE_MEASUREMENTS_SPACE.shape:
            return "measurements", slice(0, meas_dim), np.ones(meas_dim, dtype=np.float32)
        if "measurements" not in spaces and "obs" in spaces and spaces["obs"].shape == (ego_state_dim(meas_dim),):
            return "obs", ego_state_measurement_slice(meas_dim), measurement_scale(LAVA_MAZE_MEASUREMENTS_SPACE)
        raise ValueError(
            f"LavaMazeOracleMessageWrapper has no message channel in observation space {observation_space}: needs "
            f"LavaMazeAdditionalInput measurements (pixel) or the ego state vector built from them (state)"
        )

    def _write_message(self, obs, message):
        nav_obs = dict(obs[self.navigator_index])
        vec = np.array(nav_obs[self._channel_key], dtype=np.float32, copy=True)
        raw = vec[self._channel_slice] * self._channel_scale
        vec[self._channel_slice] = _set_direction_one_hot(raw, message) / self._channel_scale
        nav_obs[self._channel_key] = vec
        obs = list(obs)
        obs[self.navigator_index] = nav_obs
        return obs

    def _message_from(self, actions):
        try:
            return int(np.asarray(actions[self.observer_index]).reshape(-1)[self.message_head])
        except (IndexError, TypeError, ValueError):
            return 0

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        return self._write_message(obs, 0), infos

    def step(self, actions):
        message = self._message_from(actions)
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        dones = [bool(t or tr) for t, tr in zip(terminated, truncated)]
        if dones and all(dones):
            # MultiAgentEnv already auto-reset: the returned obs start a new episode with an empty channel
            message = 0
        return self._write_message(obs, message), rewards, terminated, truncated, infos
