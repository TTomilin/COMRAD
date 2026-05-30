import gymnasium as gym
from collections import deque
import numpy as np


class LavaMazeSimpleRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        grid_size=18,
        goal_reward=2.0,
        death_penalty=-5.0,
        step_penalty=-0.005,
        distance_reward_scale=0.1,
        lava_burn_penalty_scale=0.02,
        signal_penalty=-0.01,
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
        self.flash_timer = 0
        self.current_goal = None
        self.best_distance = None

        self.prev_vars = {}
        self.orig_env_reward = 0.0

    def _safe_int(self, value, default):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    def sync_vars(self, info):
        self.prev_vars = {k: v for k, v in info.items()}

    def _seed_initial_prev_vars(self, info):
        baseline = {k: v for k, v in info.items()}
        baseline["USER16"] = self._safe_int(info.get("USER16", 0), 0)
        baseline["USER17"] = self._safe_int(info.get("USER17", 0), 0)
        baseline["USER22"] = self._safe_int(info.get("USER22", 0), 0)
        baseline["HEALTH"] = self._safe_int(info.get("HEALTH", 100), 100)
        self.prev_vars = baseline

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.flash_timer = 0
        self.current_goal = None
        self.best_distance = None
        self.orig_env_reward = 0.0

        if info is not None:
            self.sync_vars(info)
            self._reset_frontier(info)
        else:
            self.prev_vars = {}

        return obs, info

    def _decode_maze_grid(self, bits_0, bits_1, bits_2, active_y):
        s = self.grid_size
        grid = [[0 for _ in range(s)] for _ in range(s)]
        chunks = [int(bits_0), int(bits_1), int(bits_2)]

        for x in range(s):
            bit_idx = x
            chunk = bit_idx // 27
            shift = bit_idx % 27

            is_safe = (chunks[chunk] & (1 << shift)) != 0
            if 0 <= active_y < s:
                grid[active_y][x] = 1 if is_safe else 0

        return grid

    def _get_bfs_distance(self, grid, start_x, start_y, goal_x, goal_y):
        start_x = self._safe_int(start_x, -1)
        start_y = self._safe_int(start_y, -1)
        goal_x = self._safe_int(goal_x, -1)
        goal_y = self._safe_int(goal_y, -1)

        if start_x == goal_x and start_y == goal_y:
            return 0

        s = self.grid_size
        if not (0 <= start_x < s and 0 <= start_y < s) or grid[start_y][start_x] == 0:
            return 999

        queue = deque([(start_x, start_y, 0)])
        visited = set([(start_x, start_y)])

        while queue:
            cx, cy, dist = queue.popleft()

            if cx == goal_x and cy == goal_y:
                return dist

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy

                if (0 <= nx < s and 0 <= ny < s and
                    (nx, ny) not in visited and
                    grid[ny][nx] == 1):

                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))

        return 999

    def _distance_from_info(self, info):
        p1_x = self._safe_int(info.get("USER14", -1), -1)
        p1_y = self._safe_int(info.get("USER15", -1), -1)
        goal_x = self._safe_int(info.get("USER13", -1), -1)
        goal_y = self._safe_int(info.get("USER18", -1), -1)

        grid = self._decode_maze_grid(
            self._safe_int(info.get("USER19", 0), 0),
            self._safe_int(info.get("USER20", 0), 0),
            self._safe_int(info.get("USER21", 0), 0),
            goal_y,
        )
        if p1_x == -1 or p1_y == -1 or goal_x == -1 or goal_y == -1:
            return None, 999

        return (goal_x, goal_y), self._get_bfs_distance(grid, p1_x, p1_y, goal_x, goal_y)

    def _reset_frontier(self, info):
        goal, dist = self._distance_from_info(info)
        self.current_goal = goal
        self.best_distance = dist if goal is not None and dist < 999 else None

    def _frontier_improvement(self, goal, dist):
        if goal is None or dist >= 999:
            return 0

        if self.current_goal != goal:
            self.current_goal = goal
            self.best_distance = dist
            return 0

        if self.best_distance is None:
            self.best_distance = dist
            return 0

        if dist < self.best_distance:
            improvement = self.best_distance - dist
            self.best_distance = dist
            return improvement

        return 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None or info is None:
            return obs, reward, terminated, truncated, info

        if not self.prev_vars:
            self._seed_initial_prev_vars(info)
            self._reset_frontier(info)
            return obs, 0.0, terminated, truncated, info

        shaped_reward = self.step_penalty

        goal_x = self._safe_int(info.get("USER13", -1), -1)
        prev_goal_x = self._safe_int(self.prev_vars.get("USER13", -1), -1)
        goal_y = self._safe_int(info.get("USER18", -1), -1)
        prev_goal_y = self._safe_int(self.prev_vars.get("USER18", -1), -1)

        current_levels = self._safe_int(info.get("USER22", 0), 0)
        prev_levels = self._safe_int(self.prev_vars.get("USER22", 0), 0)

        current_signal = self._safe_int(info.get("USER16", 0), 0)
        prev_signal = self._safe_int(self.prev_vars.get("USER16", 0), 0)

        current_flash_active = self._safe_int(info.get("USER17", 0), 0)
        prev_flash_active = self._safe_int(self.prev_vars.get("USER17", 0), 0)

        current_hp = self._safe_int(info.get("HEALTH", 0), 0)
        prev_hp = self._safe_int(self.prev_vars.get("HEALTH", 0), 0)

        goal, dist = self._distance_from_info(info)
        same_goal = goal_x == prev_goal_x and goal_y == prev_goal_y
        improvement = self._frontier_improvement(goal, dist) if same_goal else 0

        player_id = getattr(self.env.unwrapped, "player_id", -1)
        if player_id == 0:
            if current_flash_active == 1 and prev_flash_active == 0:
                self.flash_timer = self.flash_window

            if self.flash_timer > 0:
                self.flash_timer -= 1

            if current_levels > prev_levels:
                shaped_reward += self.goal_reward
                self._reset_frontier(info)

            if improvement > 0:
                shaped_reward += improvement * self.distance_reward_scale

            if current_hp < prev_hp:
                hp_loss = prev_hp - current_hp
                if hp_loss > 20:
                    shaped_reward -= (hp_loss * self.lava_burn_penalty_scale)

            if current_hp <= 0 and prev_hp > 0:
                shaped_reward += self.death_penalty

        elif player_id == 1:
            if current_signal != 0 and prev_signal == 0:
                shaped_reward += self.signal_penalty

            if current_flash_active == 1 and prev_flash_active == 0:
                self.flash_timer = self.flash_window

            if current_levels > prev_levels:
                shaped_reward += self.goal_reward
                self._reset_frontier(info)

            if current_hp < prev_hp:
                hp_loss = prev_hp - current_hp
                if hp_loss > 20:
                    shaped_reward -= (hp_loss * self.lava_burn_penalty_scale)

            if improvement > 0:
                if self.flash_timer > 0:
                    shaped_reward += improvement * self.sender_bonus_scale
                shaped_reward += improvement * self.distance_reward_scale

            if self.flash_timer > 0:
                self.flash_timer -= 1

            if current_hp <= 0 and prev_hp > 0:
                shaped_reward += self.death_penalty

        else:
            if current_levels > prev_levels:
                shaped_reward += self.goal_reward
                self._reset_frontier(info)

            if current_hp < prev_hp:
                hp_loss = prev_hp - current_hp
                if hp_loss > 20:
                    shaped_reward -= (hp_loss * self.lava_burn_penalty_scale)

            if improvement > 0:
                shaped_reward += improvement * self.distance_reward_scale

            if current_hp <= 0 and prev_hp > 0:
                shaped_reward += self.death_penalty

        self.orig_env_reward += reward

        if terminated or truncated:
            info["true_objective"] = current_levels

        self.sync_vars(info)
        return obs, shaped_reward, terminated, truncated, info


class LavaMazeSimpleAdditionalInput(gym.Wrapper):
    """
    health, flash_active, active color (green=0.0, red=1.0)
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            "obs": env.observation_space,
            "measurements": gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        })

    def _get_obs(self, obs, info):
        if info is None:
            return {"obs": obs, "measurements": np.zeros(3, dtype=np.float32)}

        hp = info.get("HEALTH", 100) / 100.0
        flash_active = float(bool(info.get("USER17", 0)))
        raw_color = int(info.get("USER16", 0))
        color = 1.0 if raw_color == 2 else 0.0

        measurements = np.array([hp, flash_active, color], dtype=np.float32)
        return {"obs": obs, "measurements": measurements}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_obs(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if obs is None:
            return obs, reward, terminated, truncated, info
        return self._get_obs(obs, info), reward, terminated, truncated, info
