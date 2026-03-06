import gymnasium as gym
from collections import deque

class LavaMazeRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        goal_reward=10.0,
        death_penalty=-5.0,
        step_penalty=-0.001,
        distance_reward_scale=0.1,
        lava_burn_penalty_scale=0.001,
    ):
        super().__init__(env)
        self.goal_reward = goal_reward
        self.death_penalty = death_penalty
        self.step_penalty = step_penalty
        self.distance_reward_scale = distance_reward_scale
        self.lava_burn_penalty_scale = lava_burn_penalty_scale

        self.prev_vars = {}
        self.orig_env_reward = 0.0

    def _decode_maze_grid(self, bits_0, bits_1, bits_2):
        """
        Reconstructs the 9x9 grid from the 3 ACS integer chunks.
        Returns a 9x9 2D list where 1 is safe, 0 is lava.
        """
        grid = [[0 for _ in range(9)] for _ in range(9)]
        chunks = [int(bits_0), int(bits_1), int(bits_2)]

        for y in range(9):
            for x in range(9):
                bit_idx = y * 9 + x
                chunk = bit_idx // 27
                shift = bit_idx % 27

                is_safe = (chunks[chunk] & (1 << shift)) != 0
                grid[y][x] = 1 if is_safe else 0

        return grid

    def _safe_int(self, value, default):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    def _get_bfs_distance(self, grid, start_x, start_y, goal_x, goal_y):
        """
        Calculates the shortest path distance using BFS.
        """
        start_x = self._safe_int(start_x, -1)
        start_y = self._safe_int(start_y, -1)
        goal_x = self._safe_int(goal_x, -1)
        goal_y = self._safe_int(goal_y, -1)

        if start_x == goal_x and start_y == goal_y:
            return 0

        if not (0 <= start_x < 9 and 0 <= start_y < 9) or grid[start_y][start_x] == 0:
            return 999 # Standing in lava

        queue = deque([(start_x, start_y, 0)])
        visited = set([(start_x, start_y)])

        while queue:
            cx, cy, dist = queue.popleft()

            if cx == goal_x and cy == goal_y:
                return dist

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy

                if (0 <= nx < 9 and 0 <= ny < 9 and
                    (nx, ny) not in visited and
                    grid[ny][nx] == 1): # safe floor

                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))

        return 999 # Fallback if path doesn't exist

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None or info is None:
            return obs, reward, terminated, truncated, info

        if not self.prev_vars:
            self.sync_vars(info)
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


        current_hp = self._safe_int(info.get("HEALTH", 0), 0)
        prev_hp = self._safe_int(self.prev_vars.get("HEALTH", 0), 0)

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
                    self._safe_int(info.get("USER21", 0), 0)
                )

                current_dist = self._get_bfs_distance(grid, p1_x, p1_y, goal_x, goal_y)
                prev_dist = self._get_bfs_distance(grid, prev_p1_x, prev_p1_y, goal_x, goal_y)

                if current_dist != 999 and prev_dist != 999:
                    dist_diff = prev_dist - current_dist
                    shaped_reward += dist_diff * self.distance_reward_scale

        individual_reward = reward + shaped_reward
        self.orig_env_reward += reward

        if terminated or truncated:
            info["true_objective"] = self.orig_env_reward

        self.sync_vars(info)
        return obs, individual_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.sync_vars(info)
        self.orig_env_reward = 0.0
        return obs, info

    def sync_vars(self, info):
        self.prev_vars = {
            "USER11": self._safe_int(info.get("USER11", 0), 0),
            "USER13": self._safe_int(info.get("USER13", -1), -1),
            "USER14": self._safe_int(info.get("USER14", -1), -1),
            "USER15": self._safe_int(info.get("USER15", -1), -1),
            "USER18": self._safe_int(info.get("USER18", -1), -1),
            "USER19": self._safe_int(info.get("USER19", 0), 0),
            "USER20": self._safe_int(info.get("USER20", 0), 0),
            "USER21": self._safe_int(info.get("USER21", 0), 0),
            "USER22": self._safe_int(info.get("USER22", 0), 0),
            "HEALTH": self._safe_int(info.get("HEALTH", 100), 100),
        }
