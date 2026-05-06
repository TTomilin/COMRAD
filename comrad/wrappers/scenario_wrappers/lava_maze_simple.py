import gymnasium as gym
from collections import deque
import numpy as np


class LavaMazeSimpleRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        grid_size=18,
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

        self.flash_window = 5
        self.sender_bonus_scale = 0.2
        self.no_progress_penalty = -0.005
        self.flash_timer = 0
        self.progress_occurred = False

        self.prev_vars = {}
        self.orig_env_reward = 0.0

    def _safe_int(self, value, default):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    def sync_vars(self, info):
        self.prev_vars = {k: v for k, v in info.items()}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_vars = {}
        self.flash_timer = 0
        self.progress_occurred = False
        self.orig_env_reward = 0.0
        
        if info is not None:
            self.sync_vars(info)
            
        return obs, info

    def _decode_maze_grid(self, bits_0, bits_1, bits_2):
        """
        Reconstructs the grid from the 3 ACS integer chunks.
        Returns a 2D list where 1 is safe, 0 is lava.
        In this version, it's a 1D corridor (y=0 used for the path).
        """
        s = self.grid_size
        grid = [[0 for _ in range(s)] for _ in range(s)]
        chunks = [int(bits_0), int(bits_1), int(bits_2)]

        for x in range(s):
            bit_idx = x
            chunk = bit_idx // 27
            shift = bit_idx % 27

            is_safe = (chunks[chunk] & (1 << shift)) != 0
            grid[0][x] = 1 if is_safe else 0

        return grid

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

        s = self.grid_size
        if not (0 <= start_x < s and 0 <= start_y < s) or grid[start_y][start_x] == 0:
            return 999 # Standing in lava

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
                    grid[ny][nx] == 1): # safe floor

                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))

        return 999 

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
        prev_goal_x = self._safe_int(self.prev_vars.get("USER13", -1), -1)
        goal_y = self._safe_int(info.get("USER18", -1), -1)
        prev_goal_y = self._safe_int(self.prev_vars.get("USER18", -1), -1)

        current_levels = self._safe_int(info.get("USER22", 0), 0)
        prev_levels = self._safe_int(self.prev_vars.get("USER22", 0), 0)

        current_signal = self._safe_int(info.get("USER17", 0), 0)
        prev_signal = self._safe_int(self.prev_vars.get("USER17", 0), 0)

        if current_signal == 1 and prev_signal == 0:
            shaped_reward += self.signal_penalty

        grid = self._decode_maze_grid(
            self._safe_int(info.get("USER19", 0), 0),
            self._safe_int(info.get("USER20", 0), 0),
            self._safe_int(info.get("USER21", 0), 0)
        )

        current_hp = self._safe_int(info.get("HEALTH", 0), 0)
        prev_hp = self._safe_int(self.prev_vars.get("HEALTH", 0), 0)

        player_id = getattr(self.env.unwrapped, "player_id", -1)
        if player_id == 0:
            if current_signal == 1 and prev_signal == 0:
                self.flash_timer = self.flash_window
                self.progress_occurred = False

            if self.flash_timer > 0:
                self.flash_timer -= 1
                if (p1_x != -1 and p1_y != -1 and goal_x != -1 and goal_y != -1 and
                    prev_p1_x != -1 and prev_p1_y != -1 and
                    goal_x == prev_goal_x and goal_y == prev_goal_y):
                    
                    dist = self._get_bfs_distance(grid, p1_x, p1_y, goal_x, goal_y)
                    prev_dist = self._get_bfs_distance(grid, prev_p1_x, prev_p1_y, prev_goal_x, prev_goal_y)
                    if dist < prev_dist:
                        self.progress_occurred = True
                    
            if current_levels > prev_levels:
                shaped_reward += self.goal_reward
                
            if p1_x != -1 and goal_x != -1 and prev_p1_x != -1 and goal_x == prev_goal_x:
                dist = self._get_bfs_distance(grid, p1_x, p1_y, goal_x, goal_y)
                prev_dist = self._get_bfs_distance(grid, prev_p1_x, prev_p1_y, prev_goal_x, prev_goal_y)
                if dist < prev_dist:
                    shaped_reward += self.distance_reward_scale
                elif dist > prev_dist:
                    shaped_reward -= self.distance_reward_scale

            if current_hp < prev_hp:
                hp_loss = prev_hp - current_hp
                if hp_loss > 20: 
                    shaped_reward -= (hp_loss * self.lava_burn_penalty_scale)
            
            if current_hp <= 0 and prev_hp > 0:
                shaped_reward += self.death_penalty
                
        elif player_id == 1:
            if current_levels > prev_levels:
                shaped_reward += self.goal_reward

            if current_hp < prev_hp:
                hp_loss = prev_hp - current_hp
                if hp_loss > 20: 
                    shaped_reward -= (hp_loss * self.lava_burn_penalty_scale)

            if p1_x != -1 and goal_x != -1 and prev_p1_x != -1 and goal_x == prev_goal_x:
                dist = self._get_bfs_distance(grid, p1_x, p1_y, goal_x, goal_y)
                prev_dist = self._get_bfs_distance(grid, prev_p1_x, prev_p1_y, prev_goal_x, prev_goal_y)
                
                if dist < prev_dist:
                    if self.flash_timer > 0:
                        shaped_reward += self.sender_bonus_scale
                    shaped_reward += self.distance_reward_scale
                elif dist > prev_dist:
                    shaped_reward -= self.distance_reward_scale
                    if self.flash_timer > 0:
                        shaped_reward += self.no_progress_penalty
                        
            if current_signal == 1 and prev_signal == 0:
                self.flash_timer = self.flash_window
            
            if self.flash_timer > 0:
                self.flash_timer -= 1
                
            if current_hp <= 0 and prev_hp > 0:
                shaped_reward += self.death_penalty
                
        else:
            if current_levels > prev_levels:
                shaped_reward += self.goal_reward

            if current_hp < prev_hp:
                hp_loss = prev_hp - current_hp
                if hp_loss > 20:
                    shaped_reward -= (hp_loss * self.lava_burn_penalty_scale)

            if p1_x != -1 and goal_x != -1 and prev_p1_x != -1 and goal_x == prev_goal_x:
                dist = self._get_bfs_distance(grid, p1_x, p1_y, goal_x, goal_y)
                prev_dist = self._get_bfs_distance(grid, prev_p1_x, prev_p1_y, prev_goal_x, prev_goal_y)
                if dist < prev_dist:
                    shaped_reward += self.distance_reward_scale
                elif dist > prev_dist:
                    shaped_reward -= self.distance_reward_scale
                    
            if current_hp <= 0 and prev_hp > 0:
                shaped_reward += self.death_penalty

        self.orig_env_reward += reward
        self.sync_vars(info)
        return obs, shaped_reward, terminated, truncated, info

class LavaMazeSimpleAdditionalInput(gym.Wrapper):
    """
    health, signal, weapons
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            "obs": env.observation_space,
            "measurements": gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        })

    def _get_obs(self, obs, info):
        hp = info.get("HEALTH", 100) / 100.0
        signal = info.get("USER17", 0)
        color = info.get("USER16", 0) # 0 for Green, 1 for Red

        measurements = np.array([hp, signal, color], dtype=np.float32)
        return {"obs": obs, "measurements": measurements}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_obs(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_obs(obs, info), reward, terminated, truncated, info
