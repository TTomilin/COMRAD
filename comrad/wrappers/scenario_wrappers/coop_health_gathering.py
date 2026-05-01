import gymnasium as gym
import collections
import math

class CoopHealthGatheringRewardShaping(gym.Wrapper):
    """Reward shaping for cooperative health gathering scenario."""

    def __init__(
        self,
        env,
        health_reward=0.1,
        health_loss_penalty=-0.05,
        coop_pickup_reward=0.1,
        death_penalty=-3.0,
        chain_penalty=-0.05,
        exploration_reward=0,
        exploration_distance=80.0,
        exploration_steps=200,
    ):
        super().__init__(env)
        self.health_reward = health_reward
        self.health_loss_penalty = health_loss_penalty
        self.coop_pickup_reward = coop_pickup_reward
        self.death_penalty = death_penalty
        self.chain_penalty = chain_penalty
        self.exploration_reward = exploration_reward
        self.exploration_distance = exploration_distance
        self.exploration_steps = exploration_steps

        self.past_positions = collections.deque(maxlen=exploration_steps)

        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self.ticks = 0

    def _num_agents(self) -> int:
        return int(max(1, getattr(self.env.unwrapped, "num_agents", 2)))

    def _reward_share(self) -> float:
        """Team counters (USER3/USER4) are exposed to every player env, so divide."""
        return 1.0 / float(self._num_agents())

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None or info is None:
            return obs, reward, terminated, truncated, info

        self.ticks += 1

        if not self.prev_vars:
            self.sync_vars(info)
            return obs, 0.0, terminated, truncated, info

        shaped_reward = 0.0
        share = self._reward_share()

        # Health gain
        curr_health = info.get("HEALTH", 0.0)
        prev_health = self.prev_vars.get("HEALTH", 0.0)
        delta_health = curr_health - prev_health
        if delta_health > 0.0:
            shaped_reward += self.health_reward * delta_health

        # Health loss
        elif delta_health < 0.0:
            shaped_reward += self.health_loss_penalty * abs(delta_health)

        # Death penalty
        if curr_health <= 0.0 and prev_health > 0.0:
            shaped_reward += self.death_penalty

        # Global pick ups
        curr_picked_up_kits = info.get("USER4", 0)
        prev_picked_up_kits = self.prev_vars.get("USER4", 0)

        delta_pickups = curr_picked_up_kits - prev_picked_up_kits
        if delta_pickups > 0:
            shaped_reward += self.coop_pickup_reward * delta_pickups * share

        # Chain stretches
        curr_chain_stretches = info.get("USER3", 0)
        prev_chain_stretches = self.prev_vars.get("USER3", 0)
        delta_stretches = curr_chain_stretches - prev_chain_stretches
        if delta_stretches > 0:
            shaped_reward += self.chain_penalty * delta_stretches * share

        # Exploration reward
        curr_x = info.get("POSITION_X")
        curr_y = info.get("POSITION_Y")

        if curr_x is not None and curr_y is not None:
            if len(self.past_positions) == self.exploration_steps:
                past_x, past_y = self.past_positions[0]
                dist = math.sqrt((curr_x - past_x)**2 + (curr_y - past_y)**2)
                if dist >= self.exploration_distance:
                    shaped_reward += self.exploration_reward
            self.past_positions.append((curr_x, curr_y))

        individual_reward = reward + shaped_reward
        self.orig_env_reward += reward

        if terminated or truncated:
            info["true_objective"] = self.ticks

        self.sync_vars(info)
        return obs, individual_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.past_positions.clear()
        obs, info = self.env.reset(**kwargs)
        if "POSITION_X" in info and "POSITION_Y" in info:
            self.past_positions.append((info["POSITION_X"], info["POSITION_Y"]))
        self.sync_vars(info)
        self.orig_env_reward = 0.0
        self.ticks = 0
        return obs, info

    def sync_vars(self, info):
        self.prev_vars = {
            "HEALTH": info.get("HEALTH", 0.0),
            "USER3": info.get("USER3", 0),
            "USER4": info.get("USER4", 0),
        }
