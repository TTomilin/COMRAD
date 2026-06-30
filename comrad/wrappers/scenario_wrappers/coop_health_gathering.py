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
        track_coop=True,
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
        self.episode_coop_steps = 0
        self.episode_defect_steps = 0
        self.track_coop = track_coop

    def _num_agents(self) -> int:
        return int(max(1, getattr(self.env.unwrapped, "num_agents", 2)))

    def _reward_share(self) -> float:
        """Team counters (USER3/USER4) are exposed to every player env, so divide."""
        return 1.0 / float(self._num_agents())

    def _player_id(self) -> int:
        return int(max(0, getattr(self.env.unwrapped, "player_id", 0)))

    def _own_pickup_key(self) -> str:
        return f"USER{60 + self._player_id()}"

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

        if self.track_coop:
            own_key = self._own_pickup_key()
            curr_own_pickups = info.get(own_key, 0)
            prev_own_pickups = self.prev_vars.get(own_key, 0) if self.prev_vars else 0
            delta_own_pickups = curr_own_pickups - prev_own_pickups
            delta_stretches_now = curr_chain_stretches - prev_chain_stretches
            coop = 1.0 if delta_own_pickups > 0 else 0.0
            defect = 1.0 if delta_stretches_now > 0 else 0.0

            info["coop_step_signal"] = coop
            info["defect_step_signal"] = defect

            self.episode_coop_steps += coop
            self.episode_defect_steps += defect

            if terminated or truncated:
                self._record_episode_stats(info)

        self.sync_vars(info)

        return obs, individual_reward, terminated, truncated, info

    def _record_episode_stats(self, info):
        extra = info.setdefault("episode_extra_stats", {})
        total = self.episode_coop_steps + self.episode_defect_steps
        extra["coop_steps"] = self.episode_coop_steps
        extra["defect_steps"] = self.episode_defect_steps
        extra["total_coop_defect_steps"] = total
        if total > 0:
            extra["cooperation_index"] = self.episode_coop_steps / total
            extra["defector_index"] = self.episode_defect_steps / total

    def reset(self, **kwargs):
        self.past_positions.clear()
        obs, info = self.env.reset(**kwargs)
        if "POSITION_X" in info and "POSITION_Y" in info:
            self.past_positions.append((info["POSITION_X"], info["POSITION_Y"]))
        self.sync_vars(info)
        self.orig_env_reward = 0.0
        self.ticks = 0
        self.episode_coop_steps = 0
        self.episode_defect_steps = 0
        return obs, info

    def sync_vars(self, info):
        self.prev_vars = {
            "HEALTH": info.get("HEALTH", 0.0),
            "USER3": info.get("USER3", 0),
            "USER4": info.get("USER4", 0),
        }
        own_key = self._own_pickup_key()
        self.prev_vars[own_key] = info.get(own_key, 0)
