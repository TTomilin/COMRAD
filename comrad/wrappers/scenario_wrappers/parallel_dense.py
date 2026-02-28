"""
PS: I am keeping track of which zone agent is in for the sake of knowing when to reward and when not.
However, this doesnt break Dec-POMDP as it only matters to reward and critic is centralized anyways
"""

import gymnasium as gym


class ParallelReward(gym.Wrapper):
    shared = {}

    def __init__(
        self,
        env,
        *,
        next_zone: float = 1.0,
        back_zone: float = -1.0,
        rush_pen: float = -2.0,
        press_plate: float = 3.0,
        hold_plate: float = 0.1,
        leave_plate_early: float = -1.0,
        partner_next_zone: float = 20.0,
        done: float = 50.0,
        timeout: float = -5.0,
        last_zone: int = 5,
    ):
        super().__init__(env)

        self.next_zone = next_zone
        self.back_zone = back_zone
        self.rush_pen = rush_pen
        self.press_plate = press_plate
        self.hold_plate = hold_plate
        self.leave_plate_early = leave_plate_early
        self.partner_next_zone = partner_next_zone
        self.done = done
        self.timeout = timeout
        self.last_zone = last_zone

        self.prev_zone = 0
        self.max_zone = 0
        self.was_on_plate = False
        self.pnzwop = False  # Partner next zone while standing on plate
        self.plate_state = "0"  # 0: Not pressed yet -> 1: Pressed/ing -> 2: Rewarded

        worker_index = getattr(env.unwrapped, "worker_index", None)
        vector_index = getattr(env.unwrapped, "vector_index", None)
        self.ek = (worker_index, vector_index)
        self.pid = getattr(env.unwrapped, "player_id", None)
        self.orig_env_reward = 0.0

    def update_shared(self, zone):
        if self.ek not in ParallelReward.shared:
            ParallelReward.shared[self.ek] = {"agents": {}}
        ParallelReward.shared[self.ek]["agents"][self.pid] = {"zone": zone}

    def get_partner_zone(self) -> int | None:
        if self.ek not in ParallelReward.shared:
            return None
        agents = ParallelReward.shared[self.ek]["agents"]
        for agent_id, state in agents.items():
            if agent_id != self.pid:
                return state.get("zone")
        return None

    def clear_shared(self):
        if self.ek in ParallelReward.shared:
            shared = ParallelReward.shared[self.ek]
            if self.pid in shared["agents"]:
                del shared["agents"][self.pid]
            if not shared["agents"]:
                del ParallelReward.shared[self.ek]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_zone = info.get("USER1", 0)
        self.max_zone = self.prev_zone
        self.was_on_plate = False
        self.pnzwop = False
        self.plate_state = "0"
        self.orig_env_reward = 0.0
        self.clear_shared()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward is None:
            return obs, reward, terminated, truncated, info

        r = float(reward)
        zone = info.get("USER1", 0)
        on_plate = info.get("USER2", 0) >= 1
        pzone = self.get_partner_zone()

        if pzone is not None:
            # Reward when move to next zone, but it can't be 2 zones ahead of partner, and penalize if goes back
            if zone > self.prev_zone:
                if zone > self.max_zone:
                    if zone - pzone > 1:
                        r += self.rush_pen
                    else:
                        r += self.next_zone
                    self.max_zone = zone
                    self.pnzwop = False
                    self.plate_state = "0"
            elif zone < self.prev_zone:
                r += self.back_zone

            # Pressed plate, reward only once
            if on_plate and not self.was_on_plate and self.plate_state == "0":
                r += self.press_plate
                self.plate_state = "1"
                self.pnzwop = False

            # Press plate while partner is previous zone
            if on_plate and pzone < zone:
                r += self.hold_plate

            # Leave plate early
            if self.was_on_plate and not on_plate and self.plate_state == "1":
                if pzone < zone and not self.pnzwop:
                    r += self.leave_plate_early
                    # self.plate_state = '2'

            # Pressing plate when partner goes to enxt zone
            if on_plate and self.plate_state == "1":
                if pzone >= zone and not self.pnzwop:
                    r += self.partner_next_zone
                    self.pnzwop = True
                    self.plate_state = "2"

        self.update_shared(zone)
        self.prev_zone = zone
        self.was_on_plate = on_plate
        self.orig_env_reward += r

        if terminated or truncated:
            info["true_objective"] = self.orig_env_reward
            if zone >= self.last_zone:
                r += self.done
            elif truncated:
                r += self.timeout

        return obs, r, terminated, truncated, info
