import gymnasium as gym


class CoopPuzzleRewardShaping(gym.Wrapper):
    shared = {}

    def __init__(
        self,
        env,
        *,
        zone_advance_reward: float = 0.75,
        pair_completion_reward: float = 1.0,
        success_bonus: float = 2.0,
        timeout_penalty: float = -1.0,
        step_penalty: float = -0.002,
        last_zone: int = 10,
    ):
        super().__init__(env)
        self.zone_advance_reward = float(zone_advance_reward)
        self.pair_completion_reward = float(pair_completion_reward)
        self.success_bonus = float(success_bonus)
        self.timeout_penalty = float(timeout_penalty)
        self.step_penalty = float(step_penalty)
        self.last_zone = int(last_zone)

        self.prev_zone = 0
        self._ek = None
        self.pid = getattr(env.unwrapped, "player_id", None)
        self.orig_env_reward = 0.0

    @property
    def ek(self):
        if self._ek is None:
            worker_index = getattr(self.env.unwrapped, "worker_index", 0)
            vector_index = getattr(self.env.unwrapped, "vector_index", 0)
            self._ek = (worker_index, vector_index)
        return self._ek

    def _shared_state(self):
        return CoopPuzzleRewardShaping.shared.setdefault(
            self.ek,
            {
                "agents": {},
                "best_team_max_zone": 0,
                "best_completed_pairs": 0,
                "success": False,
            },
        )

    @staticmethod
    def _safe_int(info, key: str, default: int = 0) -> int:
        try:
            return int(info.get(key, default))
        except (TypeError, ValueError, OverflowError, AttributeError):
            return default

    def _completed_pairs(self, zone_a: int, zone_b: int) -> int:
        return max(0, min(zone_a, zone_b) // 2)

    def _partner_state(self):
        shared = CoopPuzzleRewardShaping.shared.get(self.ek)
        if shared is None:
            return None
        for agent_id, state in shared["agents"].items():
            if agent_id != self.pid:
                return state
        return None

    def _update_shared_agent(self, info):
        zone = self._safe_int(info, "USER1", 0)
        plate_state = self._safe_int(info, "USER2", 0)
        room_type = self._safe_int(info, "USER3", 0)
        self._shared_state()["agents"][self.pid] = {
            "zone": zone,
            "plate_state": plate_state,
            "room_type": room_type,
        }

    def _seed_shared_progress(self):
        shared = self._shared_state()
        me = shared["agents"].get(self.pid)
        if me is None:
            return

        zone = int(me.get("zone", 0))
        if zone > shared["best_team_max_zone"]:
            shared["best_team_max_zone"] = zone

        partner = self._partner_state()
        if partner is None:
            return

        partner_zone = int(partner.get("zone", 0))
        completed_pairs = self._completed_pairs(zone, partner_zone)
        if completed_pairs > shared["best_completed_pairs"]:
            shared["best_completed_pairs"] = completed_pairs

    def clear_shared(self):
        if self.ek not in CoopPuzzleRewardShaping.shared:
            return
        shared = CoopPuzzleRewardShaping.shared[self.ek]
        shared["agents"].pop(self.pid, None)
        if not shared["agents"]:
            CoopPuzzleRewardShaping.shared.pop(self.ek, None)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_zone = self._safe_int(info, "USER1", 0)
        self.orig_env_reward = 0.0
        self.clear_shared()
        self._update_shared_agent(info)
        self._seed_shared_progress()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward is None:
            return obs, reward, terminated, truncated, info

        env_reward = float(reward)
        self.orig_env_reward += env_reward

        if info is None:
            return obs, env_reward, terminated, truncated, info

        zone = self._safe_int(info, "USER1", 0)
        partner = self._partner_state()
        shaped_reward = self.step_penalty
        success = False

        if partner is not None:
            partner_zone = int(partner.get("zone", 0))
            shared = self._shared_state()

            team_max_zone = max(zone, partner_zone)
            if team_max_zone > shared["best_team_max_zone"]:
                shaped_reward += (team_max_zone - shared["best_team_max_zone"]) * self.zone_advance_reward
                shared["best_team_max_zone"] = team_max_zone

            completed_pairs = self._completed_pairs(zone, partner_zone)
            if completed_pairs > shared["best_completed_pairs"]:
                shaped_reward += (completed_pairs - shared["best_completed_pairs"]) * self.pair_completion_reward
                shared["best_completed_pairs"] = completed_pairs

            success = bool(
                terminated
                and not truncated
                and zone >= self.last_zone
                and partner_zone >= self.last_zone
            )
            if success and not shared["success"]:
                shaped_reward += self.success_bonus
                shared["success"] = True

            info["true_objective"] = float(shared["best_completed_pairs"])
            info["success"] = shared["success"]
        else:
            info["true_objective"] = 0.0
            info["success"] = False

        if truncated and not success:
            shaped_reward += self.timeout_penalty

        self._update_shared_agent(info)
        self.prev_zone = zone

        if terminated or truncated:
            info["orig_env_reward"] = self.orig_env_reward

        return obs, env_reward + shaped_reward, terminated, truncated, info
