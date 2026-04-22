import gymnasium as gym


class PlatformChainRewardShaping(gym.Wrapper):
	def __init__(
		self,
		env,
		progress_reward_per_level=1.0,
		chain_break_penalty=-0.05,
		drag_penalty=0.0,
	):
		super().__init__(env)
		self.progress_reward_per_level = float(progress_reward_per_level)
		self.chain_break_penalty = float(chain_break_penalty)
		self.drag_penalty = float(drag_penalty)

		self.prev_chain_break_events = 0
		self.best_min_level = 0
		self.orig_env_reward = 0.0

	def _num_agents(self) -> int:
		return int(max(1, getattr(self.env.unwrapped, "num_agents", 2)))

	def _reward_share(self) -> float:
		# Per-agent shaping is scaled so off-policy reward summation reconstructs team reward.
		return 1.0 / float(self._num_agents())

	@staticmethod
	def _int_stat(info, key: str, default: int = 0) -> int:
		return int(max(0, float(info.get(key, default))))

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		self.orig_env_reward = 0.0

		if info is None:
			self.prev_chain_break_events = 0
			self.best_min_level = 0
			return obs, info

		self.prev_chain_break_events = self._int_stat(info, "USER52")
		self.best_min_level = self._int_stat(info, "USER54")
		return obs, info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)

		if reward is None:
			reward = 0.0
		reward = float(reward)
		self.orig_env_reward += reward

		if info is None:
			return obs, reward, terminated, truncated, info

		curr_min_level = self._int_stat(info, "USER54")
		prev_best_min_level = self.best_min_level
		if curr_min_level > self.best_min_level:
			self.best_min_level = curr_min_level

		curr_break_events = self._int_stat(info, "USER52")
		curr_dragged_links = self._int_stat(info, "USER53")

		shaped_team_reward = 0.0

		delta_best_min_level = self.best_min_level - prev_best_min_level
		if delta_best_min_level > 0:
			shaped_team_reward += delta_best_min_level * self.progress_reward_per_level

		delta_chain_breaks = max(0, curr_break_events - self.prev_chain_break_events)
		if delta_chain_breaks > 0 and self.chain_break_penalty != 0.0:
			shaped_team_reward += delta_chain_breaks * self.chain_break_penalty

		if curr_dragged_links > 0 and self.drag_penalty != 0.0:
			shaped_team_reward += curr_dragged_links * self.drag_penalty

		total_reward = reward + shaped_team_reward * self._reward_share()

		info["true_objective"] = float(self.best_min_level)
		if terminated or truncated:
			info["orig_env_reward"] = self.orig_env_reward

		self.prev_chain_break_events = curr_break_events
		return obs, total_reward, terminated, truncated, info
