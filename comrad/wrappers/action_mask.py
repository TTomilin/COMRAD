import gymnasium as gym
import numpy as np


def _flat_discrete_action_size(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n
    if isinstance(action_space, gym.spaces.Tuple):
        if not all(isinstance(space, gym.spaces.Discrete) for space in action_space.spaces):
            raise ValueError(f"Unsupported tuple action space for action masking: {action_space}")
        return sum(space.n for space in action_space.spaces)
    raise ValueError(f"Unsupported action space for action masking: {type(action_space)}")


class RoleActionMaskWrapper(gym.Wrapper):
    def __init__(self, env, masks_by_player_id, default_mask=None):
        super().__init__(env)
        self.action_mask_size = _flat_discrete_action_size(env.action_space)
        self.masks_by_player_id = {
            int(player_id): self._validate_mask(mask)
            for player_id, mask in masks_by_player_id.items()
        }
        self.default_mask = self._validate_mask(default_mask) if default_mask is not None else np.ones(
            self.action_mask_size, dtype=np.float32
        )

        if isinstance(env.observation_space, gym.spaces.Dict):
            spaces = dict(env.observation_space.spaces)
        else:
            spaces = {"obs": env.observation_space}
        spaces["action_mask"] = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.action_mask_size,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(spaces)

    def _validate_mask(self, mask):
        mask = np.asarray(mask, dtype=np.float32)
        if mask.shape != (self.action_mask_size,):
            raise ValueError(
                f"Action mask shape {mask.shape} does not match action mask size {self.action_mask_size}"
            )
        if np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError("Action masks must contain only 0.0 and 1.0")
        return mask

    def _current_mask(self):
        player_id = getattr(self.env.unwrapped, "player_id", -1)
        try:
            player_id = int(player_id)
        except (TypeError, ValueError):
            return self.default_mask
        return self.masks_by_player_id.get(player_id, self.default_mask)

    def _wrap_obs(self, obs):
        if obs is None:
            return None
        if isinstance(obs, dict):
            obs = dict(obs)
        else:
            obs = {"obs": obs}
        obs["action_mask"] = self._current_mask().copy()
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._wrap_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._wrap_obs(obs), reward, terminated, truncated, info
