from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

from comrad.wrappers.episode_boundary import joint_new_episode


def zero_action(action_space: gym.Space):
    """No-op action: 0th action of every discrete head, zeros for continuous heads."""
    if isinstance(action_space, gym.spaces.Tuple):
        return tuple(zero_action(space) for space in action_space.spaces)
    if isinstance(action_space, gym.spaces.Discrete):
        return 0
    if isinstance(action_space, gym.spaces.Box):
        return np.zeros(action_space.shape, dtype=action_space.dtype)
    raise TypeError(f"Unsupported action space for zero_action: {action_space}")


class NoopPartnerPolicy:
    """Partner that never acts. Used when the partner is driven by ACS inside the WAD (e.g. scripted scout)."""

    def __init__(self, action_space: gym.Space, **kwargs):
        self.action_space = action_space
        self._noop = zero_action(action_space)

    def reset(self) -> None:
        pass

    def act(self, obs: Any, info: Optional[dict]):
        return self._noop


class ScriptedPartnerWrapper(gym.Wrapper):
    """
    Hides one player of a 2-player MultiAgentEnv behind a fixed scripted policy.
    Sample Factory only sees the learner (num_agents=1, list-of-one I/O), while the Doom game still runs both players.

    `partner_policy` is a factory (class or callable) invoked with `action_space=` so every env copy gets its own
    stateful policy instance.

    `scripted_players` is read by DoomSpec: the game runs learner agents + scripted players, while models and
    --num_agents count learners only. make_doom_multiplayer_env applies SharedRewardWrapper (on-policy algos) *before*
    this wrapper, so the learner is trained on the same team/individual reward as in the two-learner benchmark.
    """

    scripted_players = 1

    def __init__(
        self,
        env,
        *,
        learner_index: int,
        partner_policy: Callable[..., Any],
        partner_kwargs: Optional[dict] = None,
    ):
        super().__init__(env)
        inner_num_agents = getattr(env, "num_agents", 1)
        if inner_num_agents != 2:
            raise ValueError(f"ScriptedPartnerWrapper expects a 2-agent env, got num_agents={inner_num_agents}")
        if learner_index not in (0, 1):
            raise ValueError(f"learner_index must be 0 or 1, not {learner_index}")

        self.learner_index = learner_index
        self.partner_index = 1 - learner_index
        self.num_agents = 1
        self.is_multiagent = True

        self.partner = partner_policy(action_space=env.action_space, **(partner_kwargs or {}))
        self._partner_obs = None
        self._partner_info: Optional[dict] = None

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        self.partner.reset()
        self._partner_obs = obs[self.partner_index]
        self._partner_info = infos[self.partner_index] if infos is not None else None
        learner_info = infos[self.learner_index] if infos is not None else {}
        return [obs[self.learner_index]], [learner_info]

    def step(self, actions):
        joint_actions = [None, None]
        joint_actions[self.learner_index] = actions[0]
        joint_actions[self.partner_index] = self.partner.act(self._partner_obs, self._partner_info)

        obs, rewards, terminated, truncated, infos = self.env.step(joint_actions)
        new_episode = joint_new_episode(terminated, truncated, infos)

        l, p = self.learner_index, self.partner_index
        learner_done = bool(terminated[l] or truncated[l])
        partner_done = bool(terminated[p] or truncated[p])

        if learner_done and not partner_done:
            # Episodes end jointly in these scenarios; if not, force the joint reset so the learner never
            # continues in a stale episode (MultiAgentEnv only auto-resets on all(dones)).
            obs, reset_infos = self.env.reset()
            infos[l].setdefault("reset_info", reset_infos[l])
            partner_info = reset_infos[p]
            partner_done = True
        elif partner_done:
            partner_info = infos[p].get("reset_info", infos[p])
        else:
            partner_info = infos[p]

        if partner_done or new_episode is not None:
            self.partner.reset()

        self._partner_obs = obs[p]
        self._partner_info = partner_info

        return [obs[l]], [rewards[l]], [terminated[l]], [truncated[l]], [infos[l]]
