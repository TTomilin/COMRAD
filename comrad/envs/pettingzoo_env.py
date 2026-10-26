"""PettingZoo `ParallelEnv` adapter around the list-based COMRAD multi-agent env."""

from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv


def noop_action(space: gym.Space):
    """All-zero action: index 0 of every Discrete head (no button pressed) and 0.0 for continuous heads."""
    if isinstance(space, gym.spaces.Tuple):
        return tuple(noop_action(s) for s in space.spaces)
    if isinstance(space, gym.spaces.Dict):
        return {key: noop_action(s) for key, s in space.spaces.items()}
    if isinstance(space, gym.spaces.Discrete):
        return int(space.start)
    if isinstance(space, gym.spaces.Box):
        return np.clip(np.zeros(space.shape, dtype=space.dtype), space.low, space.high)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, gym.spaces.MultiBinary):
        return np.zeros(space.shape, dtype=space.dtype)
    raise TypeError(f"Unsupported action space {space}")


class ComradParallelEnv(ParallelEnv):
    """Exposes a COMRAD scenario as a PettingZoo `ParallelEnv`.

    Agents are named `player_0 .. player_{N-1}`. An agent that terminates or truncates is removed from `agents`
    and stops receiving observations and rewards; the adapter keeps sending it a no-op action so the underlying
    multiplayer game stays in sync until every agent is done. Call `reset()` to start the next episode.
    """

    metadata = {"name": "comrad_v0", "render_modes": ["human", "rgb_array"], "is_parallelizable": True}

    def __init__(self, env: gym.Env, scenario: str = ""):
        self.env = env
        self.scenario = scenario
        num_agents = env.unwrapped.num_agents
        self.possible_agents: List[str] = [f"player_{i}" for i in range(num_agents)]
        self.agents: List[str] = []
        self.render_mode = getattr(env.unwrapped, "render_mode", None)

        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._noop = noop_action(self._action_space)

    def observation_space(self, agent: str) -> gym.Space:
        return self._observation_space

    def action_space(self, agent: str) -> gym.Space:
        return self._action_space

    def reset(self, seed=None, options=None):
        kwargs: Dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        if options is not None:
            kwargs["options"] = options
        obs, infos = self.env.reset(**kwargs)
        self.agents = list(self.possible_agents)
        return dict(zip(self.agents, obs)), {agent: dict(info) for agent, info in zip(self.agents, infos)}

    def step(self, actions: Dict[str, Any]):
        if not self.agents:
            raise RuntimeError("All agents are done, call reset() before step()")

        live = set(self.agents)
        action_list = [
            actions.get(agent, self._noop) if agent in live else self._noop for agent in self.possible_agents
        ]
        obs, rewards, terminated, truncated, infos = self.env.step(action_list)

        # Crash recovery reset the game underneath us: the current episode is over for everyone.
        hidden_reset = any(isinstance(info, dict) and info.get("_hidden_reset_count") for info in infos)

        out_obs, out_rew, out_term, out_trunc, out_info = {}, {}, {}, {}, {}
        for i, agent in enumerate(self.possible_agents):
            if agent not in live:
                continue
            out_obs[agent] = obs[i]
            out_rew[agent] = float(rewards[i])
            out_term[agent] = bool(terminated[i])
            out_trunc[agent] = bool(truncated[i]) or (hidden_reset and not out_term[agent])
            out_info[agent] = infos[i]

        self.agents = [agent for agent in self.agents if not (out_term[agent] or out_trunc[agent])]
        return out_obs, out_rew, out_term, out_trunc, out_info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def state(self):
        raise NotImplementedError("COMRAD does not expose a global state; use the per-agent observations")
