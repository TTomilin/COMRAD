import gymnasium as gym
import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from comrad import api
from comrad.envs.pettingzoo_env import ComradParallelEnv, noop_action

ACTION_SPACE = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(2)))
OBS_SPACE = gym.spaces.Box(0, 255, (4, 4, 3), dtype=np.uint8)


class FakeListEnv(gym.Env):
    """Mimics MultiAgentEnv(auto_reset=False): per-agent lists, agent i is done after episode_lens[i] steps."""

    observation_space = OBS_SPACE
    action_space = ACTION_SPACE

    def __init__(self, episode_lens=(3, 5), render_mode=None):
        self.episode_lens = list(episode_lens)
        self.num_agents = len(self.episode_lens)
        self.render_mode = render_mode
        self.t = 0
        self.actions = []
        self.hidden_reset_at = None
        self.last_seed = None

    def _obs(self):
        return [np.full((4, 4, 3), self.t, dtype=np.uint8) for _ in range(self.num_agents)]

    def reset(self, seed=None, options=None):
        self.t = 0
        self.last_seed = seed
        return self._obs(), [{"agent": i} for i in range(self.num_agents)]

    def step(self, actions):
        assert len(actions) == self.num_agents
        self.actions.append(list(actions))
        self.t += 1
        rewards = [float(i + 1) for i in range(self.num_agents)]
        terminated = [self.t >= n for n in self.episode_lens]
        truncated = [False] * self.num_agents
        infos = [{"agent": i} for i in range(self.num_agents)]
        for i, done in enumerate(terminated):
            if done:
                infos[i]["true_objective"] = 7
        if self.hidden_reset_at == self.t:
            # MultiAgentEnv(auto_reset=False) reports a crash-recovery reset as truncation of live agents
            truncated = [not term for term in terminated]
            for info in infos:
                info["_hidden_reset_count"] = 1
        return self._obs(), rewards, terminated, truncated, infos

    def render(self):
        return np.zeros((4, 8, 3), dtype=np.uint8)


def test_noop_action_handles_discrete_and_box_heads():
    space = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Box(-1.0, 1.0, (1,), np.float32)))
    noop = noop_action(space)
    assert noop[0] == 0
    assert np.array_equal(noop[1], np.zeros(1, dtype=np.float32))
    assert space.contains(noop)


def test_agent_names_and_spaces():
    env = ComradParallelEnv(FakeListEnv(episode_lens=(3, 3, 3)))
    assert env.possible_agents == ["player_0", "player_1", "player_2"]
    assert env.observation_space("player_1") is OBS_SPACE
    assert env.action_space("player_2") is ACTION_SPACE


def test_reset_returns_dicts_and_forwards_seed():
    fake = FakeListEnv()
    env = ComradParallelEnv(fake)
    obs, infos = env.reset(seed=123)
    assert set(obs) == set(infos) == {"player_0", "player_1"}
    assert env.agents == ["player_0", "player_1"]
    assert fake.last_seed == 123


def test_finished_agent_is_removed_and_gets_noop_without_rewards():
    fake = FakeListEnv(episode_lens=(2, 4))
    env = ComradParallelEnv(fake)
    env.reset()
    action = (2, 1)

    for _ in range(2):
        obs, rewards, term, trunc, infos = env.step({agent: action for agent in env.agents})
    assert term == {"player_0": True, "player_1": False}
    assert infos["player_0"]["true_objective"] == 7
    assert env.agents == ["player_1"]

    obs, rewards, term, trunc, infos = env.step({"player_1": action})
    assert set(rewards) == {"player_1"}, "finished agents must not receive rewards"
    assert fake.actions[-1][0] == noop_action(ACTION_SPACE)
    assert fake.actions[-1][1] == action

    env.step({"player_1": action})
    assert env.agents == []
    with pytest.raises(RuntimeError):
        env.step({})


def test_hidden_reset_truncates_every_agent():
    fake = FakeListEnv(episode_lens=(10, 10))
    fake.hidden_reset_at = 2
    env = ComradParallelEnv(fake)
    env.reset()
    env.step({agent: (0, 0) for agent in env.agents})
    _, _, term, trunc, _ = env.step({agent: (0, 0) for agent in env.agents})
    assert trunc == {"player_0": True, "player_1": True}
    assert not any(term.values())
    assert env.agents == []


def test_render_is_forwarded():
    env = ComradParallelEnv(FakeListEnv(render_mode="rgb_array"))
    env.reset()
    assert env.render_mode == "rgb_array"
    assert env.render().shape == (4, 8, 3)


def test_parallel_api_test_on_fake_env():
    parallel_api_test(ComradParallelEnv(FakeListEnv(episode_lens=(5, 8))), num_cycles=50)


def test_parallel_env_wraps_public_make(monkeypatch):
    seen = {}

    def fake_build_env(scenario, render_mode=None, **options):
        seen.update(scenario=scenario, **options)
        return FakeListEnv(episode_lens=(1, 1, 1))

    monkeypatch.setattr(api, "_build_env", fake_build_env)
    env = api.parallel_env("armory_siege", num_agents=3, shared_reward=0.0)
    assert isinstance(env, ComradParallelEnv)
    assert env.possible_agents == ["player_0", "player_1", "player_2"]
    assert seen["scenario"] == "armory_siege"
    assert seen["num_agents"] == 3
    assert seen["shared_reward"] == 0.0
