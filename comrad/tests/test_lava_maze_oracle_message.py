import gymnasium as gym
import numpy as np
import pytest

from comrad.envs.doom_params import default_doom_cfg
from comrad.utils import doom_utils
from comrad.wrappers.scenario_wrappers.lava_maze import LavaMazeOracleMessageWrapper
from comrad.wrappers.scripted_partner import zero_action
from comrad.wrappers.state_observation import EGO_DIM

SCENARIO = "lava_maze_oracle_msg"
NAVIGATOR, OBSERVER = 0, 1


class _FakeDoomPlayer(gym.Env):
    """Stands in for VizdoomEnvMultiplayer: deterministic frames and game variables, no ViZDoom process."""

    def __init__(self, action_space, env_spec_file, player_id=None, **kwargs):
        self.action_space = action_space
        self.player_id = player_id
        self.screen_w, self.screen_h = 160, 120
        self.calc_observation_space()
        self.t = 0

    def calc_observation_space(self):
        self.observation_space = gym.spaces.Box(0, 255, (self.screen_h, self.screen_w, 3), dtype=np.uint8)

    def get_info(self):
        return {"HEALTH": 100.0, "USER16": 0, "USER17": 0, "POSITION_X": 64.0 * self.player_id, "POSITION_Y": 0.0}

    def _frame(self):
        return np.full(self.observation_space.shape, 10 * self.player_id + self.t, dtype=np.uint8)

    def reset(self, **kwargs):
        self.t = 0
        return self._frame(), self.get_info()

    def step(self, action):
        self.t += 1
        return self._frame(), 0.0, False, False, self.get_info()


class _FakeMultiAgentEnv(gym.Env):
    """Stands in for MultiAgentEnv: builds the real per-player wrapper stacks, steps them in-process."""

    def __init__(self, num_agents, make_env_func, **kwargs):
        self.num_agents = num_agents
        self.is_multiagent = True
        self.envs = [make_env_func(player_id=i) for i in range(num_agents)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, **kwargs):
        obs, infos = zip(*(env.reset(**kwargs) for env in self.envs))
        return list(obs), list(infos)

    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        return tuple(list(column) for column in zip(*results))


def _make_env(monkeypatch, obs_modality, **cfg_overrides):
    import comrad.envs.multiagent.doom_multiagent as player_module
    import comrad.envs.multiagent.doom_multiagent_wrapper as mae_module

    monkeypatch.setattr(player_module, "VizdoomEnvMultiplayer", _FakeDoomPlayer)
    monkeypatch.setattr(mae_module, "MultiAgentEnv", _FakeMultiAgentEnv)
    cfg = default_doom_cfg(algo="MAPPO", env=SCENARIO)
    cfg.obs_modality = obs_modality
    cfg.num_agents = -1
    cfg.num_bots = 0
    cfg.num_humans = 0
    cfg.wandb_record_every = 0
    for key, value in cfg_overrides.items():
        setattr(cfg, key, value)
    return doom_utils.make_doom_multiplayer_env(doom_utils.doom_env_by_name(SCENARIO), cfg=cfg, env_config=None)


def _observer_says(env, message):
    actions = [zero_action(env.action_space) for _ in range(env.num_agents)]
    observer_action = list(actions[OBSERVER])
    observer_action[3] = message
    actions[OBSERVER] = tuple(observer_action)
    return actions


def _navigator_obs_after(env, message):
    env.reset()
    obs, *_ = env.step(_observer_says(env, message))
    return obs


@pytest.mark.parametrize(
    "obs_modality, key, message_offset",
    [("pixel", "measurements", 0), ("state", "obs", EGO_DIM)],
)
def test_different_messages_reach_the_navigator(monkeypatch, obs_modality, key, message_offset):
    env = _make_env(monkeypatch, obs_modality)
    obs_1 = _navigator_obs_after(env, 1)
    obs_4 = _navigator_obs_after(env, 4)

    for obs in (obs_1, obs_4):
        for agent_obs in obs:
            assert agent_obs[key].shape == env.observation_space[key].shape
    assert not np.array_equal(obs_1[NAVIGATOR][key], obs_4[NAVIGATOR][key])
    np.testing.assert_array_equal(obs_1[OBSERVER][key], obs_4[OBSERVER][key])

    for message, obs in ((1, obs_1), (4, obs_4)):
        one_hot = obs[NAVIGATOR][key][message_offset + 1 : message_offset + 7]
        expected = np.zeros(6, dtype=np.float32)
        expected[message] = 1.0
        expected[5] = 1.0
        np.testing.assert_array_equal(one_hot, expected)


def test_pixel_and_state_navigator_messages_agree(monkeypatch):
    pixel = _navigator_obs_after(_make_env(monkeypatch, "pixel"), 3)[NAVIGATOR]["measurements"]
    state = _navigator_obs_after(_make_env(monkeypatch, "state"), 3)[NAVIGATOR]["obs"]
    np.testing.assert_array_equal(state[EGO_DIM + 1 : EGO_DIM + 7], pixel[1:7])


def test_oracle_wrapper_without_message_channel_raises():
    class _NoChannelEnv(gym.Env):
        num_agents = 2
        observation_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 255, (3, 72, 128), dtype=np.uint8)})
        action_space = doom_utils.doom_env_by_name(SCENARIO).action_space

    with pytest.raises(ValueError, match="no message channel"):
        LavaMazeOracleMessageWrapper(_NoChannelEnv())


@pytest.mark.parametrize(
    "obs_modality, overrides",
    [("pixel", {"use_additional_input": False}), ("state", {"use_additional_input": False})],
)
def test_pipeline_without_measurements_raises(monkeypatch, obs_modality, overrides):
    with pytest.raises(ValueError, match="no message channel"):
        _make_env(monkeypatch, obs_modality, **overrides)
