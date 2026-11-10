import gymnasium as gym
import numpy as np
import pytest

from comrad.wrappers.shared_vision import SharedVisionWrapper


class FakeJointEnv(gym.Env):
    def __init__(self, shape=(3, 4, 5), dict_obs=True, done_at=None):
        self.num_agents = 2
        self.is_multiagent = True
        img = gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        self.observation_space = gym.spaces.Dict({"obs": img, "action_mask": gym.spaces.Box(0, 1, (4,))}) if dict_obs else img
        self.action_space = gym.spaces.Discrete(2)
        self.shape = shape
        self.dict_obs = dict_obs
        self.done_at = done_at
        self.t = 0
        self.last_raw = None

    def _obs(self):
        raw = [np.full(self.shape, 10 + i + self.t, dtype=np.uint8) for i in range(2)]
        self.last_raw = raw
        if self.dict_obs:
            return [{"obs": r, "action_mask": np.ones(4, np.float32) * i} for i, r in enumerate(raw)]
        return raw

    def reset(self, **kwargs):
        self.t = 0
        return self._obs(), [{}, {}]

    def step(self, actions):
        self.t += 1
        done = self.done_at is not None and self.t >= self.done_at
        return self._obs(), [0.0, 0.0], [done, done], [False, False], [{}, {}]


@pytest.mark.parametrize("dict_obs", [True, False])
def test_chw_obs_are_stacked_own_then_partner(dict_obs):
    inner = FakeJointEnv(dict_obs=dict_obs)
    env = SharedVisionWrapper(inner)
    space = env.observation_space["obs"] if dict_obs else env.observation_space
    assert space.shape == (6, 4, 5) and space.dtype == np.uint8
    for call in (lambda: env.reset()[0], lambda: env.step([0, 0])[0]):
        obs = call()
        raw = inner.last_raw
        for i in range(2):
            img = obs[i]["obs"] if dict_obs else obs[i]
            assert img.shape == (6, 4, 5)
            assert np.array_equal(img[:3], raw[i])
            assert np.array_equal(img[3:], raw[1 - i])
            assert space.contains(img)
        if dict_obs:
            assert obs[1]["action_mask"][0] == 1.0  # other keys preserved per agent
            assert inner.last_raw[0].shape == (3, 4, 5)  # inner obs not mutated


def test_hwc_obs_stack_on_last_axis():
    env = SharedVisionWrapper(FakeJointEnv(shape=(72, 128, 3), dict_obs=False))
    assert env.observation_space.shape == (72, 128, 6)
    obs, _ = env.reset()
    assert obs[0].shape == (72, 128, 6)


def test_none_obs_passthrough_and_agent_count_check():
    env = SharedVisionWrapper(FakeJointEnv())
    assert env._stack([None, None]) == [None, None]
    assert env._stack(None) is None
    bad = FakeJointEnv()
    bad.num_agents = 3
    with pytest.raises(ValueError):
        SharedVisionWrapper(bad)


def test_stack_measurements_and_relative_pose():
    class FakeMeasJointEnv(gym.Env):
        def __init__(self):
            self.num_agents = 2
            self.is_multiagent = True
            img = gym.spaces.Box(0, 255, shape=(3, 72, 128), dtype=np.uint8)
            meas = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
            self.observation_space = gym.spaces.Dict({"obs": img, "measurements": meas})
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, **kwargs):
            obs = [
                {"obs": np.full((3, 72, 128), 11, dtype=np.uint8), "measurements": np.ones(7, dtype=np.float32) * 0.2},
                {"obs": np.full((3, 72, 128), 22, dtype=np.uint8), "measurements": np.ones(7, dtype=np.float32) * 0.8},
            ]
            infos = [
                {"POSITION_X": 0.0, "POSITION_Y": 0.0, "ANGLE": 0.0},
                {"POSITION_X": 1024.0, "POSITION_Y": 0.0, "ANGLE": 90.0},
            ]
            return obs, infos

        def step(self, actions):
            obs, infos = self.reset()
            return obs, [0.0, 0.0], [False, False], [False, False], infos

    env = SharedVisionWrapper(FakeMeasJointEnv(), stack_measurements=True, include_relative_pose=True, map_scale=2048.0)
    assert env.observation_space["obs"].shape == (6, 72, 128)
    assert env.observation_space["measurements"].shape == (2 * 7 + 5,)
    obs, _ = env.reset()
    assert np.array_equal(obs[0]["obs"][3:6], obs[1]["obs"][0:3])
    assert np.allclose(obs[0]["measurements"][:7], 0.2)
    assert np.allclose(obs[0]["measurements"][7:14], 0.8)
    # dx = 1024/2048 = 0.5, dy = 0, dist = 0.5, cos(90) = 0, sin(90) = 1
    assert np.allclose(obs[0]["measurements"][14:], [0.5, 0.0, 0.5, 0.0, 1.0], atol=1e-5)


def test_relative_pose_uses_reset_info_on_auto_reset_step():
    class FakeAutoResetEnv(gym.Env):
        def __init__(self):
            self.num_agents = 2
            self.is_multiagent = True
            self.observation_space = gym.spaces.Box(0, 255, shape=(3, 8, 8), dtype=np.uint8)
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, **kwargs):
            return [np.zeros((3, 8, 8), np.uint8)] * 2, [{}, {}]

        def step(self, actions):
            terminal = [{"POSITION_X": 0.0, "POSITION_Y": 0.0, "ANGLE": 0.0},
                        {"POSITION_X": 2048.0, "POSITION_Y": 0.0, "ANGLE": 0.0}]
            fresh = [{"POSITION_X": 0.0, "POSITION_Y": 0.0, "ANGLE": 0.0},
                     {"POSITION_X": 0.0, "POSITION_Y": 1024.0, "ANGLE": 90.0}]
            infos = [dict(terminal[i], reset_info=fresh[i]) for i in range(2)]
            return [np.zeros((3, 8, 8), np.uint8)] * 2, [0.0, 0.0], [True, True], [False, False], infos

    env = SharedVisionWrapper(FakeAutoResetEnv(), include_relative_pose=True, map_scale=2048.0)
    obs, _, _, _, infos = env.step([0, 0])
    # the returned frame belongs to the new episode, so the pose must come from reset_info, not the terminal infos
    assert np.allclose(obs[0]["measurements"], [0.0, 0.5, 0.5, 0.0, 1.0], atol=1e-5)
    assert infos[1]["POSITION_X"] == 2048.0


def test_1d_state_stacking():
    class Fake1DStateJointEnv(gym.Env):
        def __init__(self):
            self.num_agents = 2
            self.is_multiagent = True
            state_box = gym.spaces.Box(-5.0, 5.0, shape=(18,), dtype=np.float32)
            self.observation_space = gym.spaces.Dict({"obs": state_box})
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, **kwargs):
            return [
                {"obs": np.ones(18, dtype=np.float32) * 1.0},
                {"obs": np.ones(18, dtype=np.float32) * 2.0},
            ], [{}, {}]

        def step(self, actions):
            obs, infos = self.reset()
            return obs, [0.0, 0.0], [False, False], [False, False], infos

    env = SharedVisionWrapper(Fake1DStateJointEnv(), stack_measurements=True, include_relative_pose=True)
    assert env.observation_space["obs"].shape == (36,)
    assert "measurements" not in env.observation_space.spaces
    obs, _ = env.reset()
    assert obs[0]["obs"].shape == (36,)
    assert np.allclose(obs[0]["obs"][:18], 1.0)
    assert np.allclose(obs[0]["obs"][18:], 2.0)
