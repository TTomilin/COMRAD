from contextlib import contextmanager

import gymnasium as gym
import numpy as np
import pytest

from comrad.envs.multiagent import doom_multiagent_wrapper as wrapper_module
from comrad.envs.multiagent.doom_multiagent_wrapper import MultiAgentEnv

EPISODE_LEN = 3
# {"step": k, "player": p} makes player p raise once on its k-th step of an episode
CRASH_ONCE = {}


class FakeAgentEnv(gym.Env):
    """Stands in for one VizdoomEnvMultiplayer player: the obs encodes the step counter, the episode lasts EPISODE_LEN."""

    observation_space = gym.spaces.Box(0, 255, (4, 4, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(3)

    def __init__(self, player_id):
        self.player_id = player_id
        self.worker_index = 0
        self.vector_index = 0
        self.t = 0

    def seed(self, seed=None):
        pass

    def _obs(self):
        return np.full((4, 4, 3), self.t, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        self.t = 0
        return self._obs(), {"player": self.player_id}

    def step(self, action):
        if CRASH_ONCE.get("step") == self.t + 1 and CRASH_ONCE.get("player") == self.player_id:
            CRASH_ONCE.clear()
            raise RuntimeError("simulated ViZDoom segfault")
        self.t += 1
        return self._obs(), 1.0, self.t >= EPISODE_LEN, False, {"player": self.player_id}


@pytest.fixture(autouse=True)
def no_ports(monkeypatch, tmp_path):
    @contextmanager
    def fake_reserve(port, increment=1000):
        yield port

    monkeypatch.setattr(wrapper_module, "reserve_available_port", fake_reserve)
    monkeypatch.setattr(wrapper_module, "doom_lock_file", lambda max_parallel: str(tmp_path / "doom.lock"))
    monkeypatch.setattr(wrapper_module, "sleep", lambda seconds: None)
    CRASH_ONCE.clear()
    yield
    CRASH_ONCE.clear()


def _make(auto_reset=True, env_config=None, num_agents=2):
    return MultiAgentEnv(
        num_agents=num_agents,
        make_env_func=lambda player_id: FakeAgentEnv(player_id),
        env_config=env_config,
        skip_frames=1,
        render_mode=None,
        auto_reset=auto_reset,
    )


def _run_episode(env):
    out = None
    for _ in range(EPISODE_LEN):
        out = env.step([0, 0])
    return out


def test_env_config_none_initializes_without_attribute_error():
    # Regression: _ensure_initialized() used to log env_config.worker_index and crash when env_config is None.
    env = _make(env_config=None)
    try:
        obs, infos = env.reset()
        assert len(obs) == 2
        assert [info["player"] for info in infos] == [0, 1]
    finally:
        env.close()


def test_auto_reset_default_keeps_sample_factory_behaviour():
    env = _make()
    try:
        assert env.auto_reset is True
        env.reset()
        obs, rewards, terminated, truncated, infos = _run_episode(env)
        assert all(terminated)
        assert all(int(o[0, 0, 0]) == 0 for o in obs), "auto-reset returns the first obs of the next episode"
        assert all("reset_info" in info for info in infos)
    finally:
        env.close()


def test_explicit_reset_returns_terminal_obs_and_waits_for_reset():
    env = _make(auto_reset=False)
    try:
        env.reset()
        obs, rewards, terminated, truncated, infos = _run_episode(env)
        assert all(terminated)
        assert rewards == [1.0, 1.0]
        assert all(int(o[0, 0, 0]) == EPISODE_LEN for o in obs), "terminal obs must not be replaced"
        assert not any("reset_info" in info for info in infos)

        obs, infos = env.reset()
        assert all(int(o[0, 0, 0]) == 0 for o in obs)
        obs, *_ = env.step([0, 0])
        assert all(int(o[0, 0, 0]) == 1 for o in obs)
    finally:
        env.close()


def test_explicit_reset_truncates_episode_on_crash_recovery_reset():
    # Regression: crash recovery resets the game and re-runs step() on a fresh episode. Without auto-reset the caller
    # must see that boundary as a truncation, otherwise returns get stitched across two episodes.
    env = _make(auto_reset=False)
    try:
        env.reset()
        env.step([0, 0])
        CRASH_ONCE.update(step=2, player=1)
        obs, rewards, terminated, truncated, infos = env.step([0, 0])
        assert not CRASH_ONCE, "the simulated crash must have fired"
        assert all(int(o[0, 0, 0]) == 1 for o in obs), "step was re-run on the fresh episode"
        assert truncated == [True, True]
        assert terminated == [False, False]
        assert all(info["_hidden_reset_count"] == 1 for info in infos)

        obs, infos = env.reset()
        _, _, terminated, truncated, infos = env.step([0, 0])
        assert truncated == [False, False]
        assert not any("_hidden_reset_count" in info for info in infos)
    finally:
        env.close()


def test_auto_reset_crash_recovery_keeps_done_flags():
    # Sample Factory path: the hidden reset is only surfaced via info, done flags stay untouched.
    env = _make(auto_reset=True)
    try:
        env.reset()
        env.step([0, 0])
        CRASH_ONCE.update(step=2, player=0)
        _, _, terminated, truncated, infos = env.step([0, 0])
        assert not CRASH_ONCE
        assert truncated == [False, False]
        assert terminated == [False, False]
        assert all(info["_hidden_reset_count"] == 1 for info in infos)
    finally:
        env.close()
