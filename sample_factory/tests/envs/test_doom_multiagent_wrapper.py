from __future__ import annotations

import gymnasium as gym

import comrad.envs.multiagent.doom_multiagent_wrapper as doom_multiagent_wrapper


class _TinyEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, **kwargs):
        return 0, {}

    def step(self, action):
        return 0, 0.0, False, False, {}

    def close(self):
        return None


def test_multiagent_reset_retries_after_group_crash(monkeypatch):
    monkeypatch.setattr(doom_multiagent_wrapper, "get_default_reward_shaping", lambda env: {})

    env = doom_multiagent_wrapper.MultiAgentEnv(
        num_agents=2,
        make_env_func=lambda player_id: _TinyEnv(),
        env_config=None,
        skip_frames=1,
        render_mode=None,
    )

    calls = {"await": 0, "close": 0}
    env.initialized = True

    monkeypatch.setattr(env, "_ensure_initialized", lambda: None)
    monkeypatch.setattr(doom_multiagent_wrapper, "sleep", lambda _: None)

    def fake_close():
        calls["close"] += 1

    def fake_await_tasks(data, task_type, timeout=None):
        calls["await"] += 1
        if calls["await"] == 1:
            raise doom_multiagent_wrapper._GameGroupCrashError("reset crash")
        return [0, 0], [{}, {}]

    monkeypatch.setattr(env, "close", fake_close)
    monkeypatch.setattr(env, "await_tasks", fake_await_tasks)

    obs, info = env.reset()

    assert calls["close"] == 1
    assert env.initialized is False
    assert obs == [0, 0]
    assert info == [{}, {}]
