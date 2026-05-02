from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from comrad.envs.doom_gym import VizdoomEnv


class _DummyGame:
    def get_episode_time(self):
        return 176

    def get_episode_timeout(self):
        return 5250

    def is_player_dead(self):
        return True


def test_process_game_step_done_overlays_terminal_status():
    env = object.__new__(VizdoomEnv)
    env.game = _DummyGame()
    env.black_screen = None
    env.observation_space = SimpleNamespace(shape=(3, 4, 4))
    env._prev_info = {"HEALTH": 2.0, "USER1": 79.0}
    env._last_episode_info = None

    obs, done, info = VizdoomEnv._process_game_step(env, state=None, done=True, info={})

    assert done is True
    assert obs.shape == (3, 4, 4)
    assert np.all(obs == 0)
    assert info["HEALTH"] == 2.0
    assert info["USER1"] == 79.0
    assert info["episode_time_tics"] == 176
    assert info["episode_timeout_tics"] == 5250
    assert info["player_dead"] is True


def test_process_game_step_done_without_prev_info_does_not_crash():
    env = object.__new__(VizdoomEnv)
    env.game = _DummyGame()
    env.black_screen = None
    env.observation_space = SimpleNamespace(shape=(3, 4, 4))
    env._prev_info = None
    env._last_episode_info = None

    obs, done, info = VizdoomEnv._process_game_step(env, state=None, done=True, info={})

    assert done is True
    assert obs.shape == (3, 4, 4)
    assert np.all(obs == 0)
    assert info["episode_time_tics"] == 176
    assert info["episode_timeout_tics"] == 5250
    assert info["player_dead"] is True
