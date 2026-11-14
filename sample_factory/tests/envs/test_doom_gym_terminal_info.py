from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from vizdoom.vizdoom import ViZDoomErrorException

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


class _VariableGame:
    def __init__(self, running, error=None):
        self.running = running
        self.error = error
        self.reads = []

    def is_running(self):
        return self.running

    def get_game_variable(self, variable):
        self.reads.append(variable)
        if self.error is not None:
            raise self.error
        return 7.0


def _variables_env(game):
    env = object.__new__(VizdoomEnv)
    env.game = game
    env.variable_indices = {"HEALTH": 0}
    return env


def test_game_variables_dict_propagates_engine_errors():
    env = _variables_env(_VariableGame(running=True, error=ViZDoomErrorException("engine failure")))

    with pytest.raises(ViZDoomErrorException):
        VizdoomEnv._game_variables_dict(env, SimpleNamespace(game_variables=[50.0]))


def test_game_variables_dict_reads_engine_variables_only_while_running():
    running = _variables_env(_VariableGame(running=True))
    variables = VizdoomEnv._game_variables_dict(running, SimpleNamespace(game_variables=[50.0]))
    assert variables["HEALTH"] == 50.0
    assert variables["POSITION_X"] == 7.0

    stopped_game = _VariableGame(running=False)
    variables = VizdoomEnv._game_variables_dict(_variables_env(stopped_game), SimpleNamespace(game_variables=[50.0]))
    assert variables == {"HEALTH": 50.0}
    assert stopped_game.reads == []
