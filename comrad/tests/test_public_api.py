import subprocess
import sys

import gymnasium as gym
import pytest

import comrad
from comrad import api
from comrad.envs.doom_params import default_doom_cfg
from comrad.utils.doom_utils import DOOM_ENVS, SCENARIO_METADATA, doom_env_by_name, get_alpha, get_extra_wrappers


def test_list_scenarios_matches_doom_envs():
    scenarios = comrad.list_scenarios()
    assert [s.name for s in scenarios] == [spec.name for spec in DOOM_ENVS]
    for s in scenarios:
        assert s.num_agents >= 2
        assert s.description and s.score


def test_every_scenario_has_metadata():
    assert set(SCENARIO_METADATA) == {spec.name for spec in DOOM_ENVS}


def test_gymnasium_ids_registered():
    for spec in DOOM_ENVS:
        env_spec = gym.spec(f"comrad/{spec.name}-v0")
        assert env_spec.entry_point == "comrad.api:make"
        assert env_spec.kwargs == {"scenario": spec.name}


@pytest.mark.parametrize("scenario", ["stag_hunt_arena", "lavapit", "armory_siege"])
def test_default_cfg_matches_training_defaults(scenario):
    spec, cfg = api._build_cfg(scenario)
    train_cfg = default_doom_cfg(env=scenario)

    assert spec is doom_env_by_name(scenario)
    assert cfg.env_frameskip == train_cfg.env_frameskip == 4
    assert (cfg.res_w, cfg.res_h) == (train_cfg.res_w, train_cfg.res_h) == (128, 72)
    assert cfg.wide_aspect_ratio == train_cfg.wide_aspect_ratio
    assert cfg.num_agents == spec.num_agents
    assert get_alpha(cfg, spec) == spec.shared_reward_alpha
    assert get_extra_wrappers(cfg, spec) == get_extra_wrappers(train_cfg, spec)
    assert cfg.algo.upper() not in {"HAPPO", "IDQN", "VDN", "QMIX", "QPLEX"}
    assert not cfg.with_wandb


def test_options_are_mapped_onto_cfg():
    spec, cfg = api._build_cfg(
        "armory_siege",
        num_agents=3,
        frameskip=2,
        resolution=(64, 36),
        pixel_format="CHW",
        shared_reward=0.0,
        additional_input=True,
    )
    assert cfg.num_agents == 3
    assert cfg.env_frameskip == 2
    assert (cfg.res_w, cfg.res_h) == (64, 36)
    assert cfg.pixel_format == "CHW"
    assert get_alpha(cfg, spec) == 0.0
    assert cfg.use_additional_input is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_agents": 1},
        {"frameskip": 0},
        {"pixel_format": "NCHW"},
        {"shared_reward": 1.5},
        {"resolution": (0, 72)},
    ],
)
def test_invalid_options_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        api._build_cfg("stag_hunt_arena", **kwargs)


def test_unknown_scenario_raises_value_error():
    with pytest.raises(ValueError, match="Unknown COMRAD scenario"):
        comrad.make("not_a_scenario")


def test_invalid_render_mode_raises_value_error():
    with pytest.raises(ValueError, match="render_mode"):
        comrad.make("stag_hunt_arena", render_mode="video")


def test_make_builds_explicit_reset_env_without_env_config(monkeypatch):
    calls = {}

    class StubEnv(gym.Env):
        enable_rendering = False

    stub = StubEnv()

    def fake_factory(spec, cfg=None, env_config="unset", render_mode=None, auto_reset=True):
        calls.update(spec=spec, cfg=cfg, env_config=env_config, render_mode=render_mode, auto_reset=auto_reset)
        return stub

    monkeypatch.setattr(api, "make_doom_multiplayer_env", fake_factory)
    assert comrad.make("lavapit", render_mode="rgb_array") is stub
    assert stub.enable_rendering is True
    assert calls["spec"].name == "lavapit"
    assert calls["env_config"] is None
    assert calls["auto_reset"] is False
    assert calls["render_mode"] == "rgb_array"


def test_gym_make_routes_to_comrad_make(monkeypatch):
    seen = {}

    class Dummy(gym.Env):
        observation_space = gym.spaces.Discrete(1)
        action_space = gym.spaces.Discrete(1)

    def fake_build_env(scenario, render_mode=None, **options):
        seen.update(scenario=scenario, render_mode=render_mode, **options)
        return Dummy()

    monkeypatch.setattr(api, "_build_env", fake_build_env)
    env = gym.make("comrad/stag_hunt_arena-v0", num_agents=3)
    assert isinstance(env.unwrapped, Dummy)
    assert seen["scenario"] == "stag_hunt_arena"
    assert seen["num_agents"] == 3


def test_import_has_no_side_effects():
    code = (
        "import threading, multiprocessing, comrad, torch, gymnasium as gym\n"
        "assert not torch.cuda.is_initialized()\n"
        "assert threading.active_count() == 1, threading.enumerate()\n"
        "assert not multiprocessing.active_children()\n"
        "assert 'comrad/stag_hunt_arena-v0' in gym.registry\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
