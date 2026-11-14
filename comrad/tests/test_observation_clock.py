import gymnasium as gym
import numpy as np
import pytest

from comrad.envs.doom_params import default_doom_cfg
from comrad.utils import doom_utils
from comrad.utils.doom_utils import (
    CLOCK_EGO,
    CLOCK_JOINT,
    DoomSpec,
    doom_env_by_name,
    episode_horizon_steps,
    observation_clock_horizon,
)
from comrad.wrappers.state_observation import BlindObservationWrapper, StateObservationWrapper


class FakeEngine(gym.Env):
    """Stands in for VizdoomEnvMultiplayer with skip_frames=1: one tic per call, no observation on skipped frames,
    and the engine timeout VizdoomEnvMultiplayer sets from `timelimit` minutes."""

    action_space = gym.spaces.Discrete(2)

    def __init__(self, *args, timelimit=0.0, player_id=0, **kwargs):
        self.timeout_tics = int(timelimit * 60 * 35)
        self.player_id = player_id
        self.update_state = True
        self.screen_w, self.screen_h = 160, 120
        self.tics = 0
        self.calc_observation_space()

    def calc_observation_space(self):
        self.observation_space = gym.spaces.Box(0, 255, (self.screen_h, self.screen_w, 3), dtype=np.uint8)

    def reset(self, **kwargs):
        self.tics = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self.tics += 1
        if not self.update_state:
            return None, None, None, None, None
        terminated = 0 < self.timeout_tics <= self.tics
        return np.zeros(self.observation_space.shape, dtype=np.uint8), 0.0, terminated, False, {}


def _player_env(monkeypatch, cfg, spec):
    from comrad.envs.multiagent import doom_multiagent

    monkeypatch.setattr(doom_multiagent, "VizdoomEnvMultiplayer", FakeEngine)
    return doom_utils.make_doom_env_impl(spec, cfg=cfg, player_id=0, num_agents=2, max_num_players=2, skip_frames=1)


def _joint_step(env, frameskip):
    """What MultiAgentEnv.step does to every player: frameskip - 1 frames without state, then one with state."""
    for _ in range(frameskip - 1):
        env.unwrapped.update_state = False
        assert env.step(0)[0] is None
    env.unwrapped.update_state = True
    return env.step(0)


def _cfg(env_name="coop_puzzle", **overrides):
    cfg = default_doom_cfg(env=env_name)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _run_state_clock(monkeypatch, cfg, spec):
    env = _player_env(monkeypatch, cfg, spec)
    obs, _ = env.reset()
    clocks = [float(obs["obs"][7])]
    while True:
        obs, _, terminated, truncated, _ = _joint_step(env, cfg.env_frameskip)
        clocks.append(float(obs["obs"][7]))
        if terminated or truncated:
            return clocks, terminated


@pytest.mark.parametrize(
    "frameskip, timelimit, engine_ends",
    [(1, None, False), (4, None, False), (1, 1.0, False), (4, 1.0, True), (4, 0.2, True)],
)
def test_state_clock_reaches_one_exactly_when_the_episode_ends(monkeypatch, frameskip, timelimit, engine_ends):
    spec = doom_env_by_name("coop_puzzle")
    cfg = _cfg(obs_modality="state", env_frameskip=frameskip, timelimit=timelimit)
    clocks, terminated = _run_state_clock(monkeypatch, cfg, spec)

    steps = len(clocks) - 1
    assert steps == episode_horizon_steps(spec, cfg)
    assert bool(terminated) is engine_ends, "which limit ends the episode (engine timeout vs TimeLimitWrapper)"
    assert clocks[-1] == pytest.approx(1.0)
    assert clocks[-2] < 1.0
    assert clocks[steps // 2] == pytest.approx((steps // 2) / steps)


@pytest.mark.parametrize("frameskip, timelimit", [(1, None), (4, None), (4, 1.0)])
def test_blind_clock_reaches_one_exactly_when_the_episode_ends(monkeypatch, frameskip, timelimit):
    spec = doom_env_by_name("coop_puzzle")
    cfg = _cfg(obs_modality="blind", env_frameskip=frameskip, timelimit=timelimit)
    env = _player_env(monkeypatch, cfg, spec)
    env.reset()
    pixels = []
    while True:
        obs, _, terminated, truncated, _ = _joint_step(env, frameskip)
        pixels.append(int(np.asarray(obs).flat[0]))
        if terminated or truncated:
            break
    horizon = episode_horizon_steps(spec, cfg)
    assert len(pixels) == horizon
    assert pixels[-1] == 255
    assert pixels[horizon // 2 - 1] == int(round((horizon // 2) / horizon * 255.0))


def test_coop_puzzle_frameskip_4_horizon_is_the_step_limit_not_the_cfg_timeout():
    spec = doom_env_by_name("coop_puzzle")
    cfg = _cfg(env_frameskip=4)
    assert episode_horizon_steps(spec, cfg) == 2000
    assert observation_clock_horizon(spec, cfg, CLOCK_JOINT) == 2000
    assert observation_clock_horizon(spec, cfg, CLOCK_EGO) == 2000
    cfg.timelimit = 1.0  # 2100 tics: the engine ends the episode after 525 joint steps
    assert episode_horizon_steps(spec, cfg) == 525
    cfg.env_frameskip = 1
    assert episode_horizon_steps(spec, cfg) == 2000


def test_version_1_reproduces_the_archived_clocks(monkeypatch):
    spec = doom_env_by_name("coop_puzzle")
    cfg = _cfg(obs_modality="state", env_frameskip=4, observation_clock_version=1)
    # global/entity clocks used the scenario cfg episode_timeout (2000 tics) / frameskip
    assert observation_clock_horizon(spec, cfg, CLOCK_JOINT) == 500
    # state/blind clocks counted every skipped frame against the 2000 step limit
    assert observation_clock_horizon(spec, cfg, CLOCK_EGO) == 500

    env = _player_env(monkeypatch, cfg, spec)
    env.reset()
    for step in range(1, 501):
        obs, *_ = _joint_step(env, 4)
        assert float(obs["obs"][7]) == pytest.approx(min(1.0, np.float32(step * 4 / 2000)))

    rhythm = doom_env_by_name("rhythm_sync")
    rhythm_cfg = _cfg("rhythm_sync", env_frameskip=4, observation_clock_version=1)
    assert observation_clock_horizon(rhythm, rhythm_cfg, CLOCK_EGO) == 1750 / 4
    stealth = doom_env_by_name("stealth_labyrinth")
    stealth_cfg = _cfg("stealth_labyrinth", env_frameskip=4, observation_clock_version=1)
    assert observation_clock_horizon(stealth, stealth_cfg, CLOCK_JOINT) == 788  # ceil(3150 / 4)


def test_archived_configs_resolve_to_the_version_1_clock():
    assert doom_utils.resolve_archived_cfg({"env": "coop_puzzle"})["observation_clock_version"] == 1
    assert doom_utils.resolve_archived_cfg({"observation_clock_version": 2})["observation_clock_version"] == 2
    assert default_doom_cfg(env="coop_puzzle").observation_clock_version == 2


def test_episode_without_any_limit_has_no_horizon(tmp_path):
    cfg_file = tmp_path / "unlimited.cfg"
    cfg_file.write_text("doom_map = map01\n")
    spec = DoomSpec("unlimited", str(cfg_file), gym.spaces.Discrete(2), timelimit=0.0)
    with pytest.raises(ValueError, match="neither a step limit nor an engine timeout"):
        episode_horizon_steps(spec, _cfg(env_frameskip=4))


def test_singleplayer_horizon_reads_the_scenario_cfg_and_fails_loudly(tmp_path):
    cfg_file = tmp_path / "single.cfg"
    cfg_file.write_text("episode_timeout = 1000\n")
    spec = DoomSpec("single", str(cfg_file), gym.spaces.Discrete(2), default_timeout=4000)
    assert episode_horizon_steps(spec, _cfg(env_frameskip=4)) == 250
    missing = DoomSpec("missing", str(tmp_path / "missing.cfg"), gym.spaces.Discrete(2))
    with pytest.raises(OSError):
        episode_horizon_steps(missing, _cfg(env_frameskip=4))


@pytest.mark.parametrize("horizon", [None, 0, -5])
def test_clock_wrappers_reject_missing_horizons(horizon):
    base = FakeEngine(timelimit=1.0)
    with pytest.raises(ValueError, match="horizon"):
        BlindObservationWrapper(base, horizon=horizon)
    with pytest.raises(ValueError, match="horizon"):
        StateObservationWrapper(base, scenario_name="coop_puzzle", horizon=horizon)


def test_clock_version_must_be_set_explicitly():
    from sample_factory.utils.attr_dict import AttrDict

    assert doom_utils.get_observation_clock_version(AttrDict({"observation_clock_version": 1})) == 1
    with pytest.raises(ValueError, match="observation_clock_version"):
        doom_utils.get_observation_clock_version(AttrDict({"env": "coop_puzzle"}))
