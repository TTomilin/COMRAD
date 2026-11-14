import gymnasium as gym
import numpy as np
import pytest

from comrad.utils.doom_utils import DOOM_ENVS, cue_observation_layout, doom_env_by_name
from comrad.wrappers.scenario_wrappers.rhythm_sync_dense import RhythmSyncAdditionalInput
from comrad.wrappers.visibility_diagnostics import (
    CUE_LAYOUT_NO_CUE_CHANNEL,
    CUE_LAYOUT_SCENARIO_MEASUREMENTS,
    VisibilityDiagnosticsWrapper,
    cue_measurement_slots,
)
from sample_factory.utils.attr_dict import AttrDict


IMG_SHAPE = (3, 72, 128)


class _FakeSingleEnv(gym.Env):
    def __init__(self):
        self.player_id = 0
        self.observation_space = gym.spaces.Box(0, 255, shape=IMG_SHAPE, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)


class _FakeJointEnv(gym.Env):
    """Two agents; agent 0 sees the partner sprite (labels value 7) at rows 20:40, cols 50:70 of its frame."""

    def __init__(self, measurement_dim=None):
        self.num_agents = 2
        self.is_multiagent = True
        self.measurement_dim = measurement_dim
        spaces = {"obs": gym.spaces.Box(0, 255, shape=IMG_SHAPE, dtype=np.uint8)}
        if measurement_dim is not None:
            spaces["measurements"] = gym.spaces.Box(-10.0, 10.0, shape=(measurement_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = gym.spaces.Discrete(2)

    def _data(self):
        obs = []
        for agent in range(2):
            img = np.full(IMG_SHAPE, 40, dtype=np.uint8)
            if agent == 0:
                img[:, 20:40, 50:70] = 240
            o = {"obs": img}
            if self.measurement_dim is not None:
                o["measurements"] = np.arange(1, self.measurement_dim + 1, dtype=np.float32) + 100 * agent
            obs.append(o)
        labels_buf = np.zeros((120, 160), dtype=np.uint8)
        labels_buf[30:70, 60:90] = 7
        sprite = {"value": 7, "object_name": "DoomPlayer", "width": 30, "height": 40,
                  "object_position_x": 200.0, "object_position_y": 0.0, "object_position_z": 0.0}
        infos = [
            {"POSITION_X": 0.0, "POSITION_Y": 0.0, "ANGLE": 0.0, "_labels": [sprite], "_labels_buffer": labels_buf},
            {"POSITION_X": 200.0, "POSITION_Y": 0.0, "ANGLE": 0.0, "_labels": [],
             "_labels_buffer": np.zeros((120, 160), dtype=np.uint8)},
        ]
        return obs, infos

    def reset(self, **kwargs):
        return self._data()

    def step(self, actions):
        obs, infos = self._data()
        return obs, [0.0, 0.0], [False, False], [False, False], infos


def _cfg(**overrides):
    cfg = dict(obs_modality="pixel", full_observability=False, full_obs_mode=None, use_additional_input=None)
    cfg.update(overrides)
    return AttrDict(cfg)


def _step(env):
    env.reset()
    obs, _, _, _, _ = env.step([0, 0])
    return obs


def _expected_cue_slots(scenario, dim):
    return {
        ("rhythm_sync_dense", 7): ({3, 4}, {3, 4}),
        ("rhythm_sync_dense", 9): ({3, 4, 5, 6}, {3, 4, 5, 6}),
        ("ammo_carrier", 10): (set(), {3, 4, 5}),
        ("lava_maze", 7): (set(range(7)), set()),
        ("lava_maze_simple", 3): (set(range(3)), set()),
        ("armory_siege", 9): (set(), set()),
        ("foraging_commons", 8): (set(), set()),
        ("lavapit", 1): (set(), set()),
    }[(scenario, dim)]


@pytest.mark.parametrize(
    "scenario,dim",
    [("rhythm_sync_dense", 7), ("rhythm_sync_dense", 9), ("ammo_carrier", 10), ("lava_maze", 7),
     ("lava_maze_simple", 3), ("armory_siege", 9), ("foraging_commons", 8), ("lavapit", 1)],
)
@pytest.mark.parametrize("mode", ["cues", "both"])
def test_scenario_measurements_mask_cue_slots_and_preserve_the_rest(scenario, dim, mode):
    env = VisibilityDiagnosticsWrapper(
        _FakeJointEnv(measurement_dim=dim), scenario_name=scenario, partner_occlusion_mode=mode,
        cue_layout=CUE_LAYOUT_SCENARIO_MEASUREMENTS,
    )
    obs = _step(env)
    for agent, cue_slots in enumerate(_expected_cue_slots(scenario, dim)):
        original = np.arange(1, dim + 1, dtype=np.float32) + 100 * agent
        m = obs[agent]["measurements"]
        for slot in range(dim):
            assert m[slot] == (0.0 if slot in cue_slots else original[slot]), (agent, slot)


def test_every_scenario_measurement_vector_has_a_cue_schema():
    for spec in DOOM_ENVS:
        if spec.additional_input_wrapper is None:
            continue
        w_cls, w_kwargs = spec.additional_input_wrapper
        dim = int(w_cls(_FakeSingleEnv(), **w_kwargs).observation_space["measurements"].shape[0])
        cue_measurement_slots(spec.name, CUE_LAYOUT_SCENARIO_MEASUREMENTS, dim)
    full_dim = RhythmSyncAdditionalInput(_FakeSingleEnv(), feature_set="full").observation_space["measurements"].shape[0]
    cue_measurement_slots("rhythm_sync", CUE_LAYOUT_SCENARIO_MEASUREMENTS, int(full_dim))


@pytest.mark.parametrize(
    "scenario,cfg,expected",
    [
        ("rhythm_sync_dense", _cfg(use_additional_input=True), CUE_LAYOUT_SCENARIO_MEASUREMENTS),
        ("rhythm_sync_dense", _cfg(obs_modality="blind", use_additional_input=True), CUE_LAYOUT_SCENARIO_MEASUREMENTS),
        ("rhythm_sync_dense", _cfg(), CUE_LAYOUT_NO_CUE_CHANNEL),
        ("coop_puzzle", _cfg(), CUE_LAYOUT_NO_CUE_CHANNEL),
        ("coop_puzzle", _cfg(obs_modality="blind"), CUE_LAYOUT_NO_CUE_CHANNEL),
    ],
)
def test_per_agent_pixel_and_blind_observations_have_a_maskable_layout(scenario, cfg, expected):
    assert cue_observation_layout(cfg, doom_env_by_name(scenario)) == expected


UNSUPPORTED = [
    # global state: the partner cue (e.g. USER37) is encoded inside the state vector, indices 3-6 are heading/health/ammo
    ("rhythm_sync_dense", _cfg(full_observability=True, full_obs_mode="global_state", use_additional_input=False), 40),
    ("rhythm_sync_dense", _cfg(full_observability=True, full_obs_mode="global_state", use_additional_input=True), 47),
    # state modality: the measurements are folded into the 1D state vector
    ("rhythm_sync_dense", _cfg(obs_modality="state"), None),
    ("ammo_carrier", _cfg(obs_modality="state", state_encoding="entity"), None),
    # shared vision: the partner's frame and measurements are stacked into the observation
    ("rhythm_sync_dense", _cfg(full_observability=True, full_obs_mode="shared_vision", use_additional_input=True), 23),
    ("stealth_labyrinth_shared_vision", _cfg(), None),
]


@pytest.mark.parametrize("scenario,cfg,dim", UNSUPPORTED)
@pytest.mark.parametrize("mode", ["cues", "both"])
def test_cue_masking_raises_where_cues_are_encoded_elsewhere(scenario, cfg, dim, mode):
    layout = cue_observation_layout(cfg, doom_env_by_name(scenario))
    with pytest.raises(ValueError, match="Cue masking"):
        VisibilityDiagnosticsWrapper(
            _FakeJointEnv(measurement_dim=dim), scenario_name=scenario, partner_occlusion_mode=mode, cue_layout=layout
        )


@pytest.mark.parametrize("scenario,cfg,dim", UNSUPPORTED)
def test_sprite_masking_works_for_every_layout(scenario, cfg, dim):
    layout = cue_observation_layout(cfg, doom_env_by_name(scenario))
    env = VisibilityDiagnosticsWrapper(
        _FakeJointEnv(measurement_dim=dim), scenario_name=scenario, partner_occlusion_mode="sprite", cue_layout=layout
    )
    obs = _step(env)
    assert np.all(obs[0]["obs"] == 40)
    if dim is not None:
        np.testing.assert_array_equal(obs[0]["measurements"], np.arange(1, dim + 1, dtype=np.float32))


@pytest.mark.parametrize("scenario", ["coop_puzzle", "rhythm_sync_dense", "lava_maze"])
def test_both_without_a_cue_channel_masks_the_sprite_only(scenario):
    layout = cue_observation_layout(_cfg(), doom_env_by_name(scenario))
    assert layout == CUE_LAYOUT_NO_CUE_CHANNEL
    env = VisibilityDiagnosticsWrapper(
        _FakeJointEnv(measurement_dim=None), scenario_name=scenario, partner_occlusion_mode="both", cue_layout=layout
    )
    obs = _step(env)
    assert np.all(obs[0]["obs"] == 40)
    assert np.all(obs[1]["obs"] == 40)
    assert all("measurements" not in o for o in obs)


def test_no_cue_channel_layout_rejects_an_observation_with_measurements():
    with pytest.raises(ValueError, match="measurements"):
        VisibilityDiagnosticsWrapper(
            _FakeJointEnv(measurement_dim=7), scenario_name="rhythm_sync_dense", partner_occlusion_mode="cues",
            cue_layout=CUE_LAYOUT_NO_CUE_CHANNEL,
        )


def test_cue_masking_without_a_declared_layout_raises():
    with pytest.raises(ValueError, match="Cue masking"):
        VisibilityDiagnosticsWrapper(
            _FakeJointEnv(measurement_dim=7), scenario_name="rhythm_sync_dense", partner_occlusion_mode="cues"
        )


def test_switching_a_live_wrapper_to_cue_masking_resolves_the_layout():
    supported = VisibilityDiagnosticsWrapper(
        _FakeJointEnv(measurement_dim=7), scenario_name="rhythm_sync_dense", partner_occlusion_mode="none",
        cue_layout=CUE_LAYOUT_SCENARIO_MEASUREMENTS,
    )
    supported.partner_occlusion_mode = "both"
    obs = _step(supported)
    np.testing.assert_array_equal(obs[0]["measurements"], [1, 2, 3, 0, 0, 6, 7])
    assert np.all(obs[0]["obs"] == 40)

    global_state = cue_observation_layout(
        _cfg(full_observability=True, full_obs_mode="global_state"), doom_env_by_name("rhythm_sync_dense")
    )
    unsupported = VisibilityDiagnosticsWrapper(
        _FakeJointEnv(measurement_dim=40), scenario_name="rhythm_sync_dense", partner_occlusion_mode="none",
        cue_layout=global_state,
    )
    unsupported.partner_occlusion_mode = "both"
    with pytest.raises(ValueError, match="Cue masking"):
        _step(unsupported)
