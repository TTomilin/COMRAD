import pytest

from comrad.envs.doom_params import default_doom_cfg
from comrad.utils import doom_utils
from comrad.utils.doom_utils import doom_env_by_name, get_extra_wrappers
from comrad.wrappers.scenario_wrappers import (
    ArmorySiegeRewardShaping,
)

DEFAULT_ENABLED_ENVS = {
    "lavapit",
    "armory_siege",
    "lava_maze",
    "ammo_carrier",
    "foraging_commons",
    "rhythm_sync",
    "rhythm_sync_dense",
}

def _wrapper_classes(env_name: str, use_additional_input=None):
    cfg = default_doom_cfg(env=env_name)
    cfg.use_additional_input = use_additional_input
    wrappers = get_extra_wrappers(cfg, doom_env_by_name(env_name))
    return [wrapper_cls for wrapper_cls, _ in wrappers]


def test_cli_default_is_unset_so_spec_default_applies():
    cfg = default_doom_cfg(env="armory_siege")
    assert cfg.use_additional_input is None
    # canonical specs keep use_additional_input=False, so the default stack is unchanged
    assert _wrapper_classes("armory_siege") == [ArmorySiegeRewardShaping]


def test_spec_default_true_enables_wrapper_without_cli_flag(monkeypatch):
    spec = doom_env_by_name("armory_siege")
    monkeypatch.setattr(spec, "use_additional_input", True)
    assert _wrapper_classes("armory_siege")[0] is spec.additional_input_wrapper[0]
    assert _wrapper_classes("armory_siege", use_additional_input=False) == [ArmorySiegeRewardShaping]


def test_armory_siege_can_disable_additional_input():
    assert _wrapper_classes("armory_siege", use_additional_input=False) == [
        ArmorySiegeRewardShaping,
    ]


def test_full_observability_does_not_implicitly_enable_additional_input():
    # Pixel/Full must see the same scenario features as Pixel/Partial (only the partner's view/pose is added).
    for env_name in ("armory_siege", "rhythm_sync_dense"):
        cfg = default_doom_cfg(env=env_name)
        cfg.full_observability = True
        assert doom_utils.get_use_additional_input(cfg, doom_env_by_name(env_name)) is False
        cfg.use_additional_input = True
        assert doom_utils.get_use_additional_input(cfg, doom_env_by_name(env_name)) is True


def test_archived_legacy_full_obs_runs_resolve_to_additional_input():
    resolve = doom_utils.resolve_archived_use_additional_input
    # legacy Pixel/Full runs relied on the old implicit default, their checkpoints expect the task scalars
    assert resolve({"full_observability": True, "use_additional_input": None}) is True
    # clean runs pass the flag explicitly
    assert resolve({"full_observability": True, "use_additional_input": False}) is False
    assert resolve({"full_observability": False, "use_additional_input": None}) is None
    assert resolve({}) is None


def test_full_obs_mode_defaults_to_shared_vision():
    cfg = default_doom_cfg(env="armory_siege")
    assert cfg.full_obs_mode == "shared_vision"
    assert doom_utils.get_full_obs_mode(cfg) is None
    cfg.full_observability = True
    assert doom_utils.get_full_obs_mode(cfg) == "shared_vision"
    del cfg.full_obs_mode  # archived configs predate the flag
    assert doom_utils.get_full_obs_mode(cfg) == "shared_vision"
    cfg.full_obs_mode = "global_state"
    assert doom_utils.get_full_obs_mode(cfg) == "global_state"


def test_rhythm_sync_full_feature_set_only_in_shared_vision_mode():
    spec = doom_env_by_name("rhythm_sync")
    for mode, expected in (("shared_vision", "full"), ("global_state", None)):
        cfg = default_doom_cfg(env="rhythm_sync")
        cfg.full_observability = True
        cfg.full_obs_mode = mode
        cfg.use_additional_input = True
        w_cls, w_kwargs = get_extra_wrappers(cfg, spec)[0]
        assert w_cls is spec.additional_input_wrapper[0]
        assert w_kwargs.get("feature_set") == (expected or spec.additional_input_wrapper[1].get("feature_set"))


def test_global_state_horizon_uses_engine_timeout_and_frameskip():
    cfg = default_doom_cfg(env="coop_puzzle")
    cfg.env_frameskip = 4
    spec = doom_env_by_name("coop_puzzle")
    tics = min(2000, int((cfg.timelimit or spec.timelimit) * 60 * 35))
    assert doom_utils.global_state_horizon_steps(spec, cfg) == -(-tics // 4)


def test_state_encoding_flags_default_to_ego_and_archived_configs_resolve_to_ego():
    cfg = default_doom_cfg(env="armory_siege")
    assert cfg.state_encoding == "ego"
    assert cfg.state_sight_range == 512.0
    assert doom_utils.uses_entity_state_encoding(cfg) is False
    cfg.state_encoding = "entity"
    assert doom_utils.uses_entity_state_encoding(cfg) is False  # pixel modality ignores the state encoding
    cfg.obs_modality = "state"
    assert doom_utils.uses_entity_state_encoding(cfg) is True
    del cfg.state_encoding  # archived configs predate the flag
    assert doom_utils.get_state_encoding(cfg) == "ego"
    assert doom_utils.uses_entity_state_encoding(cfg) is False


class _Recorder:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        raise _Captured(kwargs)


class _Captured(Exception):
    def __init__(self, kwargs):
        super().__init__("captured")
        self.kwargs = kwargs


@pytest.mark.parametrize(
    "player_id, encoding, full_obs_mode, expected",
    [
        (0, "entity", None, dict(enable_global_state_info=True, global_state_geometry=False, enable_depth_row=True)),
        (1, "entity", None, dict(enable_global_state_info=False, global_state_geometry=False, enable_depth_row=True)),
        (0, "ego", None, dict(enable_global_state_info=False, global_state_geometry=False, enable_depth_row=False)),
        (0, "ego", "global_state", dict(enable_global_state_info=True, global_state_geometry=True, enable_depth_row=False)),
        (1, "ego", "global_state", dict(enable_global_state_info=False, global_state_geometry=True, enable_depth_row=False)),
    ],
)
def test_entity_encoding_flags_reach_the_engine_env(monkeypatch, player_id, encoding, full_obs_mode, expected):
    from comrad.envs.multiagent import doom_multiagent

    monkeypatch.setattr(doom_multiagent, "VizdoomEnvMultiplayer", _Recorder)
    cfg = default_doom_cfg(env="armory_siege")
    cfg.obs_modality = "state"
    cfg.state_encoding = encoding
    if full_obs_mode:
        cfg.full_observability = True
        cfg.full_obs_mode = full_obs_mode
    with pytest.raises(_Captured) as captured:
        doom_utils.make_doom_env_impl(doom_env_by_name("armory_siege"), cfg=cfg, player_id=player_id, num_agents=2,
                                      max_num_players=2)
    for key, value in expected.items():
        assert captured.value.kwargs[key] is value, key


def test_state_modality_still_enables_additional_input():
    cfg = default_doom_cfg(env="armory_siege")
    cfg.obs_modality = "state"
    assert doom_utils.get_use_additional_input(cfg, doom_env_by_name("armory_siege")) is True


def test_unsupported_env_warns_when_additional_input_is_forced(monkeypatch):
    warnings = []

    def fake_warning(msg, *args):
        warnings.append(msg % args)

    monkeypatch.setattr(doom_utils.log, "warning", fake_warning)
    doom_utils._WARNED_UNSUPPORTED_ADDITIONAL_INPUT_ENVS.clear()

    wrapper_classes = _wrapper_classes("platform_chain", use_additional_input=True)

    assert wrapper_classes
    assert warnings == [
        "Scenario platform_chain does not define an additional input wrapper, ignoring --use_additional_input=True"
    ]


def test_unsupported_env_warning_is_emitted_once(monkeypatch):
    warnings = []

    def fake_warning(msg, *args):
        warnings.append(msg % args)

    monkeypatch.setattr(doom_utils.log, "warning", fake_warning)
    doom_utils._WARNED_UNSUPPORTED_ADDITIONAL_INPUT_ENVS.clear()

    _wrapper_classes("platform_chain", use_additional_input=True)
    _wrapper_classes("platform_chain", use_additional_input=True)

    assert len(warnings) == 1
