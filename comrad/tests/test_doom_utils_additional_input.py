from comrad.envs.doom_params import default_doom_cfg
from comrad.utils import doom_utils
from comrad.utils.doom_utils import DOOM_ENVS, doom_env_by_name, get_extra_wrappers
from comrad.wrappers.scenario_wrappers import (
    ArmorySiegeAdditionalInput,
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

OPTIONAL_SUPPORTED_ENVS = {
    "armory_siege_vision",
    "foraging_commons_vision",
}


def _wrapper_classes(env_name: str, use_additional_input=None):
    cfg = default_doom_cfg(env=env_name)
    cfg.use_additional_input = use_additional_input
    wrappers = get_extra_wrappers(cfg, doom_env_by_name(env_name))
    return [wrapper_cls for wrapper_cls, _ in wrappers]


def test_default_additional_input_matches_previous_env_defaults():
    for doom_spec in DOOM_ENVS:
        wrapper_classes = _wrapper_classes(doom_spec.name)
        has_additional_input = doom_spec.additional_input_wrapper is not None and (
            doom_spec.additional_input_wrapper[0] in wrapper_classes
        )
        assert has_additional_input == (doom_spec.name in DEFAULT_ENABLED_ENVS)


def test_armory_siege_can_disable_additional_input():
    assert _wrapper_classes("armory_siege", use_additional_input=False) == [
        ArmorySiegeRewardShaping,
    ]


def test_armory_siege_vision_can_enable_additional_input():
    assert _wrapper_classes("armory_siege_vision", use_additional_input=True) == [
        ArmorySiegeAdditionalInput,
        ArmorySiegeRewardShaping,
    ]


def test_supported_optional_envs_can_enable_additional_input():
    for env_name in OPTIONAL_SUPPORTED_ENVS:
        doom_spec = doom_env_by_name(env_name)
        wrapper_classes = _wrapper_classes(env_name, use_additional_input=True)
        assert doom_spec.additional_input_wrapper is not None
        assert doom_spec.additional_input_wrapper[0] in wrapper_classes


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
