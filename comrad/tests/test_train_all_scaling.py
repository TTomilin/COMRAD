import importlib
import os
import re
import sys
from pathlib import Path


def _load_train_all(profile: str):
    old_profile = os.environ.get("COMRAD_TRAIN_ALL_PROFILE")
    try:
        os.environ["COMRAD_TRAIN_ALL_PROFILE"] = profile
        if "comrad.train_all" in sys.modules:
            return importlib.reload(sys.modules["comrad.train_all"])
        return importlib.import_module("comrad.train_all")
    finally:
        if old_profile is None:
            os.environ.pop("COMRAD_TRAIN_ALL_PROFILE", None)
        else:
            os.environ["COMRAD_TRAIN_ALL_PROFILE"] = old_profile


def _generated_runs(profile: str):
    train_all = _load_train_all(profile)
    runs = list(train_all.RUN_DESCRIPTION.generate_experiments("/tmp/train_all", makedirs=False))
    return train_all, runs


def _extract_arg(cmd: str, name: str) -> str:
    match = re.search(rf"--{re.escape(name)}=([^\s]+)", cmd)
    assert match is not None, cmd
    return match.group(1)


def test_agent_scaling_profile_generates_explicit_armory_siege_sweep():
    train_all, runs = _generated_runs("agent_scaling")

    expected_runs = len(train_all.AGENT_SCALING_AGENT_COUNTS) * len(train_all.SEEDS)
    assert train_all.ACTIVE_PROFILE == train_all.PROFILE_AGENT_SCALING
    assert train_all.BENCHMARK_NAME == train_all.AGENT_SCALING_NAME
    assert train_all.TOTAL_RUNS == expected_runs
    assert len(runs) == expected_runs

    seen_agent_counts = set()
    root_names = set()

    for cmd, _, root_dir, _ in runs:
        root_names.add(Path(root_dir).name)
        seen_agent_counts.add(int(_extract_arg(cmd, "num_agents")))

        assert "--algo=QMIX" in cmd
        assert "--mixer=qmix" in cmd
        assert "--env=armory_siege" in cmd
        assert "--batch_size=2304" in cmd
        assert "--qmix_sequence_batch_size=32" in cmd
        assert "--shared_reward_alpha=" not in cmd
        assert "--train_for_env_steps=125000000" in cmd
        assert "--train_for_seconds=" not in cmd
        assert cmd.count("--experiment=") == 1
        assert cmd.count("--train_dir=") == 1
        assert "$COMRAD_RUN_NAME" not in cmd

    assert seen_agent_counts == set(train_all.AGENT_SCALING_AGENT_COUNTS)
    assert root_names == {train_all.AGENT_SCALING_ROOT_DIR}


def test_invalid_train_all_profile_is_rejected():
    try:
        _load_train_all("not_a_profile")
    except ValueError as exc:
        assert "COMRAD_TRAIN_ALL_PROFILE" in str(exc)
    else:
        raise AssertionError("Expected invalid COMRAD_TRAIN_ALL_PROFILE to raise ValueError")
