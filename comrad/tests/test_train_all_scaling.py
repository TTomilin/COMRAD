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
        num_agents = int(_extract_arg(cmd, "num_agents"))
        seen_agent_counts.add(num_agents)

        assert "--algo=QMIX" in cmd
        assert "--mixer=qmix" in cmd
        assert "--env=armory_siege" in cmd
        assert f"--batch_size={train_all.AGENT_SCALING_BATCH_SIZE_BY_NUM_AGENTS[num_agents]}" in cmd
        assert "--qmix_sequence_batch_size=32" in cmd
        assert "--learning_rate=0.0001" in cmd
        assert "--shared_reward_alpha=" not in cmd
        assert "--train_for_env_steps=125000000" in cmd
        assert "--train_for_seconds=" not in cmd
        assert cmd.count("--experiment=") == 1
        assert cmd.count("--train_dir=") == 1
        assert "$COMRAD_RUN_NAME" not in cmd

    assert seen_agent_counts == set(train_all.AGENT_SCALING_AGENT_COUNTS)
    assert root_names == {train_all.AGENT_SCALING_ROOT_DIR}


def test_qmix_lr_sensitivity_pilot_profile_generates_expected_lr_sweep():
    train_all, runs = _generated_runs("qmix_lr_sensitivity_pilot")

    expected_runs = len(train_all.QMIX_LR_SENSITIVITY_LR_CANDIDATES) * len(train_all.SEEDS)
    assert train_all.ACTIVE_PROFILE == train_all.PROFILE_QMIX_LR_SENSITIVITY_PILOT
    assert train_all.BENCHMARK_NAME == train_all.QMIX_LR_SENSITIVITY_PILOT_NAME
    assert train_all.TOTAL_RUNS == expected_runs
    assert len(runs) == expected_runs

    seen_lrs = set()
    root_names = set()

    for cmd, _, root_dir, _ in runs:
        root_names.add(Path(root_dir).name)
        seen_lrs.add(float(_extract_arg(cmd, "learning_rate")))

        assert "--algo=QMIX" in cmd
        assert "--mixer=qmix" in cmd
        assert "--env=armory_siege" in cmd
        assert f"--num_agents={train_all.QMIX_LR_SENSITIVITY_NUM_AGENTS}" in cmd
        assert f"--batch_size={train_all.QMIX_LR_SENSITIVITY_BATCH_SIZE}" in cmd
        assert "--qmix_sequence_batch_size=32" in cmd
        assert f"--train_for_env_steps={train_all.QMIX_LR_SENSITIVITY_PILOT_STEPS}" in cmd
        assert cmd.count("--learning_rate=") == 1
        assert "--shared_reward_alpha=" not in cmd
        assert "--train_for_seconds=" not in cmd
        assert cmd.count("--experiment=") == 1
        assert cmd.count("--train_dir=") == 1

    assert seen_lrs == set(train_all.QMIX_LR_SENSITIVITY_LR_CANDIDATES)
    assert root_names == {train_all.QMIX_LR_SENSITIVITY_ROOT_DIR}


def test_qmix_lr_sensitivity_full_profile_generates_single_confirmation_run():
    train_all, runs = _generated_runs("qmix_lr_sensitivity_full")

    assert train_all.ACTIVE_PROFILE == train_all.PROFILE_QMIX_LR_SENSITIVITY_FULL
    assert train_all.BENCHMARK_NAME == train_all.QMIX_LR_SENSITIVITY_FULL_NAME
    assert train_all.TOTAL_RUNS == len(train_all.SEEDS)
    assert len(runs) == len(train_all.SEEDS)

    for cmd, _, root_dir, _ in runs:
        assert Path(root_dir).name == train_all.QMIX_LR_SENSITIVITY_ROOT_DIR
        assert "--algo=QMIX" in cmd
        assert "--mixer=qmix" in cmd
        assert "--env=armory_siege" in cmd
        assert f"--num_agents={train_all.QMIX_LR_SENSITIVITY_NUM_AGENTS}" in cmd
        assert f"--batch_size={train_all.QMIX_LR_SENSITIVITY_BATCH_SIZE}" in cmd
        assert f"--learning_rate={train_all.QMIX_LR_SENSITIVITY_REPRESENTATIVE_LR}" in cmd
        assert cmd.count("--learning_rate=") == 1
        assert f"--train_for_env_steps={train_all.QMIX_LR_SENSITIVITY_FULL_STEPS}" in cmd
        assert "--qmix_sequence_batch_size=32" in cmd
        assert "--shared_reward_alpha=" not in cmd
        assert "--train_for_seconds=" not in cmd
        assert cmd.count("--experiment=") == 1
        assert cmd.count("--train_dir=") == 1


def test_invalid_train_all_profile_is_rejected():
    try:
        _load_train_all("not_a_profile")
    except ValueError as exc:
        assert "COMRAD_TRAIN_ALL_PROFILE" in str(exc)
    else:
        raise AssertionError("Expected invalid COMRAD_TRAIN_ALL_PROFILE to raise ValueError")


def test_benchmark_profile_does_not_enable_wad_batches():
    train_all, runs = _generated_runs("benchmark")

    assert train_all.ACTIVE_PROFILE == train_all.PROFILE_FULL_BENCHMARK
    assert len(runs) == train_all.TOTAL_RUNS

    for cmd, _, _, _ in runs:
        assert "--wad_batch=" not in cmd


def test_platform_chain_curriculum_profile_only_batchifies_non_baseline_variants():
    train_all, runs = _generated_runs("platform_chain_curriculum")

    assert train_all.ACTIVE_PROFILE == train_all.PROFILE_PLATFORM_CHAIN_CURRICULUM
    assert len(runs) == train_all.TOTAL_RUNS == len(train_all.PLATFORM_CHAIN_CURRICULUM_VARIANTS)

    saw_baseline = False
    saw_batch_variant = False

    for cmd, _, root_dir, _ in runs:
        assert Path(root_dir).name == train_all.PLATFORM_CHAIN_CURRICULUM_ROOT_DIR
        assert "--env=platform_chain" in cmd

        if "_platform_chain_baseline_" in cmd:
            saw_baseline = True
            assert "--wad_batch=" not in cmd
        else:
            saw_batch_variant = True
            assert (
                "--wad_batch=comrad/scenarios/batch_platform_chain_curriculum" in cmd
            )

    assert saw_baseline
    assert saw_batch_variant
