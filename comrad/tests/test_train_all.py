import re
from pathlib import Path

from comrad import train_all


def _generated_runs():
    return list(train_all.RUN_DESCRIPTION.generate_experiments("/tmp/train_all", makedirs=False))


def _base_name(root_dir: str) -> str:
    return Path(root_dir).name.removesuffix("_")


def _env_from_cmd(cmd: str) -> str:
    match = re.search(r"--env=([^\s]+)", cmd)
    assert match is not None, cmd
    return match.group(1)


def test_total_run_count_matches_full_benchmark_matrix():
    expected_algorithms = 8  # IPPO, MAPPO, HAPPO, IDQN, VDN, QMIX, QPLEX(dmaq), QPLEX(qatten)
    expected_runs = len(train_all.BENCHMARK_SCENARIOS) * expected_algorithms * len(train_all.SEEDS)

    runs = _generated_runs()

    assert train_all.TOTAL_RUNS == expected_runs
    assert len(runs) == expected_runs


def test_launcher_fields_are_only_injected_once():
    for cmd, _, _, _ in _generated_runs():
        assert cmd.count("--experiment=") == 1
        assert cmd.count("--train_dir=") == 1
        assert "$COMRAD_RUN_NAME" not in cmd


def test_benchmark_uses_fixed_env_step_budget():
    for cmd, _, _, _ in _generated_runs():
        assert "--train_for_env_steps=125000000" in cmd
        assert "--train_for_seconds=" not in cmd


def test_actor_critic_shared_reward_matches_benchmark_contract():
    expected_shared = {
        scenario.env: scenario.actor_critic_shared_reward for scenario in train_all.BENCHMARK_SCENARIOS
    }

    for cmd, _, root_dir, _ in _generated_runs():
        base_name = _base_name(root_dir)
        if base_name not in {"IPPO", "MAPPO", "HAPPO"}:
            assert "--shared_reward_alpha=" not in cmd
            continue

        env = _env_from_cmd(cmd)
        has_shared_reward = "--shared_reward_alpha=1.0" in cmd
        assert has_shared_reward is expected_shared[env]


def test_qplex_variants_are_separate_experiments():
    qplex_runs = [
        (cmd, _base_name(root_dir))
        for cmd, _, root_dir, _ in _generated_runs()
        if _base_name(root_dir).startswith("QPLEX_")
    ]

    dmaq_runs = [cmd for cmd, base_name in qplex_runs if base_name == "QPLEX_dmaq"]
    qatten_runs = [cmd for cmd, base_name in qplex_runs if base_name == "QPLEX_dmaq_qatten"]

    assert len(dmaq_runs) == len(train_all.BENCHMARK_SCENARIOS) * len(train_all.SEEDS)
    assert len(qatten_runs) == len(train_all.BENCHMARK_SCENARIOS) * len(train_all.SEEDS)

    for cmd in dmaq_runs:
        assert "--algo=QPLEX" in cmd
        assert "--mixer=dmaq" in cmd
        assert "--qmix_sequence_batch_size=16" in cmd
        assert "--qplex_grad_accum_mini_bs=16" in cmd
        assert "--qplex_state_bias=False" in cmd

    for cmd in qatten_runs:
        assert "--algo=QPLEX" in cmd
        assert "--mixer=dmaq_qatten" in cmd
        assert "--qmix_sequence_batch_size=16" in cmd
        assert "--qplex_grad_accum_mini_bs=16" in cmd
        assert "--qplex_state_bias=False" in cmd
