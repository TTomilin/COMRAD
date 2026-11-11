from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "heldout_wad_evaluation.py"
SPEC = importlib.util.spec_from_file_location("heldout_wad_evaluation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
heldout = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = heldout
SPEC.loader.exec_module(heldout)


def write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")


def test_archived_algo_requires_value(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"env": "lava_maze"}), encoding="utf-8")
    with pytest.raises(ValueError):
        heldout.archived_algo(tmp_path)


def test_select_checkpoint_prefers_periodic_checkpoint_at_equal_distance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_checkpoint(run_dir / "checkpoint_p0" / "best_000000001_100000000_true_objective_1.000.pth")
    periodic = run_dir / "checkpoint_p0" / "checkpoint_000000002_100000000.pth"
    write_checkpoint(periodic)

    selected, steps = heldout.select_checkpoint(run_dir, target_steps=100_000_000)

    assert selected == periodic
    assert steps == 100_000_000


def test_discover_policy_specs_records_config_and_checkpoint_hashes(tmp_path: Path) -> None:
    run_dir = tmp_path / "ammo_carrier" / "00_IPPO_env_ammo_carrier_seed_42"
    run_dir.mkdir(parents=True)
    config_path = run_dir / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    checkpoint_path = run_dir / "checkpoint_p0" / "checkpoint_000000002_100000000.pth"
    write_checkpoint(checkpoint_path)

    policies = heldout.discover_policy_specs(tmp_path, ["ammo_carrier"], ["IPPO"], target_steps=100_000_000)

    assert len(policies) == 1
    policy = policies[0]
    assert policy.run_dir == str(run_dir)
    assert policy.checkpoint == str(checkpoint_path)
    assert policy.checkpoint_steps == 100_000_000
    assert policy.config_sha256 == heldout.sha256_file(config_path)
    assert policy.checkpoint_sha256 == heldout.sha256_file(checkpoint_path)


def test_singleton_batch_uses_absolute_wad_path_and_seed(tmp_path: Path) -> None:
    wad_path = tmp_path / "heldout.wad"
    wad_path.write_bytes(b"wad")
    entry = heldout.WadManifestEntry(
        scenario="ammo_carrier",
        wad_id="ammo_carrier_heldout_seed_1009",
        seed=1009,
        filename=wad_path.name,
        sha256=heldout.sha256_file(wad_path),
        config={"seed": 1009},
    )

    with heldout.singleton_batch_dir(tmp_path, entry) as singleton:
        payload = json.loads((Path(singleton) / "batch_registry.json").read_text(encoding="utf-8"))

    assert payload == [{"id": entry.wad_id, "filename": str(wad_path.resolve()), "seed": 1009}]


def test_render_table_uses_layout_level_ci_and_policy_order() -> None:
    summary = {
        "policy_summaries": [
            {"scenario": "ammo_carrier", "algorithm": "IPPO", "mean_true_objective": 12.0, "layout_mean_ci95": 0.4},
            {"scenario": "ammo_carrier", "algorithm": "MAPPO", "mean_true_objective": 10.0, "layout_mean_ci95": 0.2},
        ]
    }

    table = heldout.render_table(summary)

    assert "Scenario & IPPO & MAPPO" in table
    assert "Ammo Carrier & 12.00 $\\pm$ 0.40 & 10.00 $\\pm$ 0.20" in table


def _make_run(root: Path, name: str) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    write_checkpoint(run_dir / "checkpoint_p0" / "checkpoint_000000002_100000000.pth")
    return run_dir


def test_discover_seeded_policy_specs_matches_train_all_names_and_skips_failed(tmp_path: Path) -> None:
    root = tmp_path / "comrad_very_difficult_scripted" / "very_difficult_scripted"
    ippo = _make_run(root, "00_IPPO_env_stealth_labyrinth_scripted_scout_s.r.alp_1.0_see_42")
    _make_run(root, "01_IPPO_env_stealth_labyrinth_scripted_scout_s.r.alp_1.0_see_142")
    idqn = _make_run(root, "05_IDQN_env_stealth_labyrinth_scripted_scout_see_42")
    _make_run(root, "failed_05_IDQN_env_stealth_labyrinth_scripted_scout_see_68")
    idqn_68 = _make_run(root, "06_IDQN_env_stealth_labyrinth_scripted_scout_see_68")
    _make_run(root, "10_IPPO_env_lava_maze_scripted_observer_s.r.alp_1.0_see_42")

    policies = heldout.discover_seeded_policy_specs(
        tmp_path, ["stealth_labyrinth_scripted_scout"], ["IPPO", "IDQN"], [42], target_steps=100_000_000
    )
    assert [(p.algorithm, p.train_seed, p.run_dir) for p in policies] == [
        ("IPPO", 42, str(ippo)),
        ("IDQN", 42, str(idqn)),
    ]
    (policy,) = heldout.discover_seeded_policy_specs(
        tmp_path, ["stealth_labyrinth_scripted_scout"], ["IDQN"], [68], target_steps=100_000_000
    )
    assert policy.run_dir == str(idqn_68)
    with pytest.raises(FileNotFoundError):
        heldout.discover_seeded_policy_specs(
            tmp_path, ["stealth_labyrinth_scripted_scout"], ["IPPO"], [97], target_steps=100_000_000
        )


def test_seeded_pattern_distinguishes_variants_and_qplex_token() -> None:
    pattern = heldout.seeded_run_pattern("stealth_labyrinth", "QPLEX-D", 42)
    assert pattern.search("00_QPLEX_dmaq_env_stealth_labyrinth_see_42")
    assert not pattern.search("00_QPLEX_dmaq_env_stealth_labyrinth_lit_see_42")
    hint = heldout.seeded_run_pattern("lava_maze_oracle_hint", "MAPPO", 81)
    assert hint.search("07_MAPPO_env_lava_maze_oracle_hint_s.r.alp_1.0_see_81")
    assert not hint.search("07_MAPPO_env_lava_maze_oracle_msg_s.r.alp_1.0_see_81")


def test_generate_heldout_batch_passes_variant_overrides(tmp_path: Path, monkeypatch) -> None:
    created = []

    class FakeScenario:
        def __init__(self, config, name=None):
            self.config = {"seed": None, "scripted_scout": False, "always_lit": False, **config}
            created.append((name, dict(self.config)))

        def generate(self, path):
            Path(path).write_bytes(repr(sorted(self.config.items())).encode())

    fake_module = type(sys)("fake_stealth")
    fake_module.StealthLabyrinthScenario = FakeScenario
    monkeypatch.setitem(sys.modules, "examples.benchmark.stealth_labyrinth", fake_module)

    entries = heldout.generate_heldout_batch("stealth_labyrinth_scripted_scout", tmp_path / "b", [1009, 2027])
    assert [c["scripted_scout"] for _, c in created] == [True, True]
    assert [c["seed"] for _, c in created] == [1009, 2027]
    assert entries[0].scenario == "stealth_labyrinth_scripted_scout"
    manifest = json.loads((tmp_path / "b" / "batch_registry.json").read_text())
    assert manifest[0]["config"]["scripted_scout"] is True

    created.clear()
    heldout.generate_heldout_batch("stealth_labyrinth_lit", tmp_path / "c", [1009])
    assert created[0][1]["always_lit"] is True and created[0][1]["scripted_scout"] is False
    with pytest.raises(ValueError):
        heldout.resolve_generator("unknown_env")


def test_protocol_rejects_not_exactly_five_distinct_seeds() -> None:
    assert heldout.HELDOUT_SEEDS == (1009, 2027, 3037, 4051, 5099)
    assert len(set(heldout.HELDOUT_SEEDS)) == 5
    assert 42 not in heldout.HELDOUT_SEEDS


def _archive_run(tmp_path: Path, algo: str, env: str = "lava_maze_oracle_msg") -> Path:
    """Write a real sample_factory config.json for a finished run (as the learner does)."""
    run_dir = tmp_path / f"00_{algo}_env_{env}_see_42"
    run_dir.mkdir(parents=True)
    cfg = heldout.parse_comrad_args(
        argv=[f"--algo={algo}", f"--env={env}", f"--experiment={run_dir.name}", f"--train_dir={tmp_path}"],
        evaluation=True,
    )
    payload = {key: value for key, value in vars(cfg).items() if key != "cli_args"}
    (run_dir / "config.json").write_text(json.dumps(payload, default=str), encoding="utf-8")
    return run_dir


@pytest.mark.parametrize("algo", ["MAPPO", "QMIX", "IDQN", "IPPO"])
def test_build_eval_cfg_keeps_archived_algo(tmp_path: Path, monkeypatch, algo: str) -> None:
    # regression: a hard-coded --algo=APPO on the eval command line overrode the archived algo in
    # load_from_checkpoint, so MAPPO checkpoints failed to load and QMIX/IDQN were evaluated as APPO
    run_dir = _archive_run(tmp_path, algo)
    monkeypatch.setattr(heldout, "configure_batch_env_and_agents", lambda cfg: None)
    monkeypatch.setattr(heldout, "register_model_factory", lambda cfg: None)
    policy = heldout.PolicySpec(
        scenario="lava_maze_oracle_msg",
        algorithm=algo,
        run_dir=str(run_dir),
        checkpoint=str(run_dir / "checkpoint_p0" / "checkpoint_000000001_100000000.pth"),
        checkpoint_steps=100_000_000,
        config_sha256="x",
        checkpoint_sha256="y",
        train_seed=42,
    )

    cfg = heldout.build_eval_cfg(policy, tmp_path / "batch", device="cpu", deterministic=False)

    assert cfg.algo == algo
    assert heldout.archived_algo(run_dir) == algo


def test_command_line_algo_overrides_archive_which_is_why_eval_passes_archived_algo(tmp_path: Path) -> None:
    run_dir = _archive_run(tmp_path, "MAPPO")
    cfg = heldout.parse_comrad_args(
        argv=["--algo=APPO", f"--train_dir={tmp_path}", f"--experiment={run_dir.name}", "--env=lava_maze_oracle_msg"],
        evaluation=True,
    )
    assert heldout.load_from_checkpoint(cfg).algo == "APPO"


def test_greedy_q_actions_matches_inference_worker_rule() -> None:
    torch = pytest.importorskip("torch")
    head_sizes = [3, 3, 2, 5]
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.0, 2.0, 1.0, 0.0, 0.3, 0.2, 0.0, 0.0, 0.0, 5.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 1.0, 9.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert heldout.greedy_q_actions(logits, head_sizes).tolist() == [[1, 0, 0, 3], [0, 2, 1, 0]]
    assert heldout.greedy_q_actions(torch.tensor([[0.0, 2.0, 1.0]]), [3]).tolist() == [1]


def test_action_head_sizes_for_tuple_and_discrete_spaces() -> None:
    from gymnasium import spaces

    assert heldout.action_head_sizes(spaces.Discrete(4)) == [4]
    assert heldout.action_head_sizes(spaces.Tuple([spaces.Discrete(3), spaces.Discrete(5)])) == [3, 5]


def test_goal_relocation_counter_ignores_repeated_level_ticks() -> None:
    counter = heldout.GoalRelocationCounter()
    frames = [
        {"USER13": -1, "USER18": -1, "USER22": 0},  # before GenerateMaze finished
        {"USER13": 3, "USER18": 4, "USER22": 0},  # first goal: not a completion
        {"USER13": 3, "USER18": 4, "USER22": 1},
        {"USER13": 3, "USER18": 4, "USER22": 2},  # ACS counted the same arrival twice
        {"USER13": 7, "USER18": 1, "USER22": 2},  # goal relocated -> one real completion
        {"USER13": 7, "USER18": 1, "USER22": 3},
        {"USER13": 0, "USER18": 6, "USER22": 4},
    ]
    for info in frames:
        counter.observe([info, dict(info)])
    assert counter.count == 2
    counter.observe([{}])
    assert counter.count == 2
    counter.reset()
    counter.observe({"USER13": 0, "USER18": 6})
    assert counter.count == 0


def test_summarize_policy_reports_goals_when_tracked() -> None:
    policy = heldout.PolicySpec("lava_maze_oracle_hint", "MAPPO", "r", "c", 1, "a", "b", 42)
    records = [
        {"wad_id": "w1", "true_objective": 4.0, "goals_reached": 2},
        {"wad_id": "w2", "true_objective": 6.0, "goals_reached": 3},
    ]
    summary = heldout.summarize_policy(records, policy)
    assert summary["mean_true_objective"] == 5.0
    assert summary["mean_goals_reached"] == 2.5
    assert "mean_goals_reached" not in heldout.summarize_policy([{"wad_id": "w", "true_objective": 1.0}], policy)
