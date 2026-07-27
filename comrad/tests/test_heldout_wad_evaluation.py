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


def test_protocol_rejects_not_exactly_five_distinct_seeds() -> None:
    assert heldout.HELDOUT_SEEDS == (1009, 2027, 3037, 4051, 5099)
    assert len(set(heldout.HELDOUT_SEEDS)) == 5
    assert 42 not in heldout.HELDOUT_SEEDS
