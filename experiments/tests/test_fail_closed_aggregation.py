import importlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

import experiments.iclr2027_rebuttal.aggregate_kd6j_results as agg
import experiments.iclr2027_rebuttal.rebuttal_kd6j_eval as kd6j
import experiments.very_difficult.very_difficult_reference_table as vd
from scripts.heldout_wad_evaluation import WadManifestEntry, joint_true_objective


def _run(scenario, algo, cell, seed, value=1.0):
    return {
        "scenario": scenario, "algorithm": algo, "cell_tag": cell, "train_seed": seed, "checkpoint_steps": 50_000_000,
        "normal_records": [{"true_objective": value}],
    }


def test_kd6j_aggregate_rejects_duplicate_seeds():
    runs = [_run("coop_puzzle", "MAPPO", "pixel_partial", 42), _run("coop_puzzle", "MAPPO", "pixel_partial", 42, 2.0)]
    with pytest.raises(agg.IncompleteResultsError, match="duplicate"):
        agg.aggregate(runs)


def test_kd6j_coverage_reports_missing_seeds_cells_and_unexpected_cells():
    seeds = {42, 68}
    expected = agg.expected_2x2(seeds)
    runs = [_run(s, a, c, seed) for (s, a, c) in expected for seed in seeds]
    assert agg.coverage_problems("2x2", agg.aggregate(runs), expected) == []

    dropped = ("armory_siege", "QMIX", "state_full")
    partial = [r for r in runs if (r["scenario"], r["algorithm"], r["cell_tag"]) != dropped or r["train_seed"] == 42]
    partial.append(_run("armory_siege", "MAPPO", "pixel_full_clean", 42))
    problems = agg.coverage_problems("2x2", agg.aggregate(partial), expected)
    assert problems == [
        "2x2: armory_siege/QMIX/state_full is missing train seeds [68]",
        "2x2: unexpected cell armory_siege/MAPPO/pixel_full_clean",
    ]


def test_kd6j_telemetry_cells_need_only_their_profile_seed():
    expected = agg.expected_full_obs({42, 68})
    assert expected[("lavapit", "MAPPO", "pixel_partial")] == {42}
    assert expected[("armory_siege", "MAPPO", "pixel_partial")] == {42, 68}


def test_kd6j_state_entity_ignores_out_of_protocol_seeds():
    entity = [_run("armory_siege", "MAPPO", "state_entity", 97), _run("armory_siege", "MAPPO", "pixel_partial", 42)]
    main = [_run("armory_siege", "MAPPO", "pixel_partial", 42)]
    # entity-dir seed 97 is outside the protocol and entity-dir Pixel-Partial seed 42 is reused from the main runs
    assert agg.state_entity_runs(main, entity) == main


def _vd_summary(env, algo, seeds, goals=None):
    return {"policy_summaries": [
        {"scenario": env, "algorithm": algo, "train_seed": s, "mean_true_objective": 0.5, "checkpoint_steps": 1,
         "episodes": 100, "run_dir": f"run_{env}_{algo}_{s}", "mean_goals_reached": goals}
        for s in seeds
    ]}


def _vd_complete(row_set):
    return [
        _vd_summary(env, algo, vd.EXPECTED_SEEDS, goals=1.0 if task == vd.LAVA_TASK else None)
        for _, env, algo, task, _ in vd.ROWS if env in vd.ROW_SETS[row_set]
    ]


@pytest.mark.parametrize("row_set", sorted(vd.ROW_SETS))
def test_very_difficult_table_requires_every_row_of_its_row_set(row_set):
    summaries = _vd_complete(row_set)
    assert len(vd.aggregate(summaries, row_set)) == sum(env in vd.ROW_SETS[row_set] for _, env, *_ in vd.ROWS)
    with pytest.raises(vd.IncompleteResultsError):
        vd.aggregate(summaries[1:], row_set)
    assert len(vd.aggregate(summaries[1:], row_set, allow_incomplete=True)) == len(summaries) - 1


def test_very_difficult_table_rejects_missing_seeds_and_foreign_policies():
    summaries = _vd_complete("scripted_partner")
    short = [_vd_summary("lava_maze_scripted_navigator", "IPPO", vd.EXPECTED_SEEDS[:4], goals=1.0)]
    with pytest.raises(vd.IncompleteResultsError):
        vd.aggregate([s for s in summaries if s["policy_summaries"][0]["scenario"] != "lava_maze_scripted_navigator"
                      or s["policy_summaries"][0]["algorithm"] != "IPPO"] + short, "scripted_partner")
    with pytest.raises(vd.IncompleteResultsError):
        vd.aggregate(summaries + [_vd_summary("lava_maze_oracle_hint", "MAPPO", vd.EXPECTED_SEEDS, 1.0)], "scripted_partner")


def test_very_difficult_discovery_keeps_only_the_row_set(tmp_path):
    for env in ("stealth_labyrinth_scripted_gunner", "stealth_labyrinth_lit"):
        path = tmp_path / env / "IPPO" / "seed_42" / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    found = vd.discover_summaries([tmp_path], vd.ROW_SETS["scripted_partner"])
    assert [p.parts[-4] for p in found] == ["stealth_labyrinth_scripted_gunner"]


def test_joint_true_objective_never_invents_a_score():
    assert joint_true_objective([{"true_objective": 2.0}, {"true_objective": 4.0}]) == 3.0
    assert joint_true_objective({"true_objective": 5}) == 5.0
    with pytest.raises(KeyError):
        joint_true_objective([{}, {}])


def test_visibility_rates_that_were_not_measured_are_left_out():
    stats = {key: None for key in kd6j.VISIBILITY_KEYS} | {"teammate_visible_rate": 0.25}
    assert kd6j._extract_visibility_from_infos([{"episode_visibility_stats": stats}, {}]) == {"teammate_visible_rate": 0.25}
    assert kd6j._extract_visibility_from_infos([{}, {}]) == {}
    summary = kd6j.aggregate_records_by_scenario_and_algo([
        {"scenario": "s", "algorithm": "A", "mode": "m", "train_seed": 42, "true_objective": 1.0, "teammate_visible_rate": 0.5},
        {"scenario": "s", "algorithm": "A", "mode": "m", "train_seed": 42, "true_objective": 1.0},
    ])[0]
    assert summary["teammate_visible_rate"] == 0.5
    assert "mutual_visibility_rate" not in summary


def test_random_cache_must_match_the_requested_protocol():
    entries = [WadManifestEntry("lavapit", f"w{i}", i, f"w{i}.wad", "sha", {}) for i in (1, 2)]
    records = [{"wad_id": e.wad_id, "train_seed": s} for e in entries for s in (42, 68)]
    assert kd6j.random_cache_mismatch(records, entries, (42, 68), 1) is None
    assert "episodes" in kd6j.random_cache_mismatch(records, entries, (42, 68), 2)
    assert "covers" in kd6j.random_cache_mismatch(records, entries, (42, 68, 81), 1)
    assert "covers" in kd6j.random_cache_mismatch(records[1:], entries, (42, 68), 1)


def test_learning_curves_drop_or_reject_early_stopped_seeds(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    curves_ci = importlib.import_module("baseline_learning_curves_ci")
    grid = np.linspace(0.0, curves_ci.BUDGET_STEPS, curves_ci.GRID_POINTS)
    full = np.ones_like(grid)
    stopped = np.where(grid <= 0.25 * curves_ci.BUDGET_STEPS, 5.0, np.nan)
    curves = {"coop_puzzle": {"MAPPO": {"42": full, "68": stopped}}}
    complete, incomplete = curves_ci.split_incomplete(curves, grid)
    assert list(complete["coop_puzzle"]["MAPPO"]) == ["42"]
    assert incomplete["coop_puzzle"]["MAPPO"] == {"68": pytest.approx(0.25 * curves_ci.BUDGET_STEPS)}

    heldout = {s: {a: (0.0, 0.0) for a in curves_ci.ALGORITHM_ORDER} for s in curves_ci.SCENARIO_ORDER}
    entry = curves_ci.summarize(complete, heldout, 2, incomplete)["coop_puzzle"]["MAPPO"]
    assert entry["seeds"] == ["42"] and entry["train_final_mean"] == 1.0
    assert entry["dropped_incomplete_seeds"] == {"68": pytest.approx(0.25 * curves_ci.BUDGET_STEPS)}
    json.dumps(entry)


def test_kd6j_metric_measured_on_fewer_seeds_than_its_cell_is_reported():
    runs = [_run("armory_siege", "MAPPO", "pixel_partial", seed) for seed in (42, 68)]
    runs[0]["normal_records"][0]["teammate_visible_rate"] = 0.5
    table = agg.aggregate(runs)
    assert table["armory_siege"]["MAPPO"]["pixel_partial"]["teammate_visible_rate"]["seeds"] == [42]
    expected = {("armory_siege", "MAPPO", "pixel_partial"): {42, 68}}
    assert agg.coverage_problems("x", table, expected) == [
        "x: armory_siege/MAPPO/pixel_partial teammate_visible_rate covers seeds [42] of [42, 68]"
    ]


def test_kd6j_results_from_an_older_protocol_are_never_aggregated(tmp_path):
    from experiments.iclr2027_rebuttal.kd6j_protocol import KD6J_PACK_PROTOCOL_VERSION, is_current_result_file

    current = {**_run("coop_puzzle", "MAPPO", "pixel_partial", 42), "protocol_version": KD6J_PACK_PROTOCOL_VERSION}
    (tmp_path / "current.json").write_text(json.dumps(current))
    assert agg.load_runs(tmp_path, min_steps=1) == [current]
    assert is_current_result_file(tmp_path / "current.json")

    (tmp_path / "legacy.json").write_text(json.dumps(_run("coop_puzzle", "MAPPO", "pixel_partial", 68)))
    (tmp_path / "partial.json").write_text('{"protocol_version": "kd6j')
    assert not is_current_result_file(tmp_path / "legacy.json")
    assert not is_current_result_file(tmp_path / "partial.json")
    assert not is_current_result_file(tmp_path / "missing.json")
    (tmp_path / "partial.json").unlink()
    with pytest.raises(agg.IncompleteResultsError, match="legacy.json"):
        agg.load_runs(tmp_path, min_steps=1)
