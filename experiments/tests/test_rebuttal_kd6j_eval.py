from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

import experiments.iclr2027_rebuttal.rebuttal_kd6j_eval as kd6j
import scripts.heldout_wad_evaluation as heldout


def test_all_13_benchmark_scenarios_registered_and_random_evaluator_aggregates() -> None:
    assert len(kd6j.ALL_13_SCENARIOS) == 13
    for scenario in kd6j.ALL_13_SCENARIOS:
        gen, overrides = heldout.resolve_generator(scenario)
        assert gen in heldout.SCENARIO_CLASSES
        assert isinstance(overrides, dict)

    records = [
        {
            "scenario": "stag_hunt_arena",
            "algorithm": "Random",
            "mode": "random",
            "train_seed": s,
            "true_objective": float(idx),
            "teammate_in_frustum_rate": 0.25,
            "teammate_visible_rate": 0.15,
            "teammate_obscured_rate": 0.40,
            "teammate_out_of_view_rate": 0.85,
            "mutual_visibility_rate": 0.05,
            "target_visible_rate": 0.60,
            "target_obscured_rate": 0.40,
            "mean_teammate_pixel_fraction": 0.01,
        }
        for idx, s in enumerate((42, 68, 81, 97, 154))
    ]
    summaries = kd6j.aggregate_records_by_scenario_and_algo(records)
    assert len(summaries) == 1
    assert summaries[0]["num_seeds"] == 5
    assert summaries[0]["mean_true_objective"] == pytest.approx(2.0)
    assert summaries[0]["ci95_half_width"] > 0.0
