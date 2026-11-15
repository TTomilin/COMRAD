"""
Aggregate per-run Reviewer kd6J held-out JSONs (written by `experiments/iclr2027_rebuttal/eval_kd6j_checkpoint_pack.py`) into
`summary_2x2.json`, `summary_full_obs.json`, and `summary_interventions.json`.

Protocol: per training seed, average `true_objective` (and visibility rates) over its held-out episodes;
then report mean +/- 95% Student-t CI half-width across training seeds.
The rebuttal uses 2 training seeds {42, 68}, so the CI uses df=1 (t=12.706): CI = 12.706 * |s1 - s2| / 2.
The unconfounded MAPPO Full-vs-Partial follow-up (`pixel_partial` vs `pixel_full_clean`) uses 3 seeds {42, 68, 81}
(df=2, t=4.303) and is written to `summary_full_obs_clean.json`, together with the true full-observability cell
`pixel_full_global` (own RGB + global state vector).
The fixed global-state runs (`pixel_full_global_v2`: encoding v2, image-only input normalization, 64-layout training
pool, 100M frames, Table 3 seeds {42, 68, 81, 97, 154}) are written to `summary_full_obs_global_v2.json`; they are
compared with MAPPO (Partial) in Table 3 by `experiments/iclr2027_rebuttal/generate_kd6j_tables.py`.
The entity-state rerun (`--state_encoding=entity`, Table tab:po_state) is written to `summary_state_entity.json`:
`state_entity` runs and the retrained Pixel-Partial seed 81 come from `--state-entity-runs-dir`, Pixel-Partial seeds
42 and 68 from `--runs-dir`.

Every summary is checked against the cells and seeds its training profiles produce (see `expected_*`): a duplicate,
missing or unexpected (scenario, algorithm, cell, seed) aborts the aggregation unless `--allow-incomplete` is given.
Held-out JSONs from an older checkpoint-pack protocol (see kd6j_protocol.py) are never aggregated.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from comrad.utils.confidence import ci95_half_width, t_critical_95
from experiments.iclr2027_rebuttal.kd6j_protocol import KD6J_PACK_PROTOCOL_VERSION, is_current_result
from experiments.iclr2027_rebuttal.profiles import (
    BLIND_ABLATION_ALGORITHMS,
    FULL_OBSERVABILITY_SCENARIOS,
    REBUTTAL_KD6J_ALGORITHMS,
    VISIBILITY_TELEMETRY_SEEDS,
)

DEFAULT_SEEDS = "42,68"
VISIBILITY_KEYS = (
    "teammate_in_frustum_rate",
    "teammate_visible_rate",
    "teammate_obscured_rate",
    "teammate_out_of_view_rate",
    "mutual_visibility_rate",
    "target_visible_rate",
    "target_obscured_rate",
    "mean_teammate_pixel_fraction",
    "agent0_teammate_in_frustum_rate",
    "agent0_teammate_visible_rate",
    "agent0_teammate_obscured_rate",
    "agent1_teammate_in_frustum_rate",
    "agent1_teammate_visible_rate",
    "agent1_teammate_obscured_rate",
)
REPRESENTATIVE_2X2 = ("rhythm_sync_dense", "armory_siege", "ammo_carrier", "stealth_labyrinth")
CELLS_2X2 = ("pixel_partial", "pixel_full", "state_partial", "state_full", "blind")
CELLS_FULL_OBS = ("pixel_partial", "pixel_full")
# Unconfounded Full-vs-Partial (MAPPO; no AdditionalInput in either cell), 3 seeds.
CLEAN_SCENARIOS = ("rhythm_sync_dense", "armory_siege", "coop_puzzle")
CELLS_FULL_OBS_CLEAN = ("pixel_partial", "pixel_full_clean", "pixel_full_global")
DEFAULT_CLEAN_SEEDS = "42,68,81"
CELL_FULL_OBS_GLOBAL_V2 = "pixel_full_global_v2"
DEFAULT_V2_SEEDS = "42,68,81,97,154"
# v2 runs stop at ~100.02M frames; reject resumed runs evaluated before they finished.
V2_MIN_STEPS = 95_000_000
STATE_ENTITY_CELLS = (
    ("armory_siege", "MAPPO"), ("armory_siege", "QMIX"), ("ammo_carrier", "MAPPO"), ("ammo_carrier", "QMIX"),
    ("rhythm_sync_dense", "MAPPO"),
)
STATE_ENTITY_PIXEL_REUSED_SEEDS = {42, 68}
STATE_ENTITY_SEEDS = {42, 68, 81}

Cell = tuple[str, str, str]


class IncompleteResultsError(RuntimeError):
    """The held-out JSONs do not cover exactly the cells and seeds a summary reports."""


def state_entity_runs(main_runs: list[dict[str, Any]], entity_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Entity-state runs plus Pixel-Partial runs for the same cells: seeds 42/68 reused, seed 81 retrained."""
    cells = set(STATE_ENTITY_CELLS)
    out = [
        r for r in entity_runs
        if (r["scenario"], r["algorithm"]) in cells and r["cell_tag"] in ("state_entity", "pixel_partial")
        and int(r["train_seed"]) in STATE_ENTITY_SEEDS
        and (r["cell_tag"] == "state_entity" or int(r["train_seed"]) not in STATE_ENTITY_PIXEL_REUSED_SEEDS)
    ]
    out += [
        r for r in main_runs
        if (r["scenario"], r["algorithm"]) in cells and r["cell_tag"] == "pixel_partial"
        and int(r["train_seed"]) in STATE_ENTITY_PIXEL_REUSED_SEEDS
    ]
    return out


def expected_2x2(seeds: set[int]) -> dict[Cell, set[int]]:
    cells = {(s, a, c): set(seeds) for s in REPRESENTATIVE_2X2 for a in REBUTTAL_KD6J_ALGORITHMS for c in CELLS_2X2 if c != "blind"}
    cells.update({(s, a, "blind"): set(seeds) for s in REPRESENTATIVE_2X2 for a in BLIND_ABLATION_ALGORITHMS})
    return cells


def expected_full_obs(seeds: set[int]) -> dict[Cell, set[int]]:
    """2x2 pixel cells plus the single-seed MAPPO Pixel-Partial telemetry runs of every other scenario."""
    cells = {cell: s for cell, s in expected_2x2(seeds).items() if cell[2] in CELLS_FULL_OBS}
    telemetry_seeds = set(VISIBILITY_TELEMETRY_SEEDS) & set(seeds)
    cells.update({(scenario.env, "MAPPO", "pixel_partial"): telemetry_seeds for scenario in FULL_OBSERVABILITY_SCENARIOS})
    return cells


def expected_interventions(seeds: set[int]) -> dict[Cell, set[int]]:
    return {cell: s for cell, s in expected_full_obs(seeds).items() if cell[2] == "pixel_partial"}


def expected_clean(seeds: set[int]) -> dict[Cell, set[int]]:
    return {(s, "MAPPO", c): set(seeds) for s in CLEAN_SCENARIOS for c in CELLS_FULL_OBS_CLEAN}


def expected_v2(seeds: set[int]) -> dict[Cell, set[int]]:
    return {(s, "MAPPO", CELL_FULL_OBS_GLOBAL_V2): set(seeds) for s in CLEAN_SCENARIOS}


def expected_state_entity() -> dict[Cell, set[int]]:
    return {(s, a, c): set(STATE_ENTITY_SEEDS) for s, a in STATE_ENTITY_CELLS for c in ("state_entity", "pixel_partial")}


def coverage_problems(name: str, table: dict[str, Any], expected: dict[Cell, set[int]]) -> list[str]:
    """Cells whose seeds lack a required seed, reported cells no profile produces, and metrics measured on fewer seeds
    than their cell."""
    problems = []
    for (scenario, algo, cell), required in sorted(expected.items()):
        summary = table.get(scenario, {}).get(algo, {}).get(cell)
        missing = sorted(set(required) - set(summary["seeds"] if summary else ()))
        if missing:
            problems.append(f"{name}: {scenario}/{algo}/{cell} is missing train seeds {missing}")
    for scenario, algos in sorted(table.items()):
        for algo, cells in sorted(algos.items()):
            for cell, summary in sorted(cells.items()):
                if summary is not None and (scenario, algo, cell) not in expected:
                    problems.append(f"{name}: unexpected cell {scenario}/{algo}/{cell}")
                for key in ("true_objective",) + VISIBILITY_KEYS:
                    metric = (summary or {}).get(key)
                    if metric is not None and metric["seeds"] != summary["seeds"]:
                        problems.append(
                            f"{name}: {scenario}/{algo}/{cell} {key} covers seeds {metric['seeds']} of {summary['seeds']}"
                        )
    return problems


def _seed_mean(records: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in records if key in r and r[key] is not None and math.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else None


def summarize_cell(per_seed: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    seeds = sorted(per_seed)
    out: dict[str, Any] = {"seeds": seeds, "n_seeds": len(seeds)}
    out["episodes_per_seed"] = {str(s): len(per_seed[s]) for s in seeds}
    for key in ("true_objective",) + VISIBILITY_KEYS:
        measured = {s: v for s, v in ((s, _seed_mean(per_seed[s], key)) for s in seeds) if v is not None}
        if not measured:
            continue
        seed_vals = list(measured.values())
        out[key] = {
            "seeds": sorted(measured),
            "mean": float(np.mean(seed_vals)),
            "ci95": ci95_half_width(seed_vals),
            "ci_method": f"student_t_df{len(seed_vals) - 1}" if len(seed_vals) > 1 else "none_single_seed",
            "t_crit": t_critical_95(len(seed_vals)) if len(seed_vals) > 1 else None,
            "per_seed": seed_vals,
        }
    return out


def load_runs(runs_dir: Path, min_steps: int, seeds: set[int] | None = None) -> list[dict[str, Any]]:
    runs = []
    stale = []
    for path in sorted(runs_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not is_current_result(payload):
            stale.append(f"{path.name} (protocol_version={payload.get('protocol_version')!r})")
            continue
        if seeds is not None and int(payload.get("train_seed", -1)) not in seeds:
            print(f"[skip] {path.name}: train_seed={payload.get('train_seed')} not in {sorted(seeds)}")
            continue
        if int(payload.get("checkpoint_steps", 0)) < min_steps:
            print(f"[skip] {path.name}: checkpoint_steps={payload.get('checkpoint_steps')} < {min_steps}")
            continue
        runs.append(payload)
    if stale:
        raise IncompleteResultsError(
            f"{len(stale)} held-out JSONs in {runs_dir} are not protocol {KD6J_PACK_PROTOCOL_VERSION!r}; re-evaluate "
            f"them with eval_kd6j_checkpoint_pack.py: {', '.join(stale[:5])}{' ...' if len(stale) > 5 else ''}"
        )
    return runs


def aggregate(runs: list[dict[str, Any]], record_key: str = "normal_records") -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    for run in runs:
        records = run.get(record_key)
        if not records:
            continue
        per_seed = grouped[(run["scenario"], run["algorithm"], run["cell_tag"])]
        seed = int(run["train_seed"])
        if seed in per_seed:
            raise IncompleteResultsError(
                f"duplicate {record_key} for {run['scenario']}/{run['algorithm']}/{run['cell_tag']} seed {seed}"
            )
        per_seed[seed] = records
    table: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for (scenario, algo, cell), per_seed in sorted(grouped.items()):
        table[scenario][algo][cell] = summarize_cell(per_seed)
    return json.loads(json.dumps(table))


def parse_seeds(text: str, flag: str) -> set[int]:
    seeds = {int(s) for s in text.split(",") if s.strip()}
    if not seeds:
        raise ValueError(f"{flag} must list at least one training seed")
    return seeds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="results/rebuttal_kd6j/mig_runs")
    p.add_argument("--out-dir", default="results/rebuttal_kd6j")
    p.add_argument("--min-steps", type=int, default=1_000_000, help="Reject smoke-test checkpoints below this many env steps.")
    p.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma-separated training seeds to include.")
    p.add_argument("--clean-seeds", default=DEFAULT_CLEAN_SEEDS, help="Training seeds for summary_full_obs_clean.json.")
    p.add_argument("--state-entity-runs-dir", default="results/rebuttal_kd6j/heldout_runs_state_entity",
                   help="Held-out JSONs of the state_entity profile (entity state + Pixel-Partial seed 81).")
    p.add_argument("--v2-seeds", default=DEFAULT_V2_SEEDS, help="Training seeds for summary_full_obs_global_v2.json.")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="Write the summaries even if cells or seeds are missing, duplicated or unexpected (prints them).")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds, "--seeds")
    clean_seeds = parse_seeds(args.clean_seeds, "--clean-seeds")
    v2_seeds = parse_seeds(args.v2_seeds, "--v2-seeds")
    all_runs = load_runs(Path(args.runs_dir), args.min_steps, None)
    runs = [r for r in all_runs if int(r["train_seed"]) in seeds]
    clean_runs = [
        r for r in all_runs
        if r["scenario"] in CLEAN_SCENARIOS and r["algorithm"] == "MAPPO" and r["cell_tag"] in CELLS_FULL_OBS_CLEAN
        and int(r["train_seed"]) in clean_seeds
    ]
    v2_runs = [
        r for r in all_runs
        if r["scenario"] in CLEAN_SCENARIOS and r["algorithm"] == "MAPPO" and r["cell_tag"] == CELL_FULL_OBS_GLOBAL_V2
        and int(r.get("checkpoint_steps", 0)) >= V2_MIN_STEPS
        and int(r["train_seed"]) in v2_seeds
    ]
    normal = aggregate(runs, "normal_records")
    summary_clean = aggregate(clean_runs, "normal_records")
    summary_v2 = aggregate(v2_runs, "normal_records")
    entity_dir = Path(args.state_entity_runs_dir)
    entity_runs = load_runs(entity_dir, args.min_steps, None) if entity_dir.is_dir() else []
    summary_state_entity = aggregate(state_entity_runs(all_runs, entity_runs), "normal_records")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_2x2 = {
        s: {a: {c: v for c, v in cells.items() if c in CELLS_2X2} for a, cells in algos.items()}
        for s, algos in normal.items()
        if s in REPRESENTATIVE_2X2
    }
    summary_full = {
        s: {a: {c: v for c, v in cells.items() if c in CELLS_FULL_OBS} for a, cells in algos.items()}
        for s, algos in normal.items()
    }
    interventions = {
        "normal": {s: {a: cells.get("pixel_partial") for a, cells in algos.items()} for s, algos in normal.items()},
        "partner_occlusion": aggregate(runs, "partner_occlusion_records"),
        "blind_eval": aggregate(runs, "blind_eval_records"),
    }
    problems = (
        coverage_problems("summary_2x2", summary_2x2, expected_2x2(seeds))
        + coverage_problems("summary_full_obs", summary_full, expected_full_obs(seeds))
        + coverage_problems("partner_occlusion", interventions["partner_occlusion"], expected_interventions(seeds))
        + coverage_problems("blind_eval", interventions["blind_eval"], expected_interventions(seeds))
        + coverage_problems("summary_full_obs_clean", summary_clean, expected_clean(clean_seeds))
        + coverage_problems("summary_state_entity", summary_state_entity, expected_state_entity())
        + coverage_problems("summary_full_obs_global_v2", summary_v2, expected_v2(v2_seeds))
    )
    for problem in problems:
        print(f"[incomplete] {problem}")
    if problems and not args.allow_incomplete:
        raise IncompleteResultsError(f"{len(problems)} coverage problems (see above); pass --allow-incomplete to write anyway")
    for name, payload in (
        ("summary_2x2.json", summary_2x2),
        ("summary_full_obs.json", summary_full),
        ("summary_interventions.json", interventions),
        ("summary_full_obs_clean.json", summary_clean),
        ("summary_state_entity.json", summary_state_entity),
        ("summary_full_obs_global_v2.json", summary_v2),
    ):
        (out_dir / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_dir / name}")


if __name__ == "__main__":
    main()
