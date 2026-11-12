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
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DEFAULT_SEEDS = "42,68"
STUDENT_T_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
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


def state_entity_runs(main_runs: list[dict[str, Any]], entity_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Entity-state runs plus Pixel-Partial runs for the same cells: seeds 42/68 reused, seed 81 retrained."""
    cells = set(STATE_ENTITY_CELLS)
    out = [r for r in entity_runs if (r["scenario"], r["algorithm"]) in cells and r["cell_tag"] in ("state_entity", "pixel_partial")]
    out += [
        r for r in main_runs
        if (r["scenario"], r["algorithm"]) in cells and r["cell_tag"] == "pixel_partial"
        and int(r["train_seed"]) in STATE_ENTITY_PIXEL_REUSED_SEEDS
    ]
    return out


def ci95(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(STUDENT_T_95.get(len(arr), 1.96) * np.std(arr, ddof=1) / math.sqrt(len(arr)))


def _seed_mean(records: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in records if key in r and r[key] is not None and math.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else None


def summarize_cell(per_seed: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    seeds = sorted(per_seed)
    out: dict[str, Any] = {"seeds": seeds, "n_seeds": len(seeds)}
    out["episodes_per_seed"] = {str(s): len(per_seed[s]) for s in seeds}
    for key in ("true_objective",) + VISIBILITY_KEYS:
        seed_vals = [v for v in (_seed_mean(per_seed[s], key) for s in seeds) if v is not None]
        if not seed_vals:
            continue
        out[key] = {
            "mean": float(np.mean(seed_vals)),
            "ci95": ci95(seed_vals),
            "ci_method": f"student_t_df{len(seed_vals) - 1}" if len(seed_vals) > 1 else "none_single_seed",
            "t_crit": STUDENT_T_95.get(len(seed_vals)),
            "per_seed": seed_vals,
        }
    return out


def load_runs(runs_dir: Path, min_steps: int, seeds: set[int] | None = None) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(runs_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if seeds is not None and int(payload.get("train_seed", -1)) not in seeds:
            print(f"[skip] {path.name}: train_seed={payload.get('train_seed')} not in {sorted(seeds)}")
            continue
        if int(payload.get("checkpoint_steps", 0)) < min_steps:
            print(f"[skip] {path.name}: checkpoint_steps={payload.get('checkpoint_steps')} < {min_steps}")
            continue
        runs.append(payload)
    return runs


def aggregate(runs: list[dict[str, Any]], record_key: str = "normal_records") -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    for run in runs:
        records = run.get(record_key)
        if not records:
            continue
        grouped[(run["scenario"], run["algorithm"], run["cell_tag"])][int(run["train_seed"])] = records
    table: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for (scenario, algo, cell), per_seed in sorted(grouped.items()):
        table[scenario][algo][cell] = summarize_cell(per_seed)
    return json.loads(json.dumps(table))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="results/rebuttal_kd6j/mig_runs")
    p.add_argument("--out-dir", default="results/rebuttal_kd6j")
    p.add_argument("--min-steps", type=int, default=1_000_000, help="Reject smoke-test checkpoints below this many env steps.")
    p.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma-separated training seeds to include ('' = all).")
    p.add_argument("--clean-seeds", default=DEFAULT_CLEAN_SEEDS, help="Training seeds for summary_full_obs_clean.json.")
    p.add_argument("--state-entity-runs-dir", default="results/rebuttal_kd6j/heldout_runs_state_entity",
                   help="Held-out JSONs of the state_entity profile (entity state + Pixel-Partial seed 81).")
    p.add_argument("--v2-seeds", default=DEFAULT_V2_SEEDS, help="Training seeds for summary_full_obs_global_v2.json.")
    args = p.parse_args()

    seeds = {int(s) for s in args.seeds.split(",") if s.strip()} or None
    clean_seeds = {int(s) for s in args.clean_seeds.split(",") if s.strip()} or None
    all_runs = load_runs(Path(args.runs_dir), args.min_steps, None)
    runs = [r for r in all_runs if seeds is None or int(r["train_seed"]) in seeds]
    clean_runs = [
        r for r in all_runs
        if r["scenario"] in CLEAN_SCENARIOS and r["algorithm"] == "MAPPO" and r["cell_tag"] in CELLS_FULL_OBS_CLEAN
        and (clean_seeds is None or int(r["train_seed"]) in clean_seeds)
    ]
    v2_seeds = {int(s) for s in args.v2_seeds.split(",") if s.strip()} or None
    v2_runs = [
        r for r in all_runs
        if r["scenario"] in CLEAN_SCENARIOS and r["algorithm"] == "MAPPO" and r["cell_tag"] == CELL_FULL_OBS_GLOBAL_V2
        and int(r.get("checkpoint_steps", 0)) >= V2_MIN_STEPS
        and (v2_seeds is None or int(r["train_seed"]) in v2_seeds)
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
