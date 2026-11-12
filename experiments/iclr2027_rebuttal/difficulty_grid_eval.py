"""
3x3 difficulty grid (spatial x mechanical) evaluated zero-shot from one trained checkpoint (rebuttal C2K6).

Each cell keeps the scenario's default config and changes only the overrides of its two axis levels; the
(easy spatial, medium mechanical) cell is the default config, i.e. the training distribution. Every cell uses the same held-out
layout seeds, so the layout graph is the only thing besides the axis overrides that changes across seeds.

uv run python experiments/iclr2027_rebuttal/difficulty_grid_eval.py --scenario coop_puzzle \
    --run_dir train_dir/kd6j_snellius_copy/00_MAPPO_pixel_partial_env_coop_puzzle_s.r.alp_0.0_see_68
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import heldout_wad_evaluation as hw  # noqa: E402

LEVELS = ("easy", "medium", "hard")

# Co-op Puzzle: 10 zones (5 lane pairs) in every cell, so the ceiling stays 5 completed pairs; timeout is fixed (2000 tics).
#   spatial axis    : room length and lane height (distance to cover under the fixed timeout), plate size and
#                     number of candidate plates per lane (how much of the room must be searched for the lit plate).
#                     Easy is the training geometry: rooms smaller than in training are out of distribution for a
#                     trained policy and lower its score (probe: 256x160 rooms gave 10-36% vs 58% at default).
#   mechanical axis : how many of the 5 pairs are sync rooms where both agents must stand on one plate at once
#                     (and_frequency_pairs = 5 -> 0 sync rooms, 3 -> 1 (default), 1 -> 4)
GRIDS: dict[str, dict[str, Any]] = {
    "coop_puzzle": {
        "ceiling": 5.0,
        "spatial": {
            "easy": {},
            "medium": {"zone_length": 512, "lane_height": 320, "plate_size": 60, "plates_per_side": 3},
            "hard": {"zone_length": 640, "lane_height": 400, "plate_size": 48, "plates_per_side": 4},
        },
        "mechanical": {
            "easy": {"and_frequency_pairs": 5},
            "medium": {},
            "hard": {"and_frequency_pairs": 1},
        },
        "labels": {
            "spatial": {"easy": "training rooms, 2 plates", "medium": "512-unit rooms, 3 plates",
                        "hard": "640-unit rooms, 4 small plates"},
            "mechanical": {"easy": "0 sync rooms", "medium": "1 sync room (default)", "hard": "4 sync rooms"},
        },
    },
}


def cell_id(scenario: str, spatial: str, mechanical: str) -> str:
    return f"{scenario}_grid_s-{spatial}_m-{mechanical}"


def generate_cell(scenario: str, spatial: str, mechanical: str, seeds: list[int], batch_root: Path) -> list[hw.WadManifestEntry]:
    grid = GRIDS[scenario]
    module_name, class_name = hw.SCENARIO_CLASSES[scenario]
    scenario_cls = getattr(__import__(module_name, fromlist=[class_name]), class_name)
    overrides = {**grid["spatial"][spatial], **grid["mechanical"][mechanical]}
    out_dir = batch_root / cell_id(scenario, spatial, mechanical)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for seed in seeds:
        wad_id = f"{cell_id(scenario, spatial, mechanical)}_seed_{seed}"
        generated = scenario_cls({**overrides, "seed": int(seed)}, name=wad_id)
        generated.generate(str(out_dir / f"{wad_id}.wad"))
        entries.append(hw.WadManifestEntry(
            scenario=scenario, wad_id=wad_id, seed=int(seed), filename=f"{wad_id}.wad",
            sha256=hw.sha256_file(out_dir / f"{wad_id}.wad"), config=dict(generated.config),
        ))
    (out_dir / "batch_registry.json").write_text(
        json.dumps([hw.asdict(e) for e in entries], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entries


def make_policy(scenario: str, run_dir: Path, target_steps: int) -> hw.PolicySpec:
    checkpoint, steps = hw.select_checkpoint(run_dir, target_steps)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return hw.PolicySpec(
        scenario=scenario, algorithm=hw.archived_algo(run_dir), run_dir=str(run_dir.resolve()),
        checkpoint=str(checkpoint.resolve()), checkpoint_steps=steps,
        config_sha256=hw.sha256_file(run_dir / "config.json"), checkpoint_sha256=hw.sha256_file(checkpoint),
        train_seed=config.get("seed"),
    )


def eval_job(job: tuple) -> list[dict[str, Any]]:
    policy, batch_dir, entry, episodes, spatial, mechanical, deterministic = job
    hw.register_vizdoom_components()
    records = hw.evaluate_policy_on_wad(policy, Path(batch_dir), entry, episodes, "cpu", deterministic)
    for record in records:
        record.update({"spatial": spatial, "mechanical": mechanical})
    return records


def summarize(records: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    ceiling = GRIDS[scenario]["ceiling"]
    cells = {}
    for spatial in LEVELS:
        for mechanical in LEVELS:
            rows = [r for r in records if r["spatial"] == spatial and r["mechanical"] == mechanical]
            if not rows:
                continue
            scores = np.asarray([r["true_objective"] for r in rows], dtype=float)
            # CI over layouts: episodes on one layout are not independent draws of the layout distribution
            layout_means = np.asarray([np.mean([r["true_objective"] for r in rows if r["wad_id"] == w])
                                       for w in sorted({r["wad_id"] for r in rows})])
            ci = 1.96 * layout_means.std(ddof=1) / np.sqrt(len(layout_means)) if len(layout_means) > 1 else 0.0
            cells[f"{spatial}/{mechanical}"] = {
                "mean": float(scores.mean()), "ci95_layouts": float(ci), "episodes": int(len(scores)),
                "layouts": int(len(layout_means)), "mean_pct": float(100 * scores.mean() / ceiling),
                "ci95_pct": float(100 * ci / ceiling), "layout_means": layout_means.tolist(),
            }
    return cells


def render_markdown(cells: dict[str, Any], scenario: str, policy: hw.PolicySpec) -> str:
    labels = GRIDS[scenario]["labels"]
    lines = [
        f"### {scenario} 3x3 difficulty grid ({policy.algorithm}, train seed {policy.train_seed}, "
        f"checkpoint {policy.checkpoint_steps} frames); % of ceiling, mean +- 95% CI over layouts",
        "",
        "| Spatial \\ Mechanical | " + " | ".join(f"{m} ({labels['mechanical'][m]})" for m in LEVELS) + " |",
        "|" + "---|" * (len(LEVELS) + 1),
    ]
    for spatial in LEVELS:
        row = [f"{spatial} ({labels['spatial'][spatial]})"]
        for mechanical in LEVELS:
            c = cells.get(f"{spatial}/{mechanical}")
            row.append(f"{c['mean_pct']:.1f} +- {c['ci95_pct']:.1f}" if c else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="coop_puzzle", choices=sorted(GRIDS))
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--target_steps", type=int, default=hw.DEFAULT_TARGET_STEPS)
    parser.add_argument("--seeds", default=",".join(map(str, hw.HELDOUT_SEEDS)))
    parser.add_argument("--episodes_per_wad", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cells", default="", help="Comma list like easy/hard,hard/hard (default: all 9)")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--axis_json", default="",
                        help='Replace axis levels for probing, e.g. {"spatial": {"easy": {"plates_per_side": 1}}}')
    parser.add_argument("--batch_root", default="results/difficulty_grid/batches")
    parser.add_argument("--output_dir", default="results/difficulty_grid")
    args = parser.parse_args()

    if args.axis_json:
        for axis, levels in json.loads(args.axis_json).items():
            GRIDS[args.scenario][axis].update(levels)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    cells = [tuple(c.split("/")) for c in args.cells.split(",") if c] or [(s, m) for s in LEVELS for m in LEVELS]
    policy = make_policy(args.scenario, Path(args.run_dir), args.target_steps)
    batch_root = REPO_ROOT / args.batch_root

    jobs = []
    for spatial, mechanical in cells:
        for entry in generate_cell(args.scenario, spatial, mechanical, seeds, batch_root):
            jobs.append((policy, str(batch_root / cell_id(args.scenario, spatial, mechanical)), entry,
                         args.episodes_per_wad, spatial, mechanical, args.deterministic))

    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job_records in pool.map(eval_job, jobs):
            records.extend(job_records)
            r = job_records[0]
            print(f"{r['spatial']}/{r['mechanical']} {r['wad_id']}: "
                  f"{np.mean([x['true_objective'] for x in job_records]):.2f}", flush=True)

    summary = summarize(records, args.scenario)
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.scenario}_{policy.algorithm}_s{policy.train_seed}"
    (out_dir / f"{stem}_records.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    (out_dir / f"{stem}_summary.json").write_text(json.dumps({
        "policy": hw.asdict(policy), "grid": GRIDS[args.scenario], "seeds": seeds,
        "episodes_per_wad": args.episodes_per_wad, "deterministic": args.deterministic, "cells": summary,
    }, indent=2), encoding="utf-8")
    table = render_markdown(summary, args.scenario, policy)
    (out_dir / f"{stem}_table.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
