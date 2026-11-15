"""
Aggregate multi-seed held-out evaluations of the very-difficult learning references (scripted-partner and oracle
variants) into results/very_difficult_reference_summary.json and LaTeX rows for tab:very_difficult_reference.

Each input is a summary.json written by scripts/heldout_wad_evaluation.py --train-seeds=42,68,81,97,154. The per-seed
score is the mean held-out true objective of that seed's frozen checkpoint (5 layouts x episodes); rows report
mean +- 95% CI half-width over training seeds with Student-t t_{0.975, n-1} (n=5 -> 2.776), matching Table 2.

Every (env, algorithm) row of the chosen --row-set must be present with all EXPECTED_SEEDS, and the inputs may not
contain policies outside it; --allow-incomplete writes the table anyway (missing rows are listed on stderr).

  uv run python -m experiments.very_difficult.very_difficult_reference_table --summary-roots \
      results/very_difficult_eval/heldout_very_difficult,results/very_difficult_eval/heldout_very_difficult_rerun
  uv run python -m experiments.very_difficult.very_difficult_reference_table --row-set scripted_partner \
      --summary-roots results/heldout_very_difficult \
      --output-json results/very_difficult_converse_summary.json --output-tex results/very_difficult_converse_rows.tex
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from comrad.utils.confidence import ci95_half_width

EXPECTED_SEEDS = (42, 68, 81, 97, 154)
LAVA_TASK = "Lava Maze"

# (row label, env, algorithm, task, ceiling). Lava Maze's benchmark score J is the number of mazes completed (goals
# reached, counted as goal relocations). It has no finite ceiling: the maze regenerates after every completion until the
# timeout. The raw ACS level counter (USER22) of the released WAD advances twice per completion and is kept in the JSON
# only for traceability.
ROWS = [
    ("Scripted scout $\\rightarrow$ learned gunner", "stealth_labyrinth_scripted_scout", "IPPO", "Stealth Labyrinth", 1.0),
    ("Scripted scout $\\rightarrow$ learned gunner", "stealth_labyrinth_scripted_scout", "IDQN", "Stealth Labyrinth", 1.0),
    ("Scripted gunner $\\rightarrow$ learned scout", "stealth_labyrinth_scripted_gunner", "IPPO", "Stealth Labyrinth", 1.0),
    ("Scripted gunner $\\rightarrow$ learned scout", "stealth_labyrinth_scripted_gunner", "IDQN", "Stealth Labyrinth", 1.0),
    ("Always-lit rooms", "stealth_labyrinth_lit", "MAPPO", "Stealth Labyrinth", 1.0),
    ("Always-lit rooms", "stealth_labyrinth_lit", "QMIX", "Stealth Labyrinth", 1.0),
    ("Shared vision", "stealth_labyrinth_shared_vision", "MAPPO", "Stealth Labyrinth", 1.0),
    ("Shared vision", "stealth_labyrinth_shared_vision", "QMIX", "Stealth Labyrinth", 1.0),
    ("Scripted observer $\\rightarrow$ learned navigator", "lava_maze_scripted_observer", "IPPO", "Lava Maze", None),
    ("Scripted observer $\\rightarrow$ learned navigator", "lava_maze_scripted_observer", "IDQN", "Lava Maze", None),
    ("Scripted navigator $\\rightarrow$ learned observer", "lava_maze_scripted_navigator", "IPPO", "Lava Maze", None),
    ("Scripted navigator $\\rightarrow$ learned observer", "lava_maze_scripted_navigator", "IDQN", "Lava Maze", None),
    ("Oracle message head", "lava_maze_oracle_msg", "MAPPO", "Lava Maze", None),
    ("Oracle message head", "lava_maze_oracle_msg", "QMIX", "Lava Maze", None),
    ("Oracle direction hint", "lava_maze_oracle_hint", "MAPPO", "Lava Maze", None),
    ("Oracle direction hint", "lava_maze_oracle_hint", "QMIX", "Lava Maze", None),
]
# Envs of each published table: the reference table (scripted followers + oracle variants) and the scripted-partner
# table (both roles scripted in turn).
ROW_SETS = {
    "reference": (
        "stealth_labyrinth_scripted_scout", "stealth_labyrinth_lit", "stealth_labyrinth_shared_vision",
        "lava_maze_scripted_observer", "lava_maze_oracle_msg", "lava_maze_oracle_hint",
    ),
    "scripted_partner": (
        "stealth_labyrinth_scripted_scout", "stealth_labyrinth_scripted_gunner",
        "lava_maze_scripted_observer", "lava_maze_scripted_navigator",
    ),
}


class IncompleteResultsError(ValueError):
    """The summaries do not cover exactly the rows and seeds of the requested table."""


def collect_seed_scores(summaries: list[dict[str, Any]]) -> dict[tuple[str, str], dict[int, dict[str, Any]]]:
    scores: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for summary in summaries:
        for policy in summary["policy_summaries"]:
            seed = policy.get("train_seed")
            if seed is None:
                raise ValueError(f"policy summary without train_seed: {policy.get('run_dir')}")
            if "/failed_" in str(policy.get("run_dir", "")) or Path(str(policy.get("run_dir", ""))).name.startswith("failed_"):
                continue
            key = (policy["scenario"], policy["algorithm"])
            if int(seed) in scores.setdefault(key, {}):
                raise ValueError(f"duplicate seed {seed} for {key}")
            scores[key][int(seed)] = {
                "mean_true_objective": float(policy["mean_true_objective"]),
                "mean_goals_reached": (
                    float(policy["mean_goals_reached"]) if policy.get("mean_goals_reached") is not None else None
                ),
                "checkpoint_steps": int(policy["checkpoint_steps"]),
                "episodes": int(policy["episodes"]),
                "run_dir": policy["run_dir"],
            }
    return scores


def aggregate(
    summaries: list[dict[str, Any]], row_set: str = "reference", allow_incomplete: bool = False
) -> list[dict[str, Any]]:
    seed_scores = collect_seed_scores(summaries)
    selected = [row for row in ROWS if row[1] in ROW_SETS[row_set]]
    problems = [f"unexpected policy {env}/{algo}" for env, algo in sorted(set(seed_scores) - {(r[1], r[2]) for r in selected})]
    rows = []
    for label, env, algorithm, task, ceiling in selected:
        per_seed = seed_scores.get((env, algorithm))
        if not per_seed:
            problems.append(f"{env}/{algorithm}: no summaries")
            continue
        seeds = tuple(sorted(per_seed))
        if seeds != tuple(sorted(EXPECTED_SEEDS)):
            problems.append(f"{env}/{algorithm}: expected seeds {EXPECTED_SEEDS}, got {seeds}")
            if not allow_incomplete:
                continue
        raw = [per_seed[s]["mean_true_objective"] for s in seeds]
        goals = [per_seed[s]["mean_goals_reached"] for s in seeds]
        if task == LAVA_TASK:
            if any(value is None for value in goals):
                raise ValueError(f"{env}/{algorithm}: Lava Maze rows need mean_goals_reached for every seed")
            values = goals
        else:
            values = raw
        row = {
            "label": label,
            "env": env,
            "algorithm": algorithm,
            "task": task,
            "ceiling": ceiling,
            "seeds": list(seeds),
            "seed_means": values,
            "mean": float(np.mean(values)),
            "ci95_half_width": ci95_half_width(values),
            "checkpoint_steps": sorted({per_seed[s]["checkpoint_steps"] for s in seeds}),
            "episodes_per_seed": sorted({per_seed[s]["episodes"] for s in seeds}),
        }
        if task == LAVA_TASK:
            row["metric"] = "mazes completed per episode (goal relocations)"
            row["seed_raw_counter_means"] = raw
            row["raw_counter_mean"] = float(np.mean(raw))
            row["raw_counter_ci95_half_width"] = ci95_half_width(raw)
        rows.append(row)
    for problem in problems:
        print(f"[incomplete] {problem}", file=sys.stderr)
    if problems and not allow_incomplete:
        raise IncompleteResultsError(f"{len(problems)} coverage problems for row set {row_set!r} (see stderr)")
    return rows


def render_rows(rows: list[dict[str, Any]]) -> str:
    lines = []
    task = None
    for row in rows:
        if row["task"] != task:
            task = row["task"]
            header = f" (ceiling ${row['ceiling']:.2f}$)" if row["ceiling"] else " (mazes completed; no fixed ceiling)"
            lines.append(f"    \\multicolumn{{3}}{{@{{}}l}}{{\\textit{{{task}}}{header}}} \\\\")
        lines.append(
            f"    {row['label']} & {row['algorithm']} & \\meanci{{{row['mean']:.2f}}}{{{row['ci95_half_width']:.2f}}} \\\\"
        )
    return "\n".join(lines) + "\n"


def discover_summaries(roots: list[Path], envs: tuple[str, ...]) -> list[Path]:
    """Per-seed summaries of `envs` under <root>/<env>/<algo>/seed_<s>/summary.json; later roots override earlier ones
    (reruns)."""
    chosen: dict[tuple[str, str, str], Path] = {}
    for root in roots:
        for path in sorted(root.glob("*/*/seed_*/summary.json")):
            env, algo, seed_dir = path.parent.parent.parent.name, path.parent.parent.name, path.parent.name
            if env in envs:
                chosen[(env, algo, seed_dir)] = path
    return [chosen[key] for key in sorted(chosen)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", default="", help="comma-separated summary.json paths")
    parser.add_argument(
        "--summary-roots",
        default="",
        help="comma-separated roots of <env>/<algo>/seed_<s>/summary.json; later roots override earlier ones",
    )
    parser.add_argument("--output-json", default="results/very_difficult_reference_summary.json")
    parser.add_argument("--output-tex", default="results/very_difficult_reference_rows.tex")
    parser.add_argument("--row-set", choices=sorted(ROW_SETS), default="reference")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    paths = [Path(p) for p in args.summaries.split(",") if p.strip()]
    paths += discover_summaries([Path(p) for p in args.summary_roots.split(",") if p.strip()], ROW_SETS[args.row_set])
    if not paths:
        parser.error("pass --summaries and/or --summary-roots")
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    rows = aggregate(summaries, row_set=args.row_set, allow_incomplete=args.allow_incomplete)
    payload = {
        "schema_version": 1,
        "inputs": [str(p) for p in paths],
        "ci": "Student-t 95% CI half-width over training seeds",
        "rows": rows,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    Path(args.output_tex).write_text(render_rows(rows), encoding="utf-8")
    print(args.output_json)
    print(args.output_tex)


if __name__ == "__main__":
    main()
