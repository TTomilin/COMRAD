"""
Aggregate multi-seed held-out evaluations of the very-difficult learning references (scripted-partner and oracle
variants) into results/very_difficult_reference_summary.json and LaTeX rows for tab:very_difficult_reference.

Each input is a summary.json written by scripts/heldout_wad_evaluation.py --train-seeds=42,68,81,97,154. The per-seed
score is the mean held-out true objective of that seed's frozen checkpoint (5 layouts x episodes); rows report
mean +- 95% CI half-width over training seeds with Student-t t_{0.975, n-1} (n=5 -> 2.776), matching Table 2.

  uv run python experiments/very_difficult/very_difficult_reference_table.py \
      --summaries results/heldout_very_difficult_scripted/summary.json,results/heldout_very_difficult_oracle/summary.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# Student-t 97.5% quantiles by degrees of freedom (n_seeds - 1)
T_975 = {1: 12.706204736174698, 2: 4.302652729749464, 3: 3.182446305284263, 4: 2.7764451051977987}
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


def ci_half_width(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    return T_975[n - 1] * float(np.std(values, ddof=1)) / math.sqrt(n)


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


def aggregate(summaries: list[dict[str, Any]], require_all_seeds: bool = True) -> list[dict[str, Any]]:
    seed_scores = collect_seed_scores(summaries)
    rows = []
    for label, env, algorithm, task, ceiling in ROWS:
        per_seed = seed_scores.get((env, algorithm))
        if not per_seed:
            continue
        seeds = tuple(sorted(per_seed))
        if require_all_seeds and seeds != tuple(sorted(EXPECTED_SEEDS)):
            raise ValueError(f"{env}/{algorithm}: expected seeds {EXPECTED_SEEDS}, got {seeds}")
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
            "ci95_half_width": ci_half_width(values),
            "checkpoint_steps": sorted({per_seed[s]["checkpoint_steps"] for s in seeds}),
            "episodes_per_seed": sorted({per_seed[s]["episodes"] for s in seeds}),
        }
        if task == LAVA_TASK:
            row["metric"] = "mazes completed per episode (goal relocations)"
            row["seed_raw_counter_means"] = raw
            row["raw_counter_mean"] = float(np.mean(raw))
            row["raw_counter_ci95_half_width"] = ci_half_width(raw)
        rows.append(row)
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


def discover_summaries(roots: list[Path]) -> list[Path]:
    """Per-seed summaries under <root>/<env>/<algo>/seed_<s>/summary.json; later roots override earlier ones (reruns)."""
    chosen: dict[tuple[str, str, str], Path] = {}
    for root in roots:
        for path in sorted(root.glob("*/*/seed_*/summary.json")):
            env, algo, seed_dir = path.parent.parent.parent.name, path.parent.parent.name, path.parent.name
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
    parser.add_argument("--allow-missing-seeds", action="store_true")
    args = parser.parse_args()

    paths = [Path(p) for p in args.summaries.split(",") if p.strip()]
    paths += discover_summaries([Path(p) for p in args.summary_roots.split(",") if p.strip()])
    if not paths:
        parser.error("pass --summaries and/or --summary-roots")
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    rows = aggregate(summaries, require_all_seeds=not args.allow_missing_seeds)
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
