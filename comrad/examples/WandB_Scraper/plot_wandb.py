"""
Fetch W&B runs, cache results locally, and plot one subplot per environment.

Usage:
    python plot_wandb.py --entity <entity> --project <project> --metric <metric>
    python plot_wandb.py --entity myteam --project myproject --metric eval/mean_reward
    python plot_wandb.py ... --config-key env_id --output results.png --cache-dir .cache
    python plot_wandb.py ... --refresh   # ignore cache and re-fetch
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plot W&B metric per environment.")
    p.add_argument("--entity",     required=True,  help="W&B entity (user or team)")
    p.add_argument("--project",    required=True,  help="W&B project name")
    p.add_argument("--metric",     required=True,  help="Metric key to plot (e.g. eval/mean_reward)")
    p.add_argument("--config-key", default="env_id",
                   help="Run config field used to group runs by environment (default: env_id)")
    p.add_argument("--x-key",      default="_step",
                   help="History column to use as x-axis (default: _step)")
    p.add_argument("--output",     default="wandb_plot.png",
                   help="Output plot file (default: wandb_plot.png)")
    p.add_argument("--cache-dir",  default=".cache",
                   help="Directory to store cached JSON data (default: .cache)")
    p.add_argument("--refresh",    action="store_true",
                   help="Ignore existing cache and re-fetch from W&B")
    p.add_argument("--filters",    default=None,
                   help='Optional JSON filter dict for wandb.Api().runs(), e.g. \'{"state":"finished"}\'')
    p.add_argument("--run-ids",   default=None, nargs="+",
                   help="One or more run IDs to include (space-separated). Filters out all other runs.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_path(cache_dir: str, entity: str, project: str, metric: str, config_key: str) -> Path:
    safe = lambda s: s.replace("/", "_").replace("\\", "_")
    fname = f"{safe(entity)}__{safe(project)}__{safe(metric)}__{safe(config_key)}.json"
    return Path(cache_dir) / fname


def load_cache(path: Path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_cache(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"[cache] Saved to {path}")


# ---------------------------------------------------------------------------
# W&B fetch
# ---------------------------------------------------------------------------

def fetch_runs(entity: str, project: str, metric: str, config_key: str,
               x_key: str, filters: dict | None, run_ids: list | None = None) -> dict:
    """
    Returns a dict:
        { env_name: [ {"x": [...], "y": [...], "run_id": "..."}, ... ], ... }
    """
    api = wandb.Api(timeout=60)
    kwargs = {"filters": filters} if filters else {}
    runs = api.runs(f"{entity}/{project}", **kwargs)

    grouped: dict[str, list] = {}
    total = 0

    for run in runs:
        if run_ids and run.id not in run_ids:
            continue
        env = run.config.get(config_key)
        if env is None:
            print(f"[warn] Run {run.id} has no config key '{config_key}', skipping.")
            continue

        # Fetch history for the two columns we need (scan_history returns all steps)
        try:
            hist = list(run.scan_history(keys=[x_key, metric]))
        except Exception as e:
            print(f"[warn] Could not fetch history for run {run.id}: {e}")
            continue

        xs, ys = [], []
        for row in hist:
            x_val = row.get(x_key)
            y_val = row.get(metric)
            if x_val is not None and y_val is not None:
                xs.append(x_val)
                ys.append(y_val)

        if not xs:
            print(f"[warn] Run {run.id} ({env}): no data for metric '{metric}', skipping.")
            continue

        grouped.setdefault(str(env), []).append({"x": xs, "y": ys, "run_id": run.id})
        total += 1
        print(f"[fetch] {run.id} | {env} | {len(xs)} steps")

    print(f"[fetch] Done — {total} runs across {len(grouped)} environments.")
    return grouped


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def interpolate_to_common_grid(run_list: list, n_points: int = 300):
    """Interpolate all runs onto a shared x-grid; return (x_grid, mean, std)."""
    x_min = max(r["x"][0]  for r in run_list)
    x_max = min(r["x"][-1] for r in run_list)

    if x_min >= x_max:
        # Fallback: use union range
        x_min = min(r["x"][0]  for r in run_list)
        x_max = max(r["x"][-1] for r in run_list)

    x_grid = np.linspace(x_min, x_max, n_points)
    ys_interp = []
    for r in run_list:
        y_interp = np.interp(x_grid, r["x"], r["y"])
        ys_interp.append(y_interp)

    ys = np.array(ys_interp)
    return x_grid, ys.mean(axis=0), ys.std(axis=0)


def plot_results(grouped: dict, metric: str, x_key: str, output: str):
    envs = sorted(grouped.keys())
    n = len(envs)
    if n == 0:
        print("[error] No data to plot.")
        sys.exit(1)

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle(f"Metric: {metric}", fontsize=14, fontweight="bold")

    for idx, env in enumerate(envs):
        ax = axes[idx // cols][idx % cols]
        run_list = grouped[env]

        if len(run_list) == 1:
            r = run_list[0]
            ax.plot(r["x"], r["y"], linewidth=1.5)
        else:
            x_grid, mean, std = interpolate_to_common_grid(run_list)
            ax.plot(x_grid, mean, linewidth=1.5, label=f"mean (n={len(run_list)})")
            ax.fill_between(x_grid, mean - std, mean + std, alpha=0.25, label="±1 std")
            ax.legend(fontsize=7)

        ax.set_title(env, fontsize=10)
        ax.set_xlabel(x_key, fontsize=8)
        ax.set_ylabel(metric, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved to {output}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    filters = None
    if args.filters:
        try:
            filters = json.loads(args.filters)
        except json.JSONDecodeError as e:
            print(f"[error] --filters is not valid JSON: {e}")
            sys.exit(1)

    cpath = cache_path(args.cache_dir, args.entity, args.project, args.metric, args.config_key)

    grouped = None
    if not args.refresh:
        grouped = load_cache(cpath)
        if grouped is not None:
            print(f"[cache] Loaded from {cpath} ({sum(len(v) for v in grouped.values())} runs)")

    if grouped is None:
        grouped = fetch_runs(
            entity=args.entity,
            project=args.project,
            metric=args.metric,
            config_key=args.config_key,
            x_key=args.x_key,
            filters=filters,
            run_ids=args.run_ids,
        )
        save_cache(cpath, grouped)

    plot_results(grouped, metric=args.metric, x_key=args.x_key, output=args.output)


if __name__ == "__main__":
    main()
