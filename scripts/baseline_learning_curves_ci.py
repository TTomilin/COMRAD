"""Baseline learning curves with 95% CI bands across seeds, overlaid with the paper's held-out scores.

Curves are the measured training-layout ``policy_stats/avg_true_objective`` of every seed,
interpolated onto a shared 0-100M grid. The band is the mean +- Student-t 95% CI half-width
across seeds (the same statistic Table 3 reports). At 100M each algorithm also gets a marker
with error bar for its held-out zero-shot score, parsed from the raw baseline table of the paper (scripts/paper_tables.py).

A seed whose logs stop before 95% of the budget would bias the 100M statistics and cut every seed's curve short, so
the script refuses to run with such seeds; --allow-incomplete drops them (listed in the summary JSON) instead.

The curves are never rescaled or shifted toward the held-out values: training-layout and
held-out scores are different measurements, and any remaining gap is reported (printed and
written to the summary JSON) rather than hidden. To obtain curves whose endpoint *is* the
Table 3 value, evaluate intermediate checkpoints on the held-out batch and pass those logs.

Usage:
    python scripts/baseline_learning_curves_ci.py --logs-root <dir with per-seed run dirs>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

from comrad.utils.confidence import ci95_half_width
from baseline_results import (
    ALGORITHM_LABELS,
    ALGORITHM_ORDER,
    ALGORITHM_STYLES,
    BUDGET_STEPS,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    TEX_PATH,
    moving_average,
    parse_event_file,
    parse_heldout_table,
)

GRID_POINTS = 201
COMPLETE_FRACTION = 0.95
SMOOTH_WINDOW = 7
SEED_RE = re.compile(r"_see_(\d+)")
ALGO_RE = re.compile(r"^(?:\d+_)?(" + "|".join(sorted(ALGORITHM_ORDER, key=len, reverse=True)) + r")_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", required=True, help="Directory containing per-seed run directories")
    parser.add_argument("--tex", default=TEX_PATH)
    parser.add_argument("--figure-dir", default="paper/figure")
    parser.add_argument("--figure-name", default="baseline_learning_curves")
    parser.add_argument("--min-seeds", type=int, default=2, help="Minimum seeds needed to draw a CI band")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Drop seeds whose logs stop before 95%% of the budget instead of failing")
    return parser.parse_args()


def identify_run(run_dir_name: str) -> tuple[str, str, str] | None:
    algo_match = ALGO_RE.match(run_dir_name)
    seed_match = SEED_RE.search(run_dir_name)
    if not algo_match or not seed_match:
        return None
    scenarios = [s for s in SCENARIO_ORDER if f"env_{s}_" in run_dir_name or run_dir_name.endswith(f"env_{s}")]
    if not scenarios:
        return None
    scenario = max(scenarios, key=len)
    return scenario, algo_match.group(1), seed_match.group(1)


def load_seed_curves(logs_root: Path, grid: np.ndarray) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    curves: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    for event_path in sorted(logs_root.rglob("events.out.tfevents.*")):
        if "failed_" in str(event_path):
            continue
        ident = next((identify_run(p) for p in reversed(event_path.parts) if identify_run(p)), None)
        if ident is None:
            print(f"skip (unrecognised run name): {event_path}", flush=True)
            continue
        scenario, algo, seed = ident
        xs, ys = parse_event_file(event_path)
        order = np.argsort(xs, kind="stable")
        xs, ys = xs[order], ys[order]
        values = np.interp(grid, xs, moving_average(ys, SMOOTH_WINDOW), right=np.nan)
        if seed in curves[scenario][algo]:
            # Several event files for one run (restarts): keep the newest values where defined.
            prev = curves[scenario][algo][seed]
            values = np.where(np.isnan(values), prev, values)
        curves[scenario][algo][seed] = values
        print(f"parsed {scenario}/{algo}/seed{seed} ({len(xs)} points)", flush=True)
    return curves


def split_incomplete(curves, grid: np.ndarray):
    """Remove seeds whose (merged) curve ends before COMPLETE_FRACTION of the budget; returns (complete, incomplete)."""
    complete: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    incomplete: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for scenario, algos in curves.items():
        for algo, seeds in algos.items():
            for seed, values in seeds.items():
                last_step = float(grid[~np.isnan(values)][-1])
                if last_step < COMPLETE_FRACTION * BUDGET_STEPS:
                    incomplete[scenario][algo][seed] = last_step
                else:
                    complete[scenario][algo][seed] = values
    return complete, incomplete


def summarize(curves, heldout, min_seeds: int, incomplete=None) -> dict[str, dict[str, dict[str, object]]]:
    summary: dict[str, dict[str, dict[str, object]]] = {}
    for scenario in SCENARIO_ORDER:
        summary[scenario] = {}
        for algo in ALGORITHM_ORDER:
            seeds = curves.get(scenario, {}).get(algo, {})
            entry: dict[str, object] = {"seeds": sorted(seeds), "heldout_mean": heldout[scenario][algo][0],
                                        "heldout_ci95": heldout[scenario][algo][1]}
            dropped = (incomplete or {}).get(scenario, {}).get(algo, {})
            if dropped:
                entry["dropped_incomplete_seeds"] = {seed: step for seed, step in sorted(dropped.items())}
            if seeds:
                final = np.asarray([v[~np.isnan(v)][-1] for v in seeds.values()])
                entry["train_final_mean"] = float(final.mean())
                entry["train_final_ci95"] = ci95_half_width(final) if len(final) >= min_seeds else None
                entry["train_minus_heldout"] = float(final.mean() - heldout[scenario][algo][0])
            summary[scenario][algo] = entry
    return summary


def million_formatter(x: float, _pos: float) -> str:
    return "0" if x == 0 else f"{int(round(x / 1_000_000))}M"


def plot(curves, heldout, grid: np.ndarray, min_seeds: int, output_pdf: Path, output_png: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 4, figsize=(15.8, 10.8), dpi=300)
    axes_flat = list(axes.flat)
    offsets = np.linspace(-0.035, 0.035, len(ALGORITHM_ORDER)) * BUDGET_STEPS

    for idx, scenario in enumerate(SCENARIO_ORDER):
        ax = axes_flat[idx]
        for offset, algo in zip(offsets, ALGORITHM_ORDER, strict=True):
            style = ALGORITHM_STYLES[algo]
            seeds = curves.get(scenario, {}).get(algo, {})
            if seeds:
                stack = np.vstack(list(seeds.values()))
                valid = ~np.isnan(stack).any(axis=0)
                mean = stack[:, valid].mean(axis=0)
                ax.plot(grid[valid], mean, **style)
                if stack.shape[0] >= min_seeds:
                    half = ci95_half_width(stack[:, valid])
                    ax.fill_between(grid[valid], mean - half, mean + half, color=style["color"],
                                    alpha=0.15, linewidth=0.0)
            h_mean, h_ci = heldout[scenario][algo]
            ax.errorbar(BUDGET_STEPS + 0.06 * BUDGET_STEPS + offset, h_mean, yerr=h_ci, fmt="D",
                        ms=3.2, color=style["color"], ecolor=style["color"], elinewidth=1.0, capsize=1.5,
                        zorder=4, clip_on=False)
        if scenario not in curves:
            ax.text(0.5, 0.5, "no training logs", transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="#888888")

        ax.axvline(BUDGET_STEPS, color="#999999", linewidth=0.6, linestyle=":")
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=9.5, pad=5)
        ax.xaxis.set_major_locator(FixedLocator([0.0, 50_000_000.0, 100_000_000.0]))
        ax.xaxis.set_major_formatter(FuncFormatter(million_formatter))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="both", labelsize=7.5, length=2)
        ax.grid(True, linewidth=0.45, alpha=0.35)
        ax.set_axisbelow(True)
        ax.set_xlim(0.0, 1.12 * BUDGET_STEPS)
        if idx // 4 == 3:
            ax.set_xlabel("Frames", fontsize=8)
        if idx % 4 == 0:
            ax.set_ylabel("Avg. true objective", fontsize=8)

    handles = [Line2D([0], [0], label=ALGORITHM_LABELS[a], **ALGORITHM_STYLES[a]) for a in ALGORITHM_ORDER]
    handles.append(Line2D([0], [0], color="#555555", marker="D", linestyle="none", ms=4,
                          label="Held-out (Table 3)"))
    legend_ax = axes_flat[13]
    legend_ax.axis("off")
    legend = legend_ax.legend(handles=handles, loc="center", ncol=1, fontsize=8.5, frameon=True,
                              title="Algorithms", title_fontsize=9)
    legend.get_frame().set_linewidth(0.8)
    axes_flat[14].axis("off")
    axes_flat[15].axis("off")

    fig.suptitle("COMRAD baseline learning curves (mean $\\pm$ 95% CI over seeds)", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0.02, 0.03, 0.995, 0.975))
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logs_root = Path(args.logs_root)
    if not logs_root.exists():
        raise FileNotFoundError(f"{logs_root} does not exist")

    grid = np.linspace(0.0, BUDGET_STEPS, GRID_POINTS)
    heldout = parse_heldout_table(Path(args.tex))
    curves = load_seed_curves(logs_root, grid)
    if not curves:
        raise RuntimeError(f"No recognisable per-seed event files under {logs_root}")
    curves, incomplete = split_incomplete(curves, grid)
    stopped = [f"{s}/{a}/seed{seed} stops at {step / 1e6:.1f}M" for s, algos in incomplete.items()
               for a, seeds in algos.items() for seed, step in seeds.items()]
    for line in stopped:
        print(f"incomplete: {line}", flush=True)
    if stopped and not args.allow_incomplete:
        raise RuntimeError(f"{len(stopped)} seeds stop before {COMPLETE_FRACTION:.0%} of the budget; "
                           "pass --allow-incomplete to drop them")

    summary = summarize(curves, heldout, args.min_seeds, incomplete)
    print(f"\n{'Scenario':<24}{'Algo':<19}{'seeds':>5}{'train@100M':>18}{'held-out':>18}{'gap':>10}")
    for scenario, algos in summary.items():
        for algo, e in algos.items():
            if "train_final_mean" not in e:
                continue
            ci = e["train_final_ci95"]
            train = f"{e['train_final_mean']:.2f}" + (f" ± {ci:.2f}" if ci is not None else "")
            held = f"{e['heldout_mean']:.2f} ± {e['heldout_ci95']:.2f}"
            print(f"{SCENARIO_LABELS[scenario]:<24}{ALGORITHM_LABELS[algo]:<19}{len(e['seeds']):>5}"
                  f"{train:>18}{held:>18}{e['train_minus_heldout']:>+10.2f}")

    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot(curves, heldout, grid, args.min_seeds,
         figure_dir / f"{args.figure_name}.pdf", figure_dir / f"{args.figure_name}.png")
    json_path = figure_dir / f"{args.figure_name}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {figure_dir / args.figure_name}.pdf/.png and {json_path}")


if __name__ == "__main__":
    main()
