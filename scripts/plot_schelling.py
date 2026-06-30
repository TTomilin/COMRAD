import argparse
import csv
import os
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCENARIO_LABELS = {
    "stag_hunt_arena": "Stag Hunt Arena",
    "rhythm_sync_dense": "Rhythm Sync",
    "foraging_commons": "Foraging Commons",
    "coop_puzzle": "Co-op Puzzle",
    "platform_chain": "Platform Chain",
    "armory_siege": "Armory Siege",
    "lavapit": "Lavapit",
    "coop_health_gathering": "Co-op Health Gathering",
    "smart_enemies": "Smart Enemies",
    "dumb_enemies": "Dumb Enemies",
    "ammo_carrier": "Ammo Carrier",
    "stealth_labyrinth": "Stealth Labyrinth",
    "lava_maze": "Lava Maze",
}

ALGO_COLORS = {
    "IPPO":  "#1f77b4",
    "MAPPO": "#d62728",
    "HAPPO": "#2ca02c",
}

ALGO_MARKERS = {
    "IPPO":  "s",
    "MAPPO": "D",
    "HAPPO": "o",
}

ALGO_ORDER = ["IPPO", "MAPPO", "HAPPO"]

YLIM_MAP: dict[str, tuple[float, float]] = {
    "stag_hunt_arena": (-0.5, 8),
    "coop_puzzle": (-1, 8),
    "lava_maze": (-1, 10),
    "lavapit": (-1, 6),
    "platform_chain": (-2, 52),
    "rhythm_sync_dense": (-0.05, 1.15),
    "stealth_labyrinth": (-0.05, 1.15),
}


def load_landscape(csv_path):
    proportions, c_mean, c_std, d_mean, d_std, avg = [], [], [], [], [], []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            proportions.append(float(row["proportion"]))
            c_mean.append(float(row["C_payoff_mean"]))
            c_std.append(float(row["C_payoff_std"]))
            d_mean.append(float(row["D_payoff_mean"]))
            d_std.append(float(row["D_payoff_std"]))
            avg.append(float(row["avg_payoff"]))
    return {
        "proportion": np.array(proportions),
        "C_mean": np.array(c_mean), "C_std": np.array(c_std),
        "D_mean": np.array(d_mean), "D_std": np.array(d_std),
        "avg": np.array(avg),
    }


def load_algo_markers(csv_path):
    markers = {}
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            scenario = row["scenario"]
            algo = row["algo"]
            markers.setdefault(scenario, {})[algo] = {
                "x": float(row["coop_rate_mean"]),
                "x_err": float(row["coop_rate_std"]),
                "y": float(row["true_obj_mean"]),
                "y_err": float(row["true_obj_std"]),
            }
    return markers


def _auto_xlim(markers_for_scenario):
    return (-0.05, 1.05)


def plot_schelling_figure(landscape_dir, markers_csv, output_path):
    markers = None
    if markers_csv and os.path.exists(markers_csv):
        markers = load_algo_markers(markers_csv)

    scenarios = [k for k in SCENARIO_LABELS if os.path.exists(os.path.join(landscape_dir, f"{k}_payoff_landscape.csv"))]
    n = len(scenarios)
    if n == 0:
        raise SystemExit("No landscape CSVs found.")

    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows),
                             squeeze=False)

    handles_labels = None
    for ax_cell, scenario in zip(axes.flat, scenarios):
        ax = ax_cell
        csv_path = os.path.join(landscape_dir, f"{scenario}_payoff_landscape.csv")
        data = load_landscape(csv_path)
        p = data["proportion"]
        ax.plot(p, data["C_mean"], "k-", linewidth=1.8, label="Cooperator")
        ax.plot(p, data["D_mean"], "k--", linewidth=1.8, label="Defector")
        ax.plot(p, data["avg"], "k:", linewidth=1.2, label="Average")

        sc_markers = (markers or {}).get(scenario, {})
        if sc_markers:
            y_lo, y_hi = ax.get_ylim()
            y_jitter_step = (y_hi - y_lo) * 0.015
            x_jitter_step = 0.012
            for i, algo in enumerate(ALGO_ORDER):
                m = sc_markers.get(algo)
                if m is None:
                    continue
                x_pos = m["x"] + (i - 1) * x_jitter_step
                y_pos = m["y"] + (i - 1) * y_jitter_step
                ax.plot([x_pos, x_pos], [y_lo, y_pos], color=ALGO_COLORS[algo], linestyle=":", alpha=0.35, zorder=3)
                ax.errorbar(x_pos, y_pos, yerr=m["y_err"],
                            fmt=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                            markersize=6, capsize=3, capthick=1,
                            label=algo, zorder=5, markeredgewidth=0.8,
                            markeredgecolor="white", alpha=0.9)

        ax.set_title(SCENARIO_LABELS.get(scenario, scenario), fontsize=16, fontweight="bold", pad=3)
        ax.set_xlabel("Proportion of cooperators", fontsize=13)
        ax.set_ylabel("True objective", fontsize=13)
        ax.set_xlim(*_auto_xlim(sc_markers))
        if scenario in YLIM_MAP:
            ax.set_ylim(*YLIM_MAP[scenario])
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    # hide unused grid cells
    for idx in range(n, nrows * ncols):
        axes.flat[idx].set_visible(False)

    if handles_labels is not None:
        handles, labels = handles_labels
        gs = axes[0, 0].get_gridspec()
        leg_ax = fig.add_subplot(gs[nrows - 1, 1:4], facecolor="none")
        leg_ax.set_xticks([])
        leg_ax.set_yticks([])
        for spine in leg_ax.spines.values():
            spine.set_visible(False)
        legend_lines = []
        for h, l in zip(handles, labels):
            if l in ALGO_COLORS:
                legend_lines.append(plt.Line2D([0], [0], marker=ALGO_MARKERS[l],
                    color=ALGO_COLORS[l], markersize=8, linestyle="",
                    markeredgewidth=1.0, markeredgecolor="white"))
            else:
                legend_lines.append(h)
        leg_ax.legend(legend_lines, labels, loc="center", ncol=2,
                      fontsize=18, frameon=True, facecolor="white",
                      edgecolor="#cccccc", columnspacing=1.5,
                      handlelength=0.8, handleheight=0.8, borderpad=1.0)

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    base, _ = os.path.splitext(output_path)
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {output_path}  ({n} scenarios, {nrows}×{ncols})")
    plt.close(fig)


parser = argparse.ArgumentParser()
parser.add_argument("--landscape-dir", default="results/schelling")
parser.add_argument("--markers-csv", default="results/schelling/algo_markers.csv")
parser.add_argument("--output", default="results/schelling_figures/schelling_social_dilemmas.pdf")
args = parser.parse_args()
plot_schelling_figure(args.landscape_dir, args.markers_csv, args.output)
