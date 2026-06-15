import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCENARIO_ORDER = [
    "stag_hunt_arena",
    "rhythm_sync_dense",
    "foraging_commons",
    "coop_puzzle",
    "platform_chain",
    "armory_siege",
    "coop_health_gathering",
    "lavapit",
    "smart_enemies",
    "dumb_enemies",
    "ammo_carrier",
    "stealth_labyrinth",
    "lava_maze",
]
SCENARIO_LABELS = {
    "ammo_carrier": "Ammo Carrier",
    "armory_siege": "Armory Siege",
    "coop_health_gathering": "Co-op Health\nGathering",
    "coop_puzzle": "Co-op Puzzle",
    "dumb_enemies": "Dumb Enemies",
    "foraging_commons": "Foraging Commons",
    "lava_maze": "Lava Maze",
    "lavapit": "Lava Pit",
    "platform_chain": "Platform Chain",
    "rhythm_sync_dense": "Rhythm Sync",
    "smart_enemies": "Smart Enemies",
    "stag_hunt_arena": "Stag Hunt Arena",
    "stealth_labyrinth": "Stealth Labyrinth",
}
ALGORITHMS = ["IPPO", "MAPPO", "HAPPO", "IDQN", "VDN", "QMIX", "QPLEX_dmaq", "QPLEX_dmaq_qatten"]
ALGORITHM_LABELS = {
    "IPPO": "IPPO", "MAPPO": "MAPPO", "HAPPO": "HAPPO",
    "IDQN": "IDQN", "VDN": "VDN", "QMIX": "QMIX",
    "QPLEX_dmaq": "QPLEX-D", "QPLEX_dmaq_qatten": "QPLEX-Q",
}
ON_POLICY = ["IPPO", "MAPPO", "HAPPO"]
OFF_POLICY = ["IDQN", "VDN", "QMIX", "QPLEX_dmaq", "QPLEX_dmaq_qatten"]

CEILINGS = {
    "ammo_carrier": 8400.0, "armory_siege": 100.0,
    "coop_health_gathering": 2100.0, "coop_puzzle": 5.0,
    "dumb_enemies": 90.0, "foraging_commons": 21000.0,
    "lava_maze": 6.0, "lavapit": 11.0,
    "platform_chain": 47.0, "rhythm_sync_dense": 1.0,
    "smart_enemies": 84.0, "stag_hunt_arena": 36.0,
    "stealth_labyrinth": 1.0,
}

ON_COLORS = {"IPPO": "#c44e52", "MAPPO": "#da7d80", "HAPPO": "#eba6a8"}
OFF_COLORS = {
    "IDQN": "#1f9ac0", "VDN": "#42b3d4", "QMIX": "#65c5e0",
    "QPLEX_dmaq": "#8ed4e8", "QPLEX_dmaq_qatten": "#b8e3f0",
}

Z_95 = 1.96


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", default="paper/COMRAD/figure/baseline_results_summary.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-name", default="all_algorithms_grouped_bars")
    return parser.parse_args()


def load_summary(path):
    return json.loads(path.read_text())


def normalized_score(summary, scenario, algo):
    mean = float(summary["scenarios"][scenario][algo]["synthetic_mean"])
    std = float(summary["scenarios"][scenario][algo]["synthetic_std"])
    ceiling = CEILINGS[scenario]
    return (float(np.clip(mean / ceiling, 0.0, 1.0)), float(std / ceiling))


def plot_all_algorithms_grouped_bars(summary, output_pdf, output_png):
    n_scenarios = len(SCENARIO_ORDER)
    n_algorithms = len(ALGORITHMS)
    n_cols = 5
    n_rows = int(np.ceil(n_scenarios / n_cols))

    bw = 0.65
    gap = 0.3
    positions = np.arange(n_algorithms, dtype=float) * (bw + gap)
    positions -= positions.mean()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 9, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6,
    })

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(16.0, 9.6), dpi=300, facecolor="white",
        sharey=False, gridspec_kw={"hspace": 0.55, "wspace": 0.30},
    )
    axes_flat = axes.flatten()

    color_map = {**ON_COLORS, **OFF_COLORS}
    edge_map = {
        **{a: "#8f3438" for a in ON_POLICY},
        **{a: "#166e89" for a in OFF_POLICY},
    }

    for idx, scenario in enumerate(SCENARIO_ORDER):
        ax = axes_flat[idx]
        ax.set_facecolor("#fbfbf8")

        means, errs, colors, edges, labels = [], [], [], [], []

        for algo in ALGORITHMS:
            mn, se = normalized_score(summary, scenario, algo)
            ci = Z_95 * se
            means.append(mn)
            errs.append(ci)
            colors.append(color_map[algo])
            edges.append(edge_map[algo])
            labels.append(ALGORITHM_LABELS[algo])

        ax.bar(positions, means, width=bw, color=colors, edgecolor=edges, linewidth=0.6, zorder=3)

        for x, mn, ci in zip(positions, means, errs):
            if ci > 0 and mn > 0:
                ax.errorbar(x, mn, yerr=ci, fmt="none", ecolor="#444444",
                            capsize=1.5, capthick=0.6, elinewidth=0.6, zorder=4)

        ax.set_ylim(0.0, 1.05)
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=12, fontweight="bold", pad=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d8d8d0", linewidth=0.6, alpha=0.85)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#444444")
        ax.spines["bottom"].set_color("#444444")

        if idx % n_cols == 0:
            ax.set_ylabel("Normalized score", fontsize=9)

    for idx in range(n_scenarios, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.06)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote grouped bars: {output_pdf}")
    print(f"Wrote grouped bars preview: {output_png}")


def main():
    args = parse_args()
    summary_path = Path(args.summary_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    output_pdf = output_dir / f"{args.output_name}.pdf"
    output_png = output_dir / f"{args.output_name}.png"
    plot_all_algorithms_grouped_bars(summary, output_pdf, output_png)


if __name__ == "__main__":
    main()
