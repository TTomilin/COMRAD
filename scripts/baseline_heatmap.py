from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


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
    "coop_health_gathering": "Co-op Health Gathering",
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
ALGORITHM_ORDER = [
    "IPPO",
    "MAPPO",
    "HAPPO",
    "IDQN",
    "VDN",
    "QMIX",
    "QPLEX_dmaq",
    "QPLEX_dmaq_qatten",
]
ALGORITHM_LABELS = {
    "IPPO": "IPPO",
    "MAPPO": "MAPPO",
    "HAPPO": "HAPPO",
    "IDQN": "IDQN",
    "VDN": "VDN",
    "QMIX": "QMIX",
    "QPLEX_dmaq": "QPLEX-D",
    "QPLEX_dmaq_qatten": "QPLEX-Q",
}
HEATMAP_NORMALIZATION_CEILINGS = {
    "ammo_carrier": 8400.0,
    "armory_siege": 100.0,
    "coop_health_gathering": 8400.0,
    "coop_puzzle": 5.0,
    "dumb_enemies": 90.0,
    "foraging_commons": 21000.0,
    "lava_maze": 6.0,
    "lavapit": 11.0,
    "platform_chain": 47.0,
    "rhythm_sync_dense": 1.0,
    "smart_enemies": 84.0,
    "stag_hunt_arena": 36.0,
    "stealth_labyrinth": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the COMRAD baseline heatmap from existing summary JSON.")
    parser.add_argument("--summary-json", default="results/baseline_results_summary.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-name", default="baseline_results_heatmap")
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def build_normalized_matrix(summary: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    scenario_payload = summary["scenarios"]
    matrix = np.zeros((len(SCENARIO_ORDER), len(ALGORITHM_ORDER)), dtype=float)
    for scenario_idx, scenario in enumerate(SCENARIO_ORDER):
        ceiling = HEATMAP_NORMALIZATION_CEILINGS[scenario]
        for algo_idx, algo in enumerate(ALGORITHM_ORDER):
            synthetic_mean = float(scenario_payload[scenario][algo]["synthetic_mean"])
            matrix[scenario_idx, algo_idx] = float(np.clip(synthetic_mean / ceiling, 0.0, 1.0))
    overall = np.mean(matrix, axis=0)
    return matrix, overall


def heatmap_text_color(value: float, cmap_name: str = "RdYlBu_r") -> str:
    rgba = plt.get_cmap(cmap_name)(value)
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < 0.5 else "#1a1a1a"


def style_heatmap_axis(ax: plt.Axes, n_rows: int, n_cols: int) -> None:
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def annotate_heatmap(ax: plt.Axes, values: np.ndarray, row_bests: np.ndarray, fontsize: float) -> None:
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = float(values[row_idx, col_idx])
            is_best = bool(row_bests[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=fontsize,
                color=heatmap_text_color(value),
                fontweight="bold" if is_best else "normal",
            )
            if is_best:
                ax.add_patch(
                    Rectangle(
                        (col_idx - 0.5, row_idx - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        linewidth=1.15,
                        edgecolor="#111111",
                    )
                )


def plot_heatmap(summary: dict[str, object], output_pdf: Path, output_png: Path) -> None:
    matrix, overall = build_normalized_matrix(summary)
    cmap_name = "RdYlBu_r"
    scenario_labels = [SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER]
    algorithm_labels = [ALGORITHM_LABELS[algo] for algo in ALGORITHM_ORDER]
    row_bests = np.isclose(matrix, matrix.max(axis=1, keepdims=True))
    overall_bests = np.isclose(overall, overall.max())[np.newaxis, :]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 10,
        }
    )

    fig = plt.figure(figsize=(11.4, 8.9), dpi=300, facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, len(SCENARIO_ORDER)], hspace=0.06)
    ax_top = fig.add_subplot(grid[0])
    ax_main = fig.add_subplot(grid[1], sharex=ax_top)

    ax_top.imshow(overall[np.newaxis, :], cmap=cmap_name, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    ax_main.imshow(matrix, cmap=cmap_name, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")

    ax_top.set_xticks(np.arange(len(ALGORITHM_ORDER)))
    ax_top.set_xticklabels(algorithm_labels, rotation=38, ha="left", rotation_mode="anchor", fontsize=9.5)
    ax_top.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, pad=8)
    ax_top.set_yticks([0])
    ax_top.set_yticklabels(["Overall score"], fontsize=10, fontweight="semibold")
    ax_top.tick_params(axis="y", labelleft=True, labelright=True, pad=10)
    style_heatmap_axis(ax_top, 1, len(ALGORITHM_ORDER))

    ax_main.set_xticks(np.arange(len(ALGORITHM_ORDER)))
    ax_main.tick_params(axis="x", top=False, bottom=False, labeltop=False, labelbottom=False)
    ax_main.set_yticks(np.arange(len(SCENARIO_ORDER)))
    ax_main.set_yticklabels(scenario_labels, fontsize=9)
    ax_main.tick_params(axis="y", labelleft=True, labelright=True, pad=10)
    style_heatmap_axis(ax_main, len(SCENARIO_ORDER), len(ALGORITHM_ORDER))

    annotate_heatmap(ax_top, overall[np.newaxis, :], overall_bests, fontsize=9.2)
    annotate_heatmap(ax_main, matrix, row_bests, fontsize=8.8)

    fig.subplots_adjust(left=0.16, right=0.90, top=0.88, bottom=0.10, hspace=0.07)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    output_pdf = output_dir / f"{args.output_name}.pdf"
    output_png = output_dir / f"{args.output_name}.png"
    plot_heatmap(summary, output_pdf, output_png)

    print(f"Wrote heatmap: {output_pdf}")
    print(f"Wrote heatmap preview: {output_png}")


if __name__ == "__main__":
    main()
