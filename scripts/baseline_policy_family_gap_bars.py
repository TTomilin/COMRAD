from __future__ import annotations

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
ON_POLICY = ["IPPO", "MAPPO", "HAPPO"]
OFF_POLICY = ["IDQN", "VDN", "QMIX", "QPLEX_dmaq", "QPLEX_dmaq_qatten"]
HEATMAP_NORMALIZATION_CEILINGS = {
    "ammo_carrier": 8400.0,
    "armory_siege": 100.0,
    "coop_health_gathering": 2100.0,
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
    parser = argparse.ArgumentParser(
        description="Render grouped bars comparing the best on-policy and off-policy baseline per scenario."
    )
    parser.add_argument("--summary-json", default="results/baseline_results_summary.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-name", default="baseline_policy_family_gap")
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def normalized_score(summary: dict[str, object], scenario: str, algo: str) -> float:
    mean = float(summary["scenarios"][scenario][algo]["synthetic_mean"])
    return float(np.clip(mean / HEATMAP_NORMALIZATION_CEILINGS[scenario], 0.0, 1.0))


def collect_family_bests(
    summary: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    on_scores: list[float] = []
    off_scores: list[float] = []
    on_winners: list[str] = []
    off_winners: list[str] = []
    gaps: list[float] = []

    for scenario in SCENARIO_ORDER:
        best_on_algo = max(ON_POLICY, key=lambda algo: normalized_score(summary, scenario, algo))
        best_off_algo = max(OFF_POLICY, key=lambda algo: normalized_score(summary, scenario, algo))
        best_on = normalized_score(summary, scenario, best_on_algo)
        best_off = normalized_score(summary, scenario, best_off_algo)
        on_scores.append(best_on)
        off_scores.append(best_off)
        on_winners.append(best_on_algo)
        off_winners.append(best_off_algo)
        gaps.append(best_on - best_off)

    return (
        np.asarray(on_scores, dtype=float),
        np.asarray(off_scores, dtype=float),
        on_winners,
        off_winners,
        np.asarray(gaps, dtype=float),
    )


def plot_grouped_bars(summary: dict[str, object], output_pdf: Path, output_png: Path) -> None:
    on_scores, off_scores, on_winners, off_winners, gaps = collect_family_bests(summary)
    scenarios = [SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER]

    x = np.arange(len(scenarios), dtype=float)
    width = 0.34
    on_color = "#c44e52"
    off_color = "#1f9ac0"
    connector_color = "#4a4a4a"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(13.6, 6.8), dpi=300, facecolor="white")
    ax.set_facecolor("#fbfbf8")

    ax.bar(
        x - width / 2,
        on_scores,
        width=width,
        color=on_color,
        edgecolor="#8f3438",
        linewidth=0.8,
        label="Best on-policy",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        off_scores,
        width=width,
        color=off_color,
        edgecolor="#166e89",
        linewidth=0.8,
        label="Best off-policy",
        zorder=3,
    )

    for idx, (on_val, off_val, gap, on_algo, off_algo) in enumerate(
        zip(on_scores, off_scores, gaps, on_winners, off_winners, strict=True)
    ):
        ax.plot([x[idx], x[idx]], [min(on_val, off_val), max(on_val, off_val)], color=connector_color, linewidth=0.8, zorder=4)
        inside_threshold = 0.18
        outside_offset = 0.012
        inside_offset = 0.02
        label_height_pad = 0.12
        gap_clearance = 0.025

        if on_val >= inside_threshold:
            on_label_y = on_val - inside_offset
            on_label_va = "top"
            on_label_color = "white"
            on_label_top = on_val
        else:
            on_label_y = on_val + outside_offset
            on_label_va = "bottom"
            on_label_color = "#5a2024"
            on_label_top = on_label_y + label_height_pad

        if off_val >= inside_threshold:
            off_label_y = off_val - inside_offset
            off_label_va = "top"
            off_label_color = "white"
            off_label_top = off_val
        else:
            off_label_y = off_val + outside_offset
            off_label_va = "bottom"
            off_label_color = "#0d5064"
            off_label_top = off_label_y + label_height_pad

        ax.text(
            x[idx] - width / 2,
            on_label_y,
            ALGORITHM_LABELS[on_algo],
            ha="center",
            va=on_label_va,
            rotation=90,
            fontsize=7.1,
            color=on_label_color,
            fontweight="bold",
            zorder=5,
        )
        ax.text(
            x[idx] + width / 2,
            off_label_y,
            ALGORITHM_LABELS[off_algo],
            ha="center",
            va=off_label_va,
            rotation=90,
            fontsize=7.1,
            color=off_label_color,
            fontweight="bold",
            zorder=5,
        )

        top_val = max(on_val, off_val, on_label_top, off_label_top)
        gap_y = max(top_val + gap_clearance, 0.12)
        ax.text(
            x[idx],
            gap_y,
            f"{gap:+.2f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            color=on_color if gap >= 0 else off_color,
            fontweight="semibold",
        )

    ax.set_ylim(0.0, 1.16)
    ax.set_xlim(-0.8, len(scenarios) - 0.2)
    ax.set_ylabel("Best family score (scenario-normalized)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=32, ha="right")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d8d8d0", linewidth=0.8, alpha=0.85)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    mean_gap = float(np.mean(gaps))
    on_wins = int(np.sum(gaps > 0))
    off_wins = int(np.sum(gaps < 0))
    ties = int(np.sum(np.isclose(gaps, 0.0)))

    legend = ax.legend(loc="upper right", ncol=2, frameon=True, fontsize=9)
    legend.get_frame().set_edgecolor("#cccccc")
    legend.get_frame().set_linewidth(0.8)

    print(f"Mean gap = {mean_gap:+.2f}, on-policy {on_wins}, off-policy {off_wins}, ties {ties}.")

    fig.subplots_adjust(left=0.08, right=0.99, top=0.87, bottom=0.29)
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
    plot_grouped_bars(summary, output_pdf, output_png)

    print(f"Wrote grouped bars: {output_pdf}")
    print(f"Wrote grouped bars preview: {output_png}")


if __name__ == "__main__":
    main()
