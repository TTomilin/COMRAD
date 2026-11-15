"""Regenerate Figure 4 (baseline_policy_family_gap) directly from the raw baseline table of the paper.

The table rows are parsed from the LaTeX source so the figure cannot drift from the
reported numbers. Normalization follows the caption: s~ = clip(s / s_max, 0, 1), with
s_max = 4 completed mazes for Lava Maze (no finite ceiling).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from paper_tables import ALGORITHM_COLUMNS, TEX_PATH, BaselineRow, parse_baseline_results

FIGURE_DIR = "paper/figure"
FIGURE_NAME = "baseline_policy_family_gap"
ON_POLICY = ["IPPO", "MAPPO", "HAPPO"]
OFF_POLICY = ["IDQN", "VDN", "QMIX", "QPLEX-D", "QPLEX-Q"]
ON_COLOR = "#c44e52"
OFF_COLOR = "#1f9ac0"
ON_TEXT = "#5a1a1d"
OFF_TEXT = "#0b4a5e"
INSIDE_LABEL_MIN = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", default=TEX_PATH)
    parser.add_argument("--figure-dir", default=FIGURE_DIR)
    parser.add_argument("--figure-name", default=FIGURE_NAME)
    return parser.parse_args()


def best_of(family: list[str], normalized: dict[str, float]) -> tuple[str, float]:
    # Ties resolve to the first algorithm in table order.
    best = max(family, key=lambda algo: (normalized[algo], -family.index(algo)))
    return best, normalized[best]


def compute_gaps(rows: list[BaselineRow]):
    results = []
    for row in rows:
        normalized = dict(zip(ALGORITHM_COLUMNS, map(float, row.normalized()), strict=True))
        on_algo, on_val = best_of(ON_POLICY, normalized)
        off_algo, off_val = best_of(OFF_POLICY, normalized)
        results.append((row.scenario, on_algo, on_val, off_algo, off_val, on_val - off_val))
    return results


def label_bar(ax, x: float, height: float, text: str, inside_color: str, outside_color: str) -> None:
    if height >= INSIDE_LABEL_MIN:
        ax.text(x, height - 0.02, text, rotation=90, ha="center", va="top",
                fontsize=8, fontweight="bold", color="white")
    else:
        ax.text(x, height + 0.015, text, rotation=90, ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=outside_color)


def plot(results, output_pdf: Path, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.6, 6.0), dpi=300)
    ax.set_facecolor("#fbfbf7")
    width = 0.34
    xs = np.arange(len(results))

    for x, (_, on_algo, on_val, off_algo, off_val, gap) in zip(xs, results, strict=True):
        ax.bar(x - width / 2, on_val, width, color=ON_COLOR, edgecolor="#5a1a1d", linewidth=0.8, zorder=2)
        ax.bar(x + width / 2, off_val, width, color=OFF_COLOR, edgecolor="#0b4a5e", linewidth=0.8, zorder=2)
        label_bar(ax, x - width / 2, on_val, on_algo, "white", ON_TEXT)
        label_bar(ax, x + width / 2, off_val, off_algo, "white", OFF_TEXT)

        top = max(on_val, off_val)
        if top < INSIDE_LABEL_MIN:
            longest = max(len(on_algo), len(off_algo))
            top += 0.03 + 0.018 * longest
        ax.text(x, top + 0.03, f"{gap:+.2f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=ON_COLOR if gap >= 0 else OFF_COLOR)

    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in results], rotation=30, ha="right", fontsize=10)
    ax.set_xlim(-0.6, len(results) - 0.4)
    ax.set_ylim(0.0, 1.15)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_ylabel("Best family score (scenario-normalized)", fontsize=11)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(
        handles=[
            Patch(facecolor=ON_COLOR, edgecolor="#5a1a1d", label="Best on-policy"),
            Patch(facecolor=OFF_COLOR, edgecolor="#0b4a5e", label="Best off-policy"),
        ],
        loc="upper right", ncol=2, fontsize=10, frameon=True,
    )

    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = compute_gaps(parse_baseline_results(Path(args.tex)))

    print(f"{'Scenario':<24}{'On':>9}{'s~':>7}{'Off':>10}{'s~':>7}{'Gap':>8}")
    for scenario, on_algo, on_val, off_algo, off_val, gap in results:
        print(f"{scenario:<24}{on_algo:>9}{on_val:>7.3f}{off_algo:>10}{off_val:>7.3f}{gap:>+8.3f}")
    on_leads = sum(1 for r in results if r[5] > 0)
    print(f"On-policy strictly leads in {on_leads} of {len(results)} scenarios")

    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot(results, figure_dir / f"{args.figure_name}.pdf", figure_dir / f"{args.figure_name}.png")
    print(f"Wrote {figure_dir / args.figure_name}.pdf/.png")


if __name__ == "__main__":
    main()
