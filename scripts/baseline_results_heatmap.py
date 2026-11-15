"""Regenerate Figure 29 (baseline_results_heatmap) directly from the raw scores table of the paper.

The rows of the raw table (``tab:baseline_results_raw``) are parsed from the LaTeX source so the
figure cannot drift from the reported numbers. Normalization follows the caption:
s~ = clip(s / s_max, 0, 1), with s_max = 4 completed mazes for Lava Maze (no finite ceiling).
Bold cells are the ones the table bolds (best method plus any method whose 95% CI overlaps it);
in the top "Overall score" row, bold marks the highest mean normalized score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from paper_tables import ALGORITHM_COLUMNS as ALGORITHMS
from paper_tables import TEX_PATH, BaselineRow, parse_baseline_results


FIGURE_DIR = "paper/figure"
FIGURE_NAME = "baseline_results_heatmap"
CMAP = "RdYlBu_r"
DARK_TEXT = "#1a1a1a"
LIGHT_TEXT = "white"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", default=TEX_PATH)
    parser.add_argument("--figure-dir", default=FIGURE_DIR)
    parser.add_argument("--figure-name", default=FIGURE_NAME)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def normalize(rows: list[BaselineRow]) -> tuple[list[str], np.ndarray, np.ndarray]:
    scenarios = [row.scenario for row in rows]
    scores = np.asarray([row.normalized() for row in rows])
    bold = np.asarray([row.best for row in rows], dtype=bool)
    return scenarios, scores, bold


def text_color(rgba) -> str:
    red, green, blue = rgba[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return DARK_TEXT if luminance > 0.56 else LIGHT_TEXT


def draw_cells(ax, values: np.ndarray, bold: np.ndarray, cmap, norm) -> None:
    ax.pcolormesh(values, cmap=cmap, norm=norm, edgecolors="white", linewidth=1.5)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(col + 0.5, row + 0.5, f"{value:.2f}", ha="center", va="center", fontsize=11,
                    color=text_color(cmap(norm(value))), fontweight="bold" if bold[row, col] else "normal")
    ax.set_xlim(0, values.shape[1])
    ax.set_ylim(values.shape[0], 0)
    ax.tick_params(length=0, labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot(scenarios: list[str], scores: np.ndarray, bold: np.ndarray, output_pdf: Path, output_png: Path,
         dpi: int) -> None:
    cmap = plt.get_cmap(CMAP)
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    overall = scores.mean(axis=0, keepdims=True)
    overall_bold = np.isclose(overall, overall.max())

    fig, (ax_top, ax_main) = plt.subplots(
        2, 1, figsize=(10.35, 8.95), gridspec_kw={"height_ratios": [1, len(scenarios)], "hspace": 0.08}
    )

    draw_cells(ax_top, overall, overall_bold, cmap, norm)
    ax_top.set_yticks([0.5])
    ax_top.set_yticklabels(["Overall score"])
    ax_top.tick_params(labelright=True, right=False)
    for tick in ax_top.yaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")
        tick.label2.set_fontweight("bold")
    ax_top.xaxis.tick_top()
    ax_top.set_xticks(np.arange(len(ALGORITHMS)) + 0.5)
    ax_top.set_xticklabels(ALGORITHMS, rotation=40, ha="left", rotation_mode="anchor")

    draw_cells(ax_main, scores, bold, cmap, norm)
    ax_main.set_yticks(np.arange(len(scenarios)) + 0.5)
    ax_main.set_yticklabels(scenarios)
    ax_main.tick_params(labelright=True, right=False)
    ax_main.set_xticks([])

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scenarios, scores, bold = normalize(parse_baseline_results(Path(args.tex)))

    print(f"{'Scenario':<24}" + "".join(f"{algo:>9}" for algo in ALGORITHMS))
    print(f"{'Overall score':<24}" + "".join(f"{value:>9.3f}" for value in scores.mean(axis=0)))
    for scenario, row, flags in zip(scenarios, scores, bold, strict=True):
        cells = "".join(f"{(f'*{value:.3f}' if flag else f'{value:.3f}'):>9}" for value, flag in zip(row, flags))
        print(f"{scenario:<24}{cells}")

    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot(scenarios, scores, bold, figure_dir / f"{args.figure_name}.pdf", figure_dir / f"{args.figure_name}.png",
         args.dpi)
    print(f"Wrote {figure_dir / args.figure_name}.pdf/.png")


if __name__ == "__main__":
    main()
