"""Redraw the left panel of the curriculum figure (curriculum_protocol_best) from the paper's table.

``platform_chain_curriculum_protocol.py`` draws this panel from its evaluation summary JSON. When that
summary is not available, this script redraws the panel from ``tab:curriculum_protocol_results``,
which lists, for every variant, the frames at the best retained checkpoint and its target progress,
i.e. exactly the points of the panel. The styling is the same as in ``plot_summary``.

The y-range of the original panel is the min/max over *all* retained checkpoints +- 0.6
(``retained_progress_limits``), and the lowest retained checkpoint is not in the table, so the
default ``--ylim`` is the range of the original figure (3.65 to 7.5375).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from platform_chain_curriculum_protocol import VARIANT_LABELS, VARIANT_ORDER, VARIANT_STYLES


TEX_PATH = "paper/iclr2027/iclr2027_conference.tex"
FIGURE_DIR = "paper/figure"
FIGURE_NAME = "curriculum_protocol_best"
TABLE_LABEL = r"\label{tab:curriculum_protocol_results}"
LABEL_TO_VARIANT = {label: key for key, label in VARIANT_LABELS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", default=TEX_PATH)
    parser.add_argument("--figure-dir", default=FIGURE_DIR)
    parser.add_argument("--figure-name", default=FIGURE_NAME)
    parser.add_argument("--ylim", nargs=2, type=float, default=(3.65, 7.5375))
    return parser.parse_args()


def strip_bold(cell: str) -> str:
    cell = cell.strip()
    if cell.startswith(r"\textbf{") and cell.endswith("}"):
        cell = cell[len(r"\textbf{"):-1]
    return cell.strip()


def parse_table(tex_path: Path) -> dict[str, tuple[float, float]]:
    text = tex_path.read_text()
    start = text.index(TABLE_LABEL)
    body = text[text.index(r"\midrule", start) + len(r"\midrule"): text.index(r"\bottomrule", start)]

    points: dict[str, tuple[float, float]] = {}
    for raw_row in body.split(r"\\"):
        row = " ".join(raw_row.split())
        if not row:
            continue
        cells = [strip_bold(cell) for cell in row.split("&")]
        if len(cells) != 4:
            raise ValueError(f"Expected 4 cells, got {len(cells)}: {row!r}")
        points[LABEL_TO_VARIANT[cells[0]]] = (float(cells[1]), float(cells[2]))
    missing = [key for key in VARIANT_ORDER if key not in points]
    if missing:
        raise ValueError(f"Missing variants in the table: {missing}")
    return points


def plot(points: dict[str, tuple[float, float]], y_limits: tuple[float, float], output_dir: Path, name: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
        }
    )
    y_min, y_max = y_limits

    fig, ax = plt.subplots(figsize=(4.8, 4.1), dpi=300)
    ax.set_xlabel("Frames at best retained checkpoint (millions)")
    ax.set_ylabel("Common-target progress (checkpoints)")
    ax.set_xlim(0.0, 130.0)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.8, alpha=0.7)

    for key in VARIANT_ORDER:
        style = VARIANT_STYLES[key]
        x_best, y_best = points[key]
        ax.hlines(y_best, xmin=0.0, xmax=x_best, colors=style["color"], linestyles=":", linewidth=1.5, alpha=0.9,
                  zorder=1)
        ax.vlines(x_best, ymin=y_min, ymax=y_best, colors=style["color"], linestyles=":", linewidth=1.5, alpha=0.9,
                  zorder=1)
        ax.scatter([x_best], [y_best], color=style["color"], marker=style["marker"], s=115, edgecolors="white",
                   linewidths=0.9, zorder=3)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    points = parse_table(Path(args.tex))
    for key in VARIANT_ORDER:
        x_best, y_best = points[key]
        print(f"{VARIANT_LABELS[key]:<20}{x_best:>8.1f}M{y_best:>8.2f}")
    plot(points, tuple(args.ylim), Path(args.figure_dir), args.figure_name)
    print(f"Wrote {Path(args.figure_dir) / args.figure_name}.pdf/.png")


if __name__ == "__main__":
    main()
