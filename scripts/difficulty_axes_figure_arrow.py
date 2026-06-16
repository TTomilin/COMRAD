from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch


OUTPUT_PATH = Path("results/difficulty_axes_arrow.png")


SCENARIOS = [
    {
        "name": "Stag Hunt Arena",
        "x": 1.35,
        "y": 1.15,
        "color": "#65a30d",
        "label_dx": 0.26,
        "label_dy": 0.18,
        "ha": "left",
    },
    {
        "name": "Rhythm Sync",
        "x": 2.45,
        "y": 1.35,
        "color": "#0f766e",
        "label_dx": 0.28,
        "label_dy": 0.28,
        "ha": "left",
    },
    {
        "name": "Foraging Commons",
        "x": 2.15,
        "y": 2.85,
        "color": "#0284c7",
        "label_dx": 0.25,
        "label_dy": 0.25,
        "ha": "left",
    },
    {
        "name": "Armory Siege",
        "x": 3.2,
        "y": 3.4,
        "color": "#7c3aed",
        "label_dx": 0.22,
        "label_dy": 0.02,
        "ha": "left",
    },
    {
        "name": "Platform Chain",
        "x": 3.0,
        "y": 4.45,
        "color": "#2563eb",
        "label_dx": 0.3,
        "label_dy": -0.3,
        "ha": "left",
    },
    {
        "name": "Lava Maze",
        "x": 4.8,
        "y": 4.75,
        "color": "#111827",
        "label_dx": -0.36,
        "label_dy": -0.36,
        "ha": "right",
    },
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def add_axis_background(ax: plt.Axes) -> None:
    quadrants = [
        ((0.5, 0.5), 2.5, 2.5, "#ecfccb"),
        ((3.0, 0.5), 2.5, 2.5, "#fef3c7"),
        ((0.5, 3.0), 2.5, 2.5, "#dbeafe"),
        ((3.0, 3.0), 2.5, 2.5, "#fee2e2"),
    ]
    for (x, y), w, h, color in quadrants:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.01,rounding_size=0.12",
                linewidth=0.0,
                facecolor=color,
                alpha=0.7,
                zorder=0,
            )
        )


def add_quadrant_labels(ax: plt.Axes) -> None:
    labels = [
        (1.75, 5.25, "Spatially Mild"),
        (4.25, 5.25, "Spatially Demanding"),
        (0.63, 1.75, "Mechanically Easier"),
        (0.63, 4.25, "Mechanically Harder"),
    ]
    for x, y, text in labels:
        rotation = 90 if x < 1.0 else 0
        ax.text(
            x,
            y,
            text,
            fontsize=12,
            color="#475569",
            rotation=rotation,
            ha="center",
            va="center",
            weight="bold",
        )


def add_scenarios(ax: plt.Axes) -> None:
    for scenario in SCENARIOS:
        x = scenario["x"]
        y = scenario["y"]
        color = scenario["color"]
        ax.scatter(
            [x],
            [y],
            s=180,
            color=color,
            edgecolors="white",
            linewidths=2.0,
            zorder=6,
        )
        dx = scenario["label_dx"]
        dy = scenario["label_dy"]
        tx = x + dx
        ty = y + dy
        ax.plot([x, tx], [y, ty], color="#94a3b8", lw=1.25, zorder=5)
        ax.text(
            tx,
            ty,
            f"{scenario['name']}",
            fontsize=9.4,
            color="#334155",
            ha=scenario["ha"],
            va="center",
            linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=color, linewidth=1.2),
            zorder=5,
        )


def add_axis_arrows(ax: plt.Axes) -> None:
    arrowprops = dict(
        arrowstyle="-|>",
        color="#64748b",
        lw=2.5,
        mutation_scale=20,
    )
    y_arrow_x = -0.05
    x_arrow_y = -0.11

    ax.add_patch(FancyArrowPatch(
        (y_arrow_x, 0.08), (y_arrow_x, 0.91),
        transform=ax.transAxes,
        clip_on=False,
        zorder=10,
        **arrowprops,
    ))
    ax.text(
        y_arrow_x, 0.455,
        "more complex layouts",
        transform=ax.transAxes,
        rotation=90,
        fontsize=16, color="#475569", style="italic",
        va="center", ha="center",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none"),
        zorder=11,
    )

    ax.add_patch(FancyArrowPatch(
        (0.08, x_arrow_y), (0.83, x_arrow_y),
        transform=ax.transAxes,
        clip_on=False,
        zorder=10,
        **arrowprops,
    ))
    ax.text(
        0.455, x_arrow_y,
        "harder task mechanics",
        transform=ax.transAxes,
        fontsize=16, color="#475569", style="italic",
        va="center", ha="center",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none"),
        zorder=11,
    )


def make_figure() -> None:
    plt.rcParams["font.family"] = "sans-serif"

    fig, ax = plt.subplots(figsize=(10.6, 5.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_xlim(0, 6)
    ax.set_ylim(0.2, 5.45)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])

    add_axis_background(ax)
    ax.grid(True, color="#cbd5e1", linestyle=(0, (2, 4)), linewidth=1.0, alpha=0.9, zorder=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)

    add_axis_arrows(ax)
    add_quadrant_labels(ax)
    add_scenarios(ax)

    fig.subplots_adjust(left=0.11, right=0.985, top=0.97, bottom=0.17)
    ensure_parent(OUTPUT_PATH)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    make_figure()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
