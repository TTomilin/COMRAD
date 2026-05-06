from __future__ import annotations

import argparse
import bisect
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea


AGENT_COUNTS = [2, 3, 4, 6, 8]
FRAMESKIP = 4
COMMON_FRONTIER = 30_000_000
PLOT_MAX_STEPS = 60_000_000
COLORS = {
    2: "#2f855a",
    3: "#2b6cb0",
    4: "#805ad5",
    6: "#dd6b20",
    8: "#c53030",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Armory Siege agent-scaling figure.")
    parser.add_argument("--event-prefix", default="results/scaling/armory_scaling_n")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-name", default="agent_scaling")
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int = 11) -> np.ndarray:
    if len(values) < 3:
        return values
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    window = max(window, 3)
    if window % 2 == 0:
        window -= 1
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def scalar_at_or_before(events, step: int) -> float:
    steps = [event.step for event in events]
    idx = bisect.bisect_right(steps, step) - 1
    if idx < 0:
        raise ValueError(f"No scalar logged at or before step {step}")
    return float(events[idx].value)


def load_run(path: Path) -> dict[str, np.ndarray | float]:
    acc = ea.EventAccumulator(str(path))
    acc.Reload()

    score_events = acc.Scalars("policy_stats/avg_true_objective")
    fps_events = acc.Scalars("perf/_fps")

    env_steps = np.asarray([float(event.step) for event in score_events], dtype=float)
    scores = np.asarray([float(event.value) for event in score_events], dtype=float)
    fps_tail = [float(event.value) for event in fps_events[-20:]]
    final_frames = float(fps_events[-1].step)
    mean_fps = float(statistics.mean(fps_tail))

    return {
        "env_steps": env_steps,
        "scores": scores,
        "scores_smooth": moving_average(scores),
        "score_at_50m": scalar_at_or_before(score_events, COMMON_FRONTIER),
        "final_frames_m": final_frames / 1_000_000.0,
        "env_steps_s": mean_fps / (int(path.stem.split("n")[-1]) * FRAMESKIP),
    }


def build_learning_figure(data: dict[int, dict[str, np.ndarray | float]], output_pdf: Path, output_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, ax_left = plt.subplots(figsize=(6.2, 4.2), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax_left.set_facecolor("#fbfbf8")

    for n in AGENT_COUNTS:
        run = data[n]
        env_steps = run["env_steps"] / 1_000_000.0
        scores = run["scores"]
        scores_smooth = run["scores_smooth"]
        color = COLORS[n]

        ax_left.plot(env_steps, scores, color=color, alpha=0.16, linewidth=0.9)
        ax_left.plot(env_steps, scores_smooth, color=color, linewidth=2.2, label=fr"$N={n}$")
        ax_left.scatter(
            [COMMON_FRONTIER / 1_000_000.0],
            [run["score_at_50m"]],
            s=28,
            color=color,
            edgecolors="white",
            linewidth=0.7,
            zorder=5,
        )

    ax_left.axvline(COMMON_FRONTIER / 1_000_000.0, color="#4a5568", linestyle="--", linewidth=1.2)
    ax_left.axvline(PLOT_MAX_STEPS / 1_000_000.0, color="#a0aec0", linestyle=":", linewidth=0.8)
    ax_left.text(
        COMMON_FRONTIER / 1_000_000.0 + 0.9,
        ax_left.get_ylim()[1] - 0.08 * (ax_left.get_ylim()[1] - ax_left.get_ylim()[0]),
        "50M frontier",
        color="#4a5568",
        fontsize=8.5,
        va="top",
    )
    ax_left.set_xlabel("Environment Steps (Millions)")
    ax_left.set_ylabel("Average shaped episodic return")
    ax_left.set_xlim(0, PLOT_MAX_STEPS / 1_000_000.0 + 1.0)
    ax_left.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.legend(loc="upper left", ncol=1, frameon=False)

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight")


def build_throughput_figure(data: dict[int, dict[str, np.ndarray | float]], output_pdf: Path, output_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, ax_right = plt.subplots(figsize=(4.4, 4.2), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax_right.set_facecolor("#fbfbf8")

    x = np.arange(len(AGENT_COUNTS), dtype=float)
    env_steps_s = np.asarray([data[n]["env_steps_s"] for n in AGENT_COUNTS], dtype=float)
    final_frames_m = np.asarray([data[n]["final_frames_m"] for n in AGENT_COUNTS], dtype=float)

    bars = ax_right.bar(
        x,
        env_steps_s,
        width=0.62,
        color="#7fb3d5",
        edgecolor="#2b6cb0",
        linewidth=0.9,
        zorder=3,
        label="Env Steps/s",
    )
    ax_right.set_xlabel("Number of Agents")
    ax_right.set_ylabel("Env Steps/s")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([str(n) for n in AGENT_COUNTS])
    ax_right.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax_right.spines["top"].set_visible(False)

    ax_right_twin = ax_right.twinx()
    ax_right_twin.plot(
        x,
        final_frames_m,
        color="#c05621",
        marker="o",
        markersize=5.5,
        linewidth=2.0,
        label="Final Frames",
        zorder=4,
    )
    ax_right_twin.set_ylabel("Final Frames (Millions)")
    ax_right_twin.spines["top"].set_visible(False)

    for bar, val in zip(bars, env_steps_s, strict=True):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 22,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#1f2937",
        )

    for xi, val in zip(x, final_frames_m, strict=True):
        ax_right_twin.text(
            xi,
            val + 1.7,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#9c4221",
        )

    handles_1, labels_1 = ax_right.get_legend_handles_labels()
    handles_2, labels_2 = ax_right_twin.get_legend_handles_labels()
    ax_right.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right", frameon=False)

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for n in AGENT_COUNTS:
        path = Path(f"{args.event_prefix}{n}.tfevents")
        if not path.exists():
            raise FileNotFoundError(path)
        data[n] = load_run(path)

    learning_pdf = output_dir / f"{args.output_name}_learning.pdf"
    learning_png = output_dir / f"{args.output_name}_learning.png"
    throughput_pdf = output_dir / f"{args.output_name}_throughput.pdf"
    throughput_png = output_dir / f"{args.output_name}_throughput.png"
    build_learning_figure(data, learning_pdf, learning_png)
    build_throughput_figure(data, throughput_pdf, throughput_png)


if __name__ == "__main__":
    main()
