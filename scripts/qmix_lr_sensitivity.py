from __future__ import annotations

import argparse
import bisect
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea


METRIC_KEY = "policy_stats/avg_true_objective"
FPS_KEY = "perf/_fps"
DEFAULT_FRONTIER = 50_000_000
TIE_BREAK_EPS = 1.0
MOVING_AVERAGE_WINDOW = 11
COLORS = ["#1f6feb", "#238636", "#d29922", "#cf222e", "#8250df"]


@dataclass(frozen=True)
class RunSummary:
    experiment: str
    learning_rate: float
    seed: int
    final_step: int
    score_at_frontier: float | None
    smoothed_score_at_frontier: float | None
    final_score: float
    tail_mean_fps: float
    status: str
    event_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the QMIX Armory Siege learning-rate sensitivity sweep.")
    parser.add_argument("--run-root", default="results/qmix_lr_sensitivity")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-name", default="qmix_lr_sensitivity")
    parser.add_argument("--frontier", type=int, default=DEFAULT_FRONTIER)
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int = MOVING_AVERAGE_WINDOW) -> np.ndarray:
    if len(values) < 3:
        return values
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    window = max(window, 3)
    if window % 2 == 0:
        window -= 1
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def scalar_at_or_before(steps: np.ndarray, values: np.ndarray, step: int) -> float | None:
    idx = bisect.bisect_right(steps.tolist(), step) - 1
    if idx < 0:
        return None
    return float(values[idx])


def event_file_for_run(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob(".summary/0/events.out.tfevents.*"))
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event file found under {run_dir}")
    return candidates[-1]


def load_run(run_dir: Path, frontier: int) -> tuple[RunSummary, dict[str, np.ndarray]]:
    config = json.loads((run_dir / "config.json").read_text())
    event_file = event_file_for_run(run_dir)

    acc = ea.EventAccumulator(str(event_file))
    acc.Reload()

    score_events = acc.Scalars(METRIC_KEY)
    fps_events = acc.Scalars(FPS_KEY)
    if not score_events or not fps_events:
        raise ValueError(f"Missing required scalar streams in {event_file}")

    steps = np.asarray([int(event.step) for event in score_events], dtype=np.int64)
    scores = np.asarray([float(event.value) for event in score_events], dtype=float)
    scores_smooth = moving_average(scores)
    fps_tail = [float(event.value) for event in fps_events[-20:]]
    final_step = int(steps[-1])
    raw_frontier = scalar_at_or_before(steps, scores, frontier)
    smooth_frontier = scalar_at_or_before(steps, scores_smooth, frontier)
    status = "valid" if final_step >= frontier else "incomplete"

    summary = RunSummary(
        experiment=str(config["experiment"]),
        learning_rate=float(config["learning_rate"]),
        seed=int(config["seed"]),
        final_step=final_step,
        score_at_frontier=raw_frontier,
        smoothed_score_at_frontier=smooth_frontier,
        final_score=float(scores[-1]),
        tail_mean_fps=float(statistics.mean(fps_tail)),
        status=status,
        event_file=str(event_file),
    )
    series = {
        "steps": steps.astype(float),
        "scores": scores,
        "scores_smooth": scores_smooth,
    }
    return summary, series


def discover_runs(run_root: Path, frontier: int) -> tuple[list[RunSummary], dict[float, dict[str, np.ndarray]]]:
    summaries: list[RunSummary] = []
    series_by_lr: dict[float, dict[str, np.ndarray]] = {}
    for config_path in sorted(run_root.rglob("config.json")):
        run_dir = config_path.parent
        summary, series = load_run(run_dir, frontier)
        summaries.append(summary)
        series_by_lr[summary.learning_rate] = series
    if not summaries:
        raise FileNotFoundError(f"No sensitivity runs found under {run_root}")
    summaries.sort(key=lambda item: item.learning_rate)
    return summaries, series_by_lr


def select_representative_lr(summaries: list[RunSummary], frontier: int) -> tuple[float | None, str]:
    valid = [summary for summary in summaries if summary.status == "valid" and summary.smoothed_score_at_frontier is not None]
    if not valid:
        return None, "No run reached the configured frontier."

    ranked = sorted(
        valid,
        key=lambda item: (-float(item.smoothed_score_at_frontier), float(item.learning_rate)),
    )
    winner = ranked[0]
    if len(ranked) > 1:
        margin = float(winner.smoothed_score_at_frontier) - float(ranked[1].smoothed_score_at_frontier)
        if margin <= TIE_BREAK_EPS:
            smaller_lr = min(winner.learning_rate, ranked[1].learning_rate)
            if smaller_lr != winner.learning_rate:
                winner = next(item for item in ranked if item.learning_rate == smaller_lr)
    rationale = (
        f"Selected lr={winner.learning_rate:g} by highest smoothed {METRIC_KEY} at {frontier // 1_000_000}M steps; "
        f"ties within {TIE_BREAK_EPS:.1f} prefer the smaller learning rate."
    )
    return winner.learning_rate, rationale


def build_figure(
    summaries: list[RunSummary],
    series_by_lr: dict[float, dict[str, np.ndarray]],
    frontier: int,
    output_pdf: Path,
    output_png: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbf8")

    max_step = frontier
    for idx, summary in enumerate(summaries):
        color = COLORS[idx % len(COLORS)]
        series = series_by_lr[summary.learning_rate]
        env_steps = series["steps"] / 1_000_000.0
        max_step = max(max_step, int(series["steps"][-1]))
        ax.plot(env_steps, series["scores"], color=color, alpha=0.16, linewidth=0.9)
        ax.plot(
            env_steps,
            series["scores_smooth"],
            color=color,
            linewidth=2.0,
            label=f"lr={summary.learning_rate:g}",
        )
        if summary.smoothed_score_at_frontier is not None:
            ax.scatter(
                [frontier / 1_000_000.0],
                [summary.smoothed_score_at_frontier],
                s=28,
                color=color,
                edgecolors="white",
                linewidth=0.7,
                zorder=5,
            )

    ax.axvline(frontier / 1_000_000.0, color="#4a5568", linestyle="--", linewidth=1.2)
    ax.text(
        frontier / 1_000_000.0 + 0.7,
        ax.get_ylim()[1] - 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
        f"{int(frontier / 1_000_000)}M frontier",
        color="#4a5568",
        fontsize=8.5,
        va="top",
    )
    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("Average shaped episodic return")
    ax.set_xlim(0, max_step / 1_000_000.0 + 1.0)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight")


def write_outputs(
    summaries: list[RunSummary],
    representative_lr: float | None,
    rationale: str,
    frontier: int,
    output_dir: Path,
    output_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_name}_summary.json"
    md_path = output_dir / f"{output_name}_summary.md"

    payload = {
        "frontier_steps": frontier,
        "selection_tie_break_eps": TIE_BREAK_EPS,
        "representative_learning_rate": representative_lr,
        "selection_rationale": rationale,
        "runs": [asdict(summary) for summary in summaries],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# QMIX Learning-Rate Sensitivity",
        "",
        f"- Frontier: {frontier:,} environment steps",
        f"- Representative learning rate: `{representative_lr:g}`" if representative_lr is not None else "- Representative learning rate: inconclusive",
        f"- Selection rule: {rationale}",
        "",
        "| LR | Status | Final Step | Smoothed Score @ Frontier | Final Score | Tail Mean FPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        smoothed = "n/a" if summary.smoothed_score_at_frontier is None else f"{summary.smoothed_score_at_frontier:.3f}"
        lines.append(
            f"| `{summary.learning_rate:g}` | {summary.status} | {summary.final_step:,} | {smoothed} | "
            f"{summary.final_score:.3f} | {summary.tail_mean_fps:.1f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)

    summaries, series_by_lr = discover_runs(run_root, args.frontier)
    representative_lr, rationale = select_representative_lr(summaries, args.frontier)

    learning_pdf = output_dir / f"{args.output_name}_learning.pdf"
    learning_png = output_dir / f"{args.output_name}_learning.png"
    build_figure(summaries, series_by_lr, args.frontier, learning_pdf, learning_png)
    write_outputs(summaries, representative_lr, rationale, args.frontier, output_dir, args.output_name)


if __name__ == "__main__":
    main()
