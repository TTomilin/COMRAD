from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import gymnasium as gym
import numpy as np
import torch

from comrad.train import parse_args, register_model_factory, register_vizdoom_components
from comrad.utils.doom_utils import get_num_agents
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.enjoy import make_env as make_eval_env
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size


DEFAULT_RUN_ROOT = Path(
    "train_dir/mode3_platform_chain_curriculum_compare_20260506_162935/platform_chain_curriculum_compare"
)
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "curriculum_protocol_eval_summary.json"
DEFAULT_TABLE_TEX = DEFAULT_OUTPUT_DIR / "curriculum_protocol_results_table.tex"
DEFAULT_PAPER_FIGURE_DIR = Path("paper/figure")
DEFAULT_FIGURE_NAME = "curriculum_protocol"
DEFAULT_TARGET_ENV = "platform_chain"
DEFAULT_TARGET_MAX_SCORE = 47.0

VARIANT_ORDER = [
    "baseline",
    "uniform",
    "sequential",
    "learning_progress",
    "omni",
    "plr",
]
VARIANT_LABELS = {
    "baseline": "Direct hard",
    "uniform": "Uniform",
    "sequential": "Sequential",
    "learning_progress": "Learning progress",
    "omni": "OMNI",
    "plr": "PLR",
}
VARIANT_SHORT_LABELS = {
    "baseline": "Direct",
    "uniform": "Uniform",
    "sequential": "Seq.",
    "learning_progress": "LP",
    "omni": "OMNI",
    "plr": "PLR",
}
VARIANT_STYLES = {
    "baseline": {"color": "#595959", "marker": "o"},
    "uniform": {"color": "#2b8cbe", "marker": "s"},
    "sequential": {"color": "#d95f0e", "marker": "D"},
    "learning_progress": {"color": "#4d9221", "marker": "^"},
    "omni": {"color": "#542788", "marker": "v"},
    "plr": {"color": "#c51b7d", "marker": "P"},
}
CHECKPOINT_RE = re.compile(
    r"^(?P<kind>best|checkpoint)_(?P<train_updates>\d+)_(?P<train_env_steps>\d+)"
)


@dataclass(frozen=True)
class CheckpointSpec:
    path: str
    kind: str
    train_updates: int
    train_env_steps: int


@dataclass(frozen=True)
class VariantRun:
    key: str
    label: str
    run_dir: str
    checkpoints: list[CheckpointSpec]


@dataclass(frozen=True)
class CheckpointEval:
    path: str
    kind: str
    train_updates: int
    train_env_steps: int
    joint_episode_scores: list[float]
    mean_true_objective: float
    std_true_objective: float
    mean_normalized_progress: float
    std_normalized_progress: float


def parse_args_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retained curriculum checkpoints on a common Platform Chain target and render the paper figure/table."
    )
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--target-env", default=DEFAULT_TARGET_ENV)
    parser.add_argument("--target-max-score", default=DEFAULT_TARGET_MAX_SCORE, type=float)
    parser.add_argument("--episodes", default=8, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-deterministic", default=False, action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--table-output", default=str(DEFAULT_TABLE_TEX))
    parser.add_argument("--figure-name", default=DEFAULT_FIGURE_NAME)
    parser.add_argument("--paper-figure-dir", default=str(DEFAULT_PAPER_FIGURE_DIR))
    parser.add_argument("--variants", default="")
    parser.add_argument("--max-checkpoints-per-variant", default=0, type=int)
    parser.add_argument("--skip-eval", default=False, action="store_true")
    return parser.parse_args()


def variant_key_from_run_name(run_name: str) -> str:
    for key in VARIANT_ORDER:
        token = f"_platform_chain_{key}_"
        if token in run_name:
            return key
    raise ValueError(f"Unrecognized curriculum run name: {run_name}")


def parse_checkpoint_filename(path: Path) -> CheckpointSpec:
    match = CHECKPOINT_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unrecognized checkpoint filename: {path.name}")
    return CheckpointSpec(
        path=str(path),
        kind=match.group("kind"),
        train_updates=int(match.group("train_updates")),
        train_env_steps=int(match.group("train_env_steps")),
    )


def discover_variant_runs(run_root: Path) -> list[VariantRun]:
    run_dirs: dict[str, Path] = {}
    for config_path in run_root.rglob("config.json"):
        run_dir = config_path.parent
        key = variant_key_from_run_name(run_dir.name)
        run_dirs[key] = run_dir

    missing = [key for key in VARIANT_ORDER if key not in run_dirs]
    if missing:
        raise FileNotFoundError(f"Missing expected curriculum runs: {missing}")

    variants: list[VariantRun] = []
    for key in VARIANT_ORDER:
        run_dir = run_dirs[key]
        checkpoint_dir = run_dir / "checkpoint_p0"
        checkpoints = sorted(
            (parse_checkpoint_filename(path) for path in checkpoint_dir.glob("*.pth")),
            key=lambda item: (item.train_env_steps, item.kind != "best", item.path),
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        variants.append(
            VariantRun(
                key=key,
                label=VARIANT_LABELS[key],
                run_dir=str(run_dir),
                checkpoints=checkpoints,
            )
        )
    return variants


def filter_variants(
    variants: list[VariantRun],
    selected_keys: set[str],
    max_checkpoints_per_variant: int,
) -> list[VariantRun]:
    filtered: list[VariantRun] = []
    for variant in variants:
        if selected_keys and variant.key not in selected_keys:
            continue
        checkpoints = variant.checkpoints
        if max_checkpoints_per_variant > 0:
            checkpoints = checkpoints[:max_checkpoints_per_variant]
        filtered.append(
            VariantRun(
                key=variant.key,
                label=variant.label,
                run_dir=variant.run_dir,
                checkpoints=checkpoints,
            )
        )
    return filtered


def build_eval_cfg(run_dir: Path, target_env: str, device: str, deterministic: bool) -> Any:
    argv = [
        "--algo=MAPPO",
        f"--train_dir={run_dir.parent}",
        f"--experiment={run_dir.name}",
        f"--env={target_env}",
        f"--device={device}",
        "--with_wandb=False",
        "--wandb_record_every=0",
        "--no_render",
    ]
    if deterministic:
        argv.append("--eval_deterministic=True")
    cfg = parse_args(argv=argv, evaluation=True)
    cfg = load_from_checkpoint(cfg)

    cfg.train_dir = str(run_dir.parent)
    cfg.experiment = run_dir.name
    cfg.env = target_env
    cfg.wad_batch = ""
    cfg.curriculum = "uniform"
    cfg.no_render = True
    cfg.save_video = False
    cfg.with_wandb = False
    cfg.device = device
    cfg.eval_deterministic = deterministic
    cfg.policy_index = 0
    cfg.num_envs = 1

    if cfg.num_agents < 1:
        cfg.num_agents = get_num_agents(cfg, target_env)

    register_model_factory(cfg)
    return cfg


def joint_true_objective(infos: Any, rewards: torch.Tensor) -> float:
    if isinstance(infos, dict):
        if "true_objective" in infos:
            return float(infos["true_objective"])
        return float(torch.mean(rewards).item())

    if isinstance(infos, (list, tuple)):
        objectives = [
            float(agent_info["true_objective"])
            for agent_info in infos
            if isinstance(agent_info, dict) and "true_objective" in agent_info
        ]
        if objectives:
            return float(np.mean(objectives))

    return float(torch.mean(rewards).item())


def maybe_reshape_deterministic_tuple_actions(action_space: Any, num_agents: int, actions: torch.Tensor) -> torch.Tensor:
    if not isinstance(action_space, gym.spaces.Tuple) or num_agents <= 1:
        return actions

    if actions.ndim != 2 or actions.shape[0] != 1:
        return actions

    num_heads = len(action_space.spaces)
    flat = actions.squeeze(0)
    if flat.numel() != num_heads * num_agents:
        return actions

    return flat.view(num_heads, num_agents).transpose(0, 1).contiguous()


class CheckpointEvaluator:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
        self.env = make_eval_env(cfg, render_mode=None)
        self.env_info = extract_env_info(self.env, cfg)
        self.actor_critic = create_actor_critic(cfg, self.env.observation_space, self.env.action_space)
        self.actor_critic.eval()
        self.actor_critic.model_to_device(self.device)

    def close(self) -> None:
        self.env.close()

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_dict = Learner.load_checkpoint([str(checkpoint_path)], self.device)
        if checkpoint_dict is None:
            raise RuntimeError(f"Could not load checkpoint: {checkpoint_path}")
        self.actor_critic.load_state_dict(checkpoint_dict["model"])

    def evaluate_checkpoint(self, checkpoint: CheckpointSpec, num_joint_episodes: int, target_max_score: float) -> CheckpointEval:
        self.load_checkpoint(Path(checkpoint.path))

        obs, infos = self.env.reset()
        action_mask = obs.pop("action_mask").to(self.device) if "action_mask" in obs else None
        rnn_states = torch.zeros([self.env.num_agents, get_rnn_size(self.cfg)], dtype=torch.float32, device=self.device)
        joint_scores: list[float] = []

        with torch.no_grad():
            while len(joint_scores) < num_joint_episodes:
                normalized_obs = prepare_and_normalize_obs(self.actor_critic, obs)
                policy_outputs = self.actor_critic(normalized_obs, rnn_states, action_mask=action_mask)
                actions = policy_outputs["actions"]

                if self.cfg.eval_deterministic:
                    actions = argmax_actions(self.actor_critic.action_distribution())
                    actions = maybe_reshape_deterministic_tuple_actions(self.env.action_space, self.env.num_agents, actions)

                if actions.ndim == 1:
                    actions = unsqueeze_tensor(actions, dim=-1)
                actions = preprocess_actions(self.env_info, actions)
                rnn_states = policy_outputs["new_rnn_states"]

                obs, rewards, terminated, truncated, infos = self.env.step(actions)
                action_mask = obs.pop("action_mask").to(self.device) if "action_mask" in obs else None
                dones = make_dones(terminated, truncated).cpu().numpy()

                for agent_idx, done_flag in enumerate(dones):
                    if done_flag:
                        rnn_states[agent_idx].zero_()

                if np.all(dones):
                    joint_scores.append(joint_true_objective(infos, rewards))

        raw_scores = np.asarray(joint_scores, dtype=float)
        normalized_scores = np.clip(raw_scores / target_max_score, 0.0, 1.0)
        return CheckpointEval(
            path=checkpoint.path,
            kind=checkpoint.kind,
            train_updates=checkpoint.train_updates,
            train_env_steps=checkpoint.train_env_steps,
            joint_episode_scores=joint_scores,
            mean_true_objective=float(np.mean(raw_scores)),
            std_true_objective=float(np.std(raw_scores)),
            mean_normalized_progress=float(np.mean(normalized_scores)),
            std_normalized_progress=float(np.std(normalized_scores)),
        )


def evaluate_variants(
    variants: list[VariantRun],
    target_env: str,
    target_max_score: float,
    episodes: int,
    device: str,
    deterministic: bool,
) -> dict[str, Any]:
    register_vizdoom_components()

    variant_payloads: list[dict[str, Any]] = []
    for variant in variants:
        run_dir = Path(variant.run_dir)
        cfg = build_eval_cfg(run_dir, target_env, device, deterministic)
        evaluator = CheckpointEvaluator(cfg)
        try:
            checkpoint_payloads = [
                asdict(evaluator.evaluate_checkpoint(checkpoint, episodes, target_max_score))
                for checkpoint in variant.checkpoints
            ]
        finally:
            evaluator.close()

        variant_payloads.append(
            {
                "key": variant.key,
                "label": variant.label,
                "run_dir": variant.run_dir,
                "checkpoints": checkpoint_payloads,
            }
        )

    return {
        "target_env": target_env,
        "target_max_score": target_max_score,
        "joint_eval_episodes_per_checkpoint": episodes,
        "eval_deterministic": deterministic,
        "variants": variant_payloads,
    }


def best_checkpoint_payload(variant: dict[str, Any]) -> dict[str, Any]:
    return max(
        variant["checkpoints"],
        key=lambda item: (float(item["mean_true_objective"]), -int(item["train_env_steps"])),
    )


def final_checkpoint_payload(variant: dict[str, Any]) -> dict[str, Any]:
    return max(variant["checkpoints"], key=lambda item: int(item["train_env_steps"]))


def format_step_millions(env_steps: int) -> str:
    return f"{env_steps / 1_000_000.0:.1f}"


def build_small_table(summary: dict[str, Any]) -> str:
    episode_count = int(summary["joint_eval_episodes_per_checkpoint"])
    variants = {variant["key"]: variant for variant in summary["variants"]}
    present_keys = [key for key in VARIANT_ORDER if key in variants]
    best_score = max(float(best_checkpoint_payload(variant)["mean_true_objective"]) for variant in variants.values())
    best_final_score = max(float(final_checkpoint_payload(variant)["mean_true_objective"]) for variant in variants.values())

    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{4.6pt}",
        (
            "  \\caption{Common-target evaluation on the default \\textit{Platform Chain} task "
            f"(maximum progress $=47$). Each entry reports the mean joint checkpoint progress over {episode_count} "
            "evaluation episodes for the retained best checkpoint and the last retained checkpoint available in the downloaded archive.}"
        ),
        "  \\label{tab:curriculum_protocol_results}",
        "  \\begin{tabular}{lccc}",
        "    \\toprule",
        "    Variant & Best step (M) & Best target progress & Last retained progress \\\\",
        "    \\midrule",
    ]

    for key in present_keys:
        variant = variants[key]
        best_ckpt = best_checkpoint_payload(variant)
        final_ckpt = final_checkpoint_payload(variant)
        best_value = f"{best_ckpt['mean_true_objective']:.2f}"
        final_value = f"{float(final_ckpt['mean_true_objective']):.2f}"
        if np.isclose(float(best_ckpt["mean_true_objective"]), best_score):
            best_value = f"\\textbf{{{best_value}}}"
        if np.isclose(float(final_ckpt["mean_true_objective"]), best_final_score):
            final_value = f"\\textbf{{{final_value}}}"
        lines.append(
            "    "
            + " & ".join(
                [
                    VARIANT_LABELS[key],
                    format_step_millions(int(best_ckpt["train_env_steps"])),
                    best_value,
                    final_value,
                ]
            )
            + " \\\\"
        )

    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def create_figure_symlink(output_dir: Path, paper_figure_dir: Path, figure_name: str) -> None:
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        target = (output_dir / f"{figure_name}.{suffix}").resolve()
        link_path = paper_figure_dir / f"{figure_name}.{suffix}"
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(target)


def remove_stale_combined_outputs(output_dir: Path, paper_figure_dir: Path, figure_name: str) -> None:
    for parent in (output_dir, paper_figure_dir):
        for suffix in ("pdf", "png"):
            path = parent / f"{figure_name}.{suffix}"
            if path.is_symlink() or path.exists():
                path.unlink()


def retained_progress_limits(summary: dict[str, Any]) -> tuple[float, float]:
    values = [
        float(checkpoint["mean_true_objective"])
        for variant in summary["variants"]
        for checkpoint in variant["checkpoints"]
    ]
    lower = max(0.0, min(values) - 0.6)
    upper = max(values) + 0.6
    return lower, upper


def plot_summary(summary: dict[str, Any], output_dir: Path, figure_name: str, paper_figure_dir: Path) -> None:
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

    y_min, y_max = retained_progress_limits(summary)
    variants = [variant for key in VARIANT_ORDER for variant in summary["variants"] if variant["key"] == key]
    output_dir.mkdir(parents=True, exist_ok=True)

    best_name = f"{figure_name}_best"
    retained_name = f"{figure_name}_retained"

    fig_best, ax_best = plt.subplots(figsize=(4.8, 4.1), dpi=300)
    ax_best.set_xlabel("Step of best retained checkpoint (millions)")
    ax_best.set_ylabel("Common-target progress (checkpoints)")
    ax_best.set_xlim(0.0, 130.0)
    ax_best.set_ylim(y_min, y_max)
    ax_best.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_best.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_best.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.8, alpha=0.7)

    for variant in variants:
        style = VARIANT_STYLES[variant["key"]]
        best_ckpt = best_checkpoint_payload(variant)
        x_best = int(best_ckpt["train_env_steps"]) / 1_000_000.0
        y_best = float(best_ckpt["mean_true_objective"])
        ax_best.hlines(
            y_best,
            xmin=0.0,
            xmax=x_best,
            colors=style["color"],
            linestyles=":",
            linewidth=1.5,
            alpha=0.9,
            zorder=1,
        )
        ax_best.vlines(
            x_best,
            ymin=y_min,
            ymax=y_best,
            colors=style["color"],
            linestyles=":",
            linewidth=1.5,
            alpha=0.9,
            zorder=1,
        )
        ax_best.scatter(
            [x_best],
            [y_best],
            color=style["color"],
            marker=style["marker"],
            s=115,
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )

    fig_best.tight_layout()
    fig_best.savefig(output_dir / f"{best_name}.pdf", bbox_inches="tight")
    fig_best.savefig(output_dir / f"{best_name}.png", bbox_inches="tight")
    plt.close(fig_best)

    fig_retained, ax_retained = plt.subplots(figsize=(4.8, 4.1), dpi=300)
    y_positions = np.arange(len(variants))
    for y_idx, variant in zip(y_positions, variants):
        style = VARIANT_STYLES[variant["key"]]
        best_ckpt = best_checkpoint_payload(variant)
        final_ckpt = final_checkpoint_payload(variant)
        x_best = float(best_ckpt["mean_true_objective"])
        x_final = float(final_ckpt["mean_true_objective"])

        ax_retained.plot([x_best, x_final], [y_idx, y_idx], color=style["color"], linewidth=2.2, alpha=0.9, zorder=1)
        ax_retained.scatter(
            [x_best],
            [y_idx],
            s=115,
            marker=style["marker"],
            facecolors="white",
            edgecolors=style["color"],
            linewidths=2.0,
            zorder=3,
        )
        ax_retained.scatter(
            [x_final],
            [y_idx],
            s=85,
            marker=style["marker"],
            facecolors=style["color"],
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )

    ax_retained.set_xlabel("Common-target progress (checkpoints)")
    ax_retained.set_xlim(y_min, y_max)
    ax_retained.set_yticks(y_positions, [VARIANT_SHORT_LABELS[variant["key"]] for variant in variants])
    ax_retained.invert_yaxis()
    ax_retained.grid(True, axis="x", color="#d9d9d9", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_retained.tick_params(axis="y", length=0, labelsize=11)
    fig_retained.tight_layout()
    fig_retained.savefig(output_dir / f"{retained_name}.pdf", bbox_inches="tight")
    fig_retained.savefig(output_dir / f"{retained_name}.png", bbox_inches="tight")
    plt.close(fig_retained)

    create_figure_symlink(output_dir, paper_figure_dir, best_name)
    create_figure_symlink(output_dir, paper_figure_dir, retained_name)
    remove_stale_combined_outputs(output_dir, paper_figure_dir, figure_name)


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_table(path: Path, table_fragment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table_fragment, encoding="utf-8")


def main() -> None:
    args = parse_args_cli()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary_json)
    table_path = Path(args.table_output)
    paper_figure_dir = Path(args.paper_figure_dir)

    if args.skip_eval:
        summary = load_summary(summary_path)
    else:
        variants = discover_variant_runs(run_root)
        selected_keys = {item.strip() for item in args.variants.split(",") if item.strip()}
        variants = filter_variants(variants, selected_keys, int(args.max_checkpoints_per_variant))
        summary = evaluate_variants(
            variants=variants,
            target_env=args.target_env,
            target_max_score=float(args.target_max_score),
            episodes=int(args.episodes),
            device=args.device,
            deterministic=bool(args.eval_deterministic),
        )
        write_summary(summary_path, summary)

    write_table(table_path, build_small_table(summary))
    plot_summary(summary, output_dir, args.figure_name, paper_figure_dir)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote table: {table_path}")
    print(f"Wrote figures: {output_dir / (args.figure_name + '_best.pdf')}, {output_dir / (args.figure_name + '_retained.pdf')}")


if __name__ == "__main__":
    main()
