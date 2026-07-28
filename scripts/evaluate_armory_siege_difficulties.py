from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

from comrad.train import (
    configure_batch_env_and_agents,
    parse_args as parse_comrad_args,
    register_model_factory,
    register_vizdoom_components,
)
from comrad.utils.recording_actions import reshape_deterministic_actions

LEVEL_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def build_grid(registry: list[dict]):
    grid: dict[tuple[str, str], dict] = {}
    spation_set: set[str] = set()
    mech_set: set[str] = set()
    for entry in registry:
        axes = entry.get("axes", {})
        spation = axes["spatial_level"]
        mech = axes["param_level"]
        grid[(spation, mech)] = entry
        spation_set.add(spation)
        mech_set.add(mech)
    spation_labels = sorted(spation_set, key=lambda x: LEVEL_ORDER.get(x, 99))
    mech_labels = sorted(mech_set, key=lambda x: LEVEL_ORDER.get(x, 99))
    return grid, spation_labels, mech_labels


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QMIX on 9 armory_siege difficulty variants (3x3 grid).")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="Path to checkpoint_p0 directory")
    parser.add_argument("--train_dir", default=None, type=str, help="Parent of experiment dir (inferred if omitted)")
    parser.add_argument("--experiment", default=None, type=str, help="Experiment name (inferred if omitted)")
    parser.add_argument("--batch_dir", default=str(REPO_ROOT / "comrad" / "scenarios" / "batch_armory_siege_w2_eval"),
                        type=str, help="Batch dir with batch_registry.json and .wad files")
    parser.add_argument("--episodes", default=10, type=int, help="Episodes per WAD variant")
    parser.add_argument("--max_frames", default=100000, type=int, help="Max frames per eval run")
    parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "gpu"], help="Device")
    parser.add_argument("--output", default=None, type=str, help="Save markdown table to file")
    parser.add_argument("--checkpoint_file", default=None, type=str, help="Exact .pth path (uses best if omitted)")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="Deterministic action selection")
    return parser.parse_args(argv)


def resolve_checkpoint(args) -> tuple[str, str, str]:
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {checkpoint_dir}")
    if args.checkpoint_file:
        ckpt_path = Path(args.checkpoint_file).resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        train_dir = args.train_dir or str(checkpoint_dir.parent.parent)
        experiment = args.experiment or checkpoint_dir.parent.name
        return str(ckpt_path), train_dir, experiment
    best = sorted(checkpoint_dir.glob("best_*.pth"))
    ckpt_path = str(best[-1]) if best else str(sorted(checkpoint_dir.glob("checkpoint_*.pth"))[-1])
    train_dir = args.train_dir or str(checkpoint_dir.parent.parent)
    experiment = args.experiment or checkpoint_dir.parent.name
    return ckpt_path, train_dir, experiment


def singleton_batch_dir(batch_dir: Path, entry: dict) -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory(prefix="comrad_single_wad_")
    wad_path = str((batch_dir / entry["filename"]).resolve())
    payload = [{"id": entry["id"], "filename": wad_path, "seed": entry.get("config", {}).get("seed", 0)}]
    Path(temp_dir.name, "batch_registry.json").write_text(json.dumps(payload), encoding="utf-8")
    return temp_dir


def build_eval_cfg(train_dir: str, experiment: str, device: str, deterministic: bool):
    argv = [
        "--algo=QMIX", "--mixer=qmix",
        f"--train_dir={train_dir}", f"--experiment={experiment}",
        "--env=armory_siege", f"--device={device}",
        "--with_wandb=False", "--wandb_record_every=0",
    ]
    if deterministic:
        argv.append("--eval_deterministic=True")
    cfg = parse_comrad_args(argv=argv, evaluation=True)
    cfg = load_from_checkpoint(cfg)
    cfg.env = "armory_siege"
    cfg.curriculum = "uniform"
    cfg.device = device
    cfg.eval_deterministic = deterministic
    cfg.policy_index = 0
    cfg.num_envs = 1
    cfg.num_workers = 1
    return cfg


def joint_true_objective(infos, rewards: torch.Tensor) -> float:
    if isinstance(infos, dict) and "true_objective" in infos:
        return float(infos["true_objective"])
    if isinstance(infos, (list, tuple)):
        values = [float(info["true_objective"]) for info in infos if isinstance(info, dict) and "true_objective" in info]
        if values:
            return float(np.mean(values))
    return float(torch.mean(rewards).item())


def qmix_actions_from_qvalues(actor_critic, action_logits):
    action_sizes = actor_critic.agent_net.action_sizes
    offset = 0
    head_actions = []
    for size in action_sizes:
        q = action_logits[:, offset:offset + size]
        head_actions.append(torch.argmax(q, dim=-1, keepdim=True))
        offset += size
    return torch.cat(head_actions, dim=-1)


def evaluate_wad(entry: dict, batch_dir: Path, cfg, checkpoint_path: str, episodes: int, max_frames: int) -> float:
    with singleton_batch_dir(batch_dir, entry) as temp_batch:
        cfg.wad_batch = temp_batch
        configure_batch_env_and_agents(cfg)
        register_model_factory(cfg)
        torch_device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
        env = make_eval_env(cfg, render_mode=None)
        try:
            env_info = extract_env_info(env, cfg)
            actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
            actor_critic.eval()
            actor_critic.model_to_device(torch_device)
            checkpoint_dict = Learner.load_checkpoint([checkpoint_path], torch_device)
            if checkpoint_dict is None:
                raise RuntimeError(f"Could not load checkpoint: {checkpoint_path}")
            actor_critic.load_state_dict(checkpoint_dict["model"])
            objectives: list[float] = []
            obs, _infos = env.reset()
            action_mask = obs.pop("action_mask").to(torch_device) if "action_mask" in obs else None
            rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=torch_device)
            num_frames = 0
            with torch.no_grad():
                while len(objectives) < episodes and num_frames < max_frames:
                    normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                    policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)
                    actions = policy_outputs["actions"]
                    if getattr(actor_critic, "last_action_distribution", None) is not None:
                        if cfg.eval_deterministic:
                            actions = reshape_deterministic_actions(
                                actions, argmax_actions(actor_critic.action_distribution()))
                    else:
                        actions = qmix_actions_from_qvalues(actor_critic, policy_outputs["action_logits"])
                    if actions.ndim == 1:
                        actions = unsqueeze_tensor(actions, dim=-1)
                    actions = preprocess_actions(env_info, actions)
                    rnn_states = policy_outputs["new_rnn_states"]
                    obs, rewards, terminated, truncated, infos = env.step(actions)
                    action_mask = obs.pop("action_mask").to(torch_device) if "action_mask" in obs else None
                    dones = make_dones(terminated, truncated).cpu().numpy()
                    num_frames += 1
                    for agent_idx, done in enumerate(dones):
                        if done:
                            rnn_states[agent_idx].zero_()
                    if np.all(dones):
                        objectives.append(joint_true_objective(infos, rewards))
            return float(np.mean(objectives)) if objectives else 0.0
        finally:
            env.close()


def render_markdown_table(scores: dict[tuple[str, str], float],
                          spation_labels: list[str], mech_labels: list[str],
                          num_episodes: int) -> str:
    lines = [
        f"### Armory Siege - 3x3 Difficulty Grid (mean true_objective over {num_episodes} eps)",
        "",
        "| Spation \\ Mechanical | " + " | ".join(m.capitalize() for m in mech_labels) + " |",
        "|" + "|".join(["---"] * (len(mech_labels) + 1)) + "|",
    ]
    for sp in spation_labels:
        cells = [sp.capitalize()]
        for mech in mech_labels:
            val = scores.get((sp, mech))
            cells.append(f"{val:.2f}" if val is not None else "-")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    register_vizdoom_components()

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        raise NotADirectoryError(f"Batch directory not found: {batch_dir}")
    registry = json.loads((batch_dir / "batch_registry.json").read_text(encoding="utf-8"))
    grid, spation_labels, mech_labels = build_grid(registry)
    checkpoint_path, train_dir, experiment = resolve_checkpoint(args)
    logging.info("Checkpoint: %s", checkpoint_path)
    logging.info("Spation levels: %s  Mechanical levels: %s", spation_labels, mech_labels)

    base_cfg = build_eval_cfg(train_dir, experiment, args.device, bool(args.deterministic))
    results: dict[tuple[str, str], float] = {}
    for sp in spation_labels:
        for mech in mech_labels:
            entry = grid[(sp, mech)]
            if not (batch_dir / entry["filename"]).is_file():
                logging.warning("SKIP: %s not found", entry["filename"])
                continue
            logging.info("Evaluating %s  spation=%s  mechanical=%s", entry["id"], sp, mech)
            cfg = copy.deepcopy(base_cfg)
            mean_obj = evaluate_wad(entry, batch_dir, cfg, checkpoint_path, args.episodes, args.max_frames)
            results[(sp, mech)] = mean_obj
            logging.info("mean=%.2f", mean_obj)

    table = render_markdown_table(results, spation_labels, mech_labels, args.episodes)
    sys.stdout.write(table + "\n")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
