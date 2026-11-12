"""
Evaluate a trained COMRAD checkpoint directory on the 5 held-out WADs (`HELDOUT_SEEDS`)
and optionally also run Partner-Occlusion (`--partner_occlusion_mode=both`) and Zero-Shot Blind
(`--obs_modality=blind`) interventions inside the same live `MultiAgentEnv` per WAD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from comrad.train import (  # noqa: E402
    configure_batch_env_and_agents,
    register_model_factory,
    register_vizdoom_components,
)
from comrad.utils.recording_actions import reshape_deterministic_actions  # noqa: E402
from heldout_wad_evaluation import (  # noqa: E402
    GOAL_TRACKED_SCENARIOS,
    HELDOUT_SEEDS,
    VALUE_BASED_ALGOS,
    GoalRelocationCounter,
    PolicySpec,
    WadManifestEntry,
    action_head_sizes,
    build_eval_cfg,
    fresh_episode_infos,
    greedy_q_actions,
    joint_true_objective,
    select_checkpoint,
    sha256_file,
    singleton_batch_dir,
)
from rebuttal_kd6j_eval import (  # noqa: E402
    _extract_visibility_from_infos,
    ensure_heldout_batch,
)
from sample_factory.algo.learning.learner import Learner  # noqa: E402
from sample_factory.algo.sampling.batched_sampling import preprocess_actions  # noqa: E402
from sample_factory.algo.utils.action_distributions import argmax_actions  # noqa: E402
from sample_factory.algo.utils.env_info import extract_env_info  # noqa: E402
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs  # noqa: E402
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor  # noqa: E402
from sample_factory.enjoy import make_env as make_eval_env  # noqa: E402
from sample_factory.model.actor_critic import create_actor_critic  # noqa: E402
from sample_factory.model.model_utils import get_rnn_size  # noqa: E402


def _find_visibility_wrapper(env: Any) -> Any:
    cur = env
    while cur is not None:
        if "partner_occlusion_mode" in getattr(cur, "__dict__", {}):
            return cur
        cur = getattr(cur, "env", None)
    return None


def _run_episodes_in_live_env(
    env: Any,
    actor_critic: Any,
    env_info: Any,
    cfg: Any,
    policy: PolicySpec,
    entry: WadManifestEntry,
    episodes: int,
    torch_device: torch.device,
    *,
    obs: dict[str, torch.Tensor],
    infos: Any,
    mode: str,
    partner_occlusion_mode: str = "none",
    blind_override: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], Any]:
    vis_wrapper = _find_visibility_wrapper(env)
    if vis_wrapper is not None:
        vis_wrapper.partner_occlusion_mode = partner_occlusion_mode

    value_based = str(cfg.algo).upper() in VALUE_BASED_ALGOS
    head_sizes = action_head_sizes(env_info.action_space) if value_based else []
    track_goals = any(policy.scenario.startswith(prefix) for prefix in GOAL_TRACKED_SCENARIOS)
    goals = GoalRelocationCounter()
    # When a previous mode just ended, `infos` belong to its terminal step (old episode's goal) while the env has
    # already auto-reset; seed from the new episode's reset_info instead, else a fake relocation is counted.
    goals.observe(fresh_episode_infos(infos))

    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=torch_device)
    episode_records: list[dict[str, Any]] = []
    step_in_ep = 0
    horizon = 4500

    with torch.no_grad():
        while len(episode_records) < episodes:
            obs_step = dict(obs)
            action_mask = obs_step.pop("action_mask").to(torch_device) if "action_mask" in obs_step else None
            if blind_override and "obs" in obs_step:
                zeroed = torch.zeros_like(obs_step["obs"])
                progress = min(1.0, float(step_in_ep) / float(horizon))
                if zeroed.numel() > 0:
                    if zeroed.dtype == torch.uint8:
                        zeroed.view(zeroed.shape[0], -1)[:, 0] = int(round(progress * 255.0))
                    else:
                        zeroed.view(zeroed.shape[0], -1)[:, 0] = float(progress)
                obs_step["obs"] = zeroed

            normalized_obs = prepare_and_normalize_obs(actor_critic, obs_step)
            policy_outputs = actor_critic(
                normalized_obs, rnn_states, action_mask=action_mask, sample_actions=not value_based
            )
            if value_based:
                actions = greedy_q_actions(policy_outputs["action_logits"], head_sizes)
            else:
                actions = policy_outputs["actions"]
                if cfg.eval_deterministic:
                    actions = reshape_deterministic_actions(
                        actions, argmax_actions(actor_critic.action_distribution())
                    )
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)
            rnn_states = policy_outputs["new_rnn_states"]
            obs, rewards, terminated, truncated, infos = env.step(actions)
            step_in_ep += 1
            dones = make_dones(terminated, truncated).cpu().numpy()
            for agent_idx, done in enumerate(dones):
                if done:
                    rnn_states[agent_idx].zero_()
            if np.all(dones):
                goals.observe(infos)
                vis_stats = _extract_visibility_from_infos(infos, env)
                true_obj = float(goals.count) if track_goals else joint_true_objective(infos, rewards)
                record: dict[str, Any] = {
                    "scenario": policy.scenario,
                    "algorithm": policy.algorithm,
                    "mode": mode,
                    "train_seed": policy.train_seed,
                    "checkpoint": policy.checkpoint,
                    "wad_id": entry.wad_id,
                    "wad_seed": entry.seed,
                    "episode_index": len(episode_records),
                    "true_objective": true_obj,
                    **vis_stats,
                }
                if track_goals:
                    record["goals_reached"] = int(goals.count)
                episode_records.append(record)
                goals.reset()
                goals.observe(fresh_episode_infos(infos))
                step_in_ep = 0
            else:
                goals.observe(infos)

    return episode_records, obs, infos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--algorithm", required=True)
    p.add_argument("--cell-tag", required=True)
    p.add_argument("--train-seed", type=int, required=True)
    p.add_argument("--batch-root", default="results/heldout_wad_batches")
    p.add_argument("--output-json", required=True)
    p.add_argument("--episodes-per-wad", type=int, default=2)
    p.add_argument("--target-steps", type=int, default=100_000_000)
    p.add_argument("--device", default="gpu")
    p.add_argument("--run-interventions", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Several evals share one MIG slice (9 CPUs); keep each to one intra-op thread.
    torch.set_num_threads(1)
    run_dir = Path(args.run_dir).resolve()
    checkpoint, steps = select_checkpoint(run_dir, int(args.target_steps))
    policy = PolicySpec(
        scenario=args.scenario,
        algorithm=args.algorithm,
        run_dir=str(run_dir),
        checkpoint=str(checkpoint),
        checkpoint_steps=int(steps),
        config_sha256=sha256_file(run_dir / "config.json"),
        checkpoint_sha256=sha256_file(checkpoint),
        train_seed=int(args.train_seed),
    )

    batch_root = Path(args.batch_root)
    entries = ensure_heldout_batch(args.scenario, batch_root, HELDOUT_SEEDS)
    register_vizdoom_components()

    normal_records: list[dict[str, Any]] = []
    occ_records: list[dict[str, Any]] = []
    blind_records: list[dict[str, Any]] = []

    torch_device = torch.device("cpu" if args.device == "cpu" else "cuda")
    checkpoint_dict = Learner.load_checkpoint([policy.checkpoint], torch_device)
    if checkpoint_dict is None:
        raise RuntimeError(f"Could not load checkpoint: {policy.checkpoint}")

    global_state_version = None
    for entry in entries:
        with singleton_batch_dir(batch_root / args.scenario, entry) as temp_batch:
            cfg = build_eval_cfg(policy, Path(temp_batch), args.device, deterministic=False)
            if getattr(cfg, "full_obs_mode", None) == "global_state":
                global_state_version = int(cfg.global_state_version)
            cfg.enable_labels_buffer = True
            cfg.partner_occlusion_mode = "none"
            configure_batch_env_and_agents(cfg)
            register_model_factory(cfg)

            env = make_eval_env(cfg, render_mode=None)
            try:
                env_info = extract_env_info(env, cfg)
                actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
                actor_critic.eval()
                actor_critic.model_to_device(torch_device)
                actor_critic.load_state_dict(checkpoint_dict["model"])

                obs, infos = env.reset()
                recs, obs, infos = _run_episodes_in_live_env(
                    env,
                    actor_critic,
                    env_info,
                    cfg,
                    policy,
                    entry,
                    int(args.episodes_per_wad),
                    torch_device,
                    obs=obs,
                    infos=infos,
                    mode=args.cell_tag,
                    partner_occlusion_mode="none",
                    blind_override=False,
                )
                normal_records.extend(recs)

                if args.run_interventions:
                    recs_occ, obs, infos = _run_episodes_in_live_env(
                        env,
                        actor_critic,
                        env_info,
                        cfg,
                        policy,
                        entry,
                        int(args.episodes_per_wad),
                        torch_device,
                        obs=obs,
                        infos=infos,
                        mode="partner_occlusion",
                        partner_occlusion_mode="both",
                        blind_override=False,
                    )
                    occ_records.extend(recs_occ)

                    recs_blind, obs, infos = _run_episodes_in_live_env(
                        env,
                        actor_critic,
                        env_info,
                        cfg,
                        policy,
                        entry,
                        int(args.episodes_per_wad),
                        torch_device,
                        obs=obs,
                        infos=infos,
                        mode="blind_eval",
                        partner_occlusion_mode="none",
                        blind_override=True,
                    )
                    blind_records.extend(recs_blind)
            finally:
                env.close()

    result: dict[str, Any] = {
        "scenario": args.scenario,
        "algorithm": args.algorithm,
        "cell_tag": args.cell_tag,
        "train_seed": int(args.train_seed),
        "checkpoint_steps": int(steps),
        "checkpoint": policy.checkpoint,
        # Provenance: results/ is not synced to the cluster, so a stale held-out batch would otherwise go unnoticed.
        "heldout_batch_registry_sha256": sha256_file(batch_root / args.scenario / "batch_registry.json"),
        "global_state_version": global_state_version,
        "normal_records": normal_records,
    }
    if args.run_interventions:
        result["partner_occlusion_records"] = occ_records
        result["blind_eval_records"] = blind_records

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path} ({len(normal_records)} normal eps)")


if __name__ == "__main__":
    main()
