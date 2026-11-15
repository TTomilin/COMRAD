"""
Reviewer C2K6 frozen-partner baseline: evaluate one frozen two-agent checkpoint on the 5 held-out WADs
(`HELDOUT_SEEDS`) with both agents learned, and with one seat replaced by a no-op or a uniform-random partner.

All passes run sequentially inside one live 2-agent `MultiAgentEnv` per WAD. The frozen model always runs on both
agents' observations (so MAPPO's 2-agent critic reshape and QMIX's per-agent batch stay intact); for intervention
passes the partner row of the `[num_agents, num_heads]` action tensor is overwritten right before
`preprocess_actions`. Every pass starts from an explicit `env.reset()` with zeroed RNN state and goal counter, and
uses a deterministic RNG seeded by (wad_seed, pass index) for the random partner and for action sampling.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

from comrad.utils.recording_actions import reshape_deterministic_actions
from comrad.wrappers.scripted_partner import zero_action
from experiments.iclr2027_rebuttal.rebuttal_kd6j_eval import sample_uniform_random_action
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.model.model_utils import get_rnn_size
from scripts.heldout_wad_evaluation import (
    GOAL_TRACKED_SCENARIOS,
    HELDOUT_SEEDS,
    VALUE_BASED_ALGOS,
    GoalRelocationCounter,
    PolicySpec,
    WadManifestEntry,
    action_head_sizes,
    fresh_episode_infos,
    greedy_q_actions,
    joint_true_objective,
    sha256_file,
)

# (pass name, partner policy, partner seat). Seat k replaced => agent 1-k is the frozen learner.
PASS_SPECS: tuple[tuple[str, str, int | None], ...] = (
    ("both_learned", "learned", None),
    ("noop_seat1", "noop", 1),
    ("noop_seat0", "noop", 0),
    ("random_seat1", "random", 1),
    ("random_seat0", "random", 0),
)
PARTNER_POLICIES = ("learned", "noop", "random")
# ViZDoom occasionally segfaults mid-episode (seen on Stag Hunt with no-op/random partners). The interrupted episode
# is discarded and replaced by a fresh one; a pass only fails after this many crashed episodes.
MAX_CRASHED_EPISODES_PER_PASS = 10
# Bump whenever the evaluation semantics change so cached JSONs from older code are re-run.
PROTOCOL_VERSION = "c2k6-frozen-partner-v2"


def build_partner_action_tensor(
    partner_policy: str,
    action_space: gym.Space,
    rng: np.random.Generator,
    partner_action_mask: np.ndarray | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """1D tensor of shape [num_heads] that replaces `actions[partner_idx]` before `preprocess_actions`."""
    if partner_policy == "noop":
        raw = zero_action(action_space)
    elif partner_policy == "random":
        raw = sample_uniform_random_action(action_space, rng, partner_action_mask)
    else:
        raise ValueError(f"Unsupported partner policy {partner_policy!r}")
    arr = np.asarray(raw, dtype=np.int64).reshape(-1)
    return torch.as_tensor(arr, dtype=dtype, device=device)


class HiddenResetError(RuntimeError):
    """The multi-agent wrapper recovered from a crash by resetting the game underneath the episode."""


def hidden_reset_count(infos: Any) -> int:
    if not isinstance(infos, (list, tuple)):
        return 0
    return max((int(i.get("_hidden_reset_count", 0)) for i in infos if isinstance(i, dict)), default=0)


def pass_rng_seed(wad_seed: int, pass_index: int) -> int:
    return int(wad_seed) * 100 + int(pass_index)


def _policy_actions(
    actor_critic: Any,
    obs_step: dict[str, torch.Tensor],
    rnn_states: torch.Tensor,
    action_mask: torch.Tensor | None,
    value_based: bool,
    head_sizes: list[int],
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen-policy action rule identical to heldout_wad_evaluation.evaluate_policy_on_wad."""
    normalized_obs = prepare_and_normalize_obs(actor_critic, obs_step)
    policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask, sample_actions=not value_based)
    if value_based:
        actions = greedy_q_actions(policy_outputs["action_logits"], head_sizes)
    else:
        actions = policy_outputs["actions"]
        if deterministic:
            actions = reshape_deterministic_actions(actions, argmax_actions(actor_critic.action_distribution()))
    if actions.ndim == 1:
        actions = unsqueeze_tensor(actions, dim=-1)
    return actions, policy_outputs["new_rnn_states"]


def run_pass(
    env: Any,
    actor_critic: Any,
    env_info: Any,
    cfg: Any,
    policy: PolicySpec,
    entry: WadManifestEntry,
    episodes: int,
    torch_device: torch.device,
    *,
    pass_name: str,
    partner_policy: str,
    partner_idx: int | None,
    rng_seed: int,
    preprocess: Callable[[Any, torch.Tensor], Any] = preprocess_actions,
    max_crashed_episodes: int = MAX_CRASHED_EPISODES_PER_PASS,
    crash_log: list[dict[str, Any]] | None = None,
    obs_transform: Callable[[dict[str, torch.Tensor], int], dict[str, torch.Tensor]] | None = None,
    record_extras: Callable[[Any], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """`obs_transform(obs, episode_step)` edits what the policy sees (e.g. a blind intervention); `record_extras(infos)`
    adds fields from the terminal infos to each episode record."""
    if partner_policy not in PARTNER_POLICIES:
        raise ValueError(f"Unsupported partner policy {partner_policy!r}")
    num_agents = int(env.num_agents)
    if partner_policy != "learned":
        if num_agents != 2 or partner_idx not in (0, 1):
            raise ValueError(f"Frozen-partner passes need 2 agents and partner_idx in {{0, 1}}, got {partner_idx}")
    learner_idx = None if partner_idx is None else 1 - int(partner_idx)

    rng = np.random.default_rng(rng_seed)
    torch.manual_seed(rng_seed)
    value_based = str(cfg.algo).upper() in VALUE_BASED_ALGOS
    head_sizes = action_head_sizes(env_info.action_space) if value_based else []
    deterministic = bool(getattr(cfg, "eval_deterministic", False))
    track_goals = any(policy.scenario.startswith(prefix) for prefix in GOAL_TRACKED_SCENARIOS)

    # Explicit reset: nothing from the previous pass (partial episode, RNN memory, goal position) can leak in.
    obs, infos = env.reset()
    goals = GoalRelocationCounter()
    goals.observe(infos)
    rnn_states = torch.zeros([num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=torch_device)

    records: list[dict[str, Any]] = []
    crashed = 0
    episode_step = 0
    with torch.no_grad():
        while len(records) < episodes:
            obs_step = dict(obs)
            action_mask = obs_step.pop("action_mask").to(torch_device) if "action_mask" in obs_step else None
            if obs_transform is not None:
                obs_step = obs_transform(obs_step, episode_step)
            actions, rnn_states = _policy_actions(
                actor_critic, obs_step, rnn_states, action_mask, value_based, head_sizes, deterministic
            )
            if partner_policy != "learned":
                partner_mask = None if action_mask is None else action_mask[partner_idx].detach().cpu().numpy()
                actions = actions.clone()
                actions[partner_idx] = build_partner_action_tensor(
                    partner_policy, env_info.action_space, rng, partner_mask, actions.dtype, actions.device
                )
            obs, rewards, terminated, truncated, infos = env.step(preprocess(env_info, actions))
            episode_step += 1
            if hidden_reset_count(infos) > 0:
                # The game was rebuilt underneath the episode: discard the partial episode and restart cleanly so
                # RNN memory and the goal counter never continue into the fresh game.
                crashed += 1
                if crash_log is not None:
                    crash_log.append(
                        {"wad_seed": entry.seed, "pass": pass_name, "episode_index": len(records),
                         "episode_step": episode_step}
                    )
                print(f"[warn] {pass_name}: hidden crash-recovery reset at episode {len(records)}; "
                      f"discarding it ({crashed}/{max_crashed_episodes})", flush=True)
                if crashed >= max_crashed_episodes:
                    raise HiddenResetError(f"{pass_name}: {crashed} crashed episodes on wad {entry.seed}")
                obs, infos = env.reset()
                goals.reset()
                goals.observe(infos)
                rnn_states.zero_()
                episode_step = 0
                continue
            dones = make_dones(terminated, truncated).cpu().numpy()
            for agent_idx, done in enumerate(dones):
                if done:
                    rnn_states[agent_idx].zero_()
            if np.all(dones):
                episode_step = 0
                goals.observe(infos)
                true_obj = float(goals.count) if track_goals else joint_true_objective(infos)
                record: dict[str, Any] = {
                    "scenario": policy.scenario,
                    "algorithm": policy.algorithm,
                    "train_seed": policy.train_seed,
                    "pass": pass_name,
                    "partner_policy": partner_policy,
                    "partner_idx": partner_idx,
                    "learner_idx": learner_idx,
                    "wad_id": entry.wad_id,
                    "wad_seed": entry.seed,
                    "episode_index": len(records),
                    "true_objective": true_obj,
                    **(record_extras(infos) if record_extras is not None else {}),
                }
                if track_goals:
                    record["goals_reached"] = int(goals.count)
                records.append(record)
                goals.reset()
                # the env has auto-reset; seed the counter from the new episode's first frame
                goals.observe(fresh_episode_infos(infos))
            else:
                goals.observe(infos)
    return records


def pass_episodes(pass_name: str, both_eps: int, seat_eps: int) -> int:
    return int(both_eps) if pass_name == "both_learned" else int(seat_eps)


def summarize_passes(pass_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    passes: dict[str, Any] = {}
    for name, recs in pass_records.items():
        scores = [float(r["true_objective"]) for r in recs]
        passes[name] = {
            "episodes": len(scores),
            "mean_true_objective": float(np.mean(scores)) if scores else float("nan"),
            "records": recs,
        }

    def mean_of(name: str) -> float:
        return float(passes[name]["mean_true_objective"])

    aggregated = {
        "both_learned_raw": mean_of("both_learned"),
        "partner_noop_raw": 0.5 * (mean_of("noop_seat0") + mean_of("noop_seat1")),
        "partner_random_raw": 0.5 * (mean_of("random_seat0") + mean_of("random_seat1")),
    }
    return {"passes": passes, "aggregated": aggregated}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--algorithm", required=True)
    p.add_argument("--train-seed", type=int, required=True)
    p.add_argument("--batch-root", default="results/heldout_wad_batches")
    p.add_argument("--output-json", required=True)
    p.add_argument("--both-episodes-per-wad", type=int, default=20)
    p.add_argument("--seat-episodes-per-wad", type=int, default=10)
    p.add_argument("--target-steps", type=int, default=100_000_000)
    p.add_argument("--passes", default=",".join(name for name, _, _ in PASS_SPECS))
    p.add_argument("--wad-seeds", default=",".join(map(str, HELDOUT_SEEDS)))
    p.add_argument("--device", default="gpu")
    return p.parse_args()


def main() -> None:
    from comrad.train import register_vizdoom_components
    from experiments.iclr2027_rebuttal.rebuttal_kd6j_eval import ensure_heldout_batch
    from sample_factory.algo.learning.learner import Learner
    from sample_factory.algo.utils.env_info import extract_env_info
    from sample_factory.enjoy import make_env as make_eval_env
    from sample_factory.model.actor_critic import create_actor_critic
    from scripts.heldout_wad_evaluation import build_eval_cfg, select_checkpoint, singleton_batch_dir

    args = parse_args()
    # Several evals share one MIG slice (9 CPUs); keep each to one intra-op thread.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    selected = [s.strip() for s in args.passes.split(",") if s.strip()]
    specs = [spec for spec in PASS_SPECS if spec[0] in selected]
    if len(specs) != len(selected):
        raise ValueError(f"Unknown pass in {selected}; choices: {[s[0] for s in PASS_SPECS]}")
    wad_seeds = tuple(int(s) for s in args.wad_seeds.split(",") if s.strip())

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
    entries = ensure_heldout_batch(args.scenario, batch_root, wad_seeds)
    if tuple(e.seed for e in entries) != wad_seeds:
        raise RuntimeError(f"Held-out batch seeds {[e.seed for e in entries]} != {wad_seeds}")
    register_vizdoom_components()

    torch_device = torch.device("cpu" if args.device == "cpu" else "cuda")
    checkpoint_dict = Learner.load_checkpoint([policy.checkpoint], torch_device)
    if checkpoint_dict is None:
        raise RuntimeError(f"Could not load checkpoint: {policy.checkpoint}")

    pass_records: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in specs}
    crashed_episodes: list[dict[str, Any]] = []
    wad_sha256 = {str(e.seed): e.sha256 for e in entries}
    for entry in entries:
        with singleton_batch_dir(batch_root / args.scenario, entry) as temp_batch:
            cfg = build_eval_cfg(policy, Path(temp_batch), args.device, deterministic=False)
            env = make_eval_env(cfg, render_mode=None)
            try:
                env_info = extract_env_info(env, cfg)
                actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
                actor_critic.eval()
                actor_critic.model_to_device(torch_device)
                actor_critic.load_state_dict(checkpoint_dict["model"])
                for pass_index, (name, partner_policy, partner_idx) in enumerate(PASS_SPECS):
                    if name not in pass_records:
                        continue
                    # A HiddenResetError (crash budget exhausted) is fatal: no whole-pass rerun that would reset the
                    # budget, re-seed the RNG streams and throw away completed episodes.
                    recs = run_pass(
                        env,
                        actor_critic,
                        env_info,
                        cfg,
                        policy,
                        entry,
                        pass_episodes(name, args.both_episodes_per_wad, args.seat_episodes_per_wad),
                        torch_device,
                        pass_name=name,
                        partner_policy=partner_policy,
                        partner_idx=partner_idx,
                        rng_seed=pass_rng_seed(entry.seed, pass_index),
                        crash_log=crashed_episodes,
                    )
                    pass_records[name].extend(recs)
                    mean = float(np.mean([r["true_objective"] for r in recs]))
                    print(f"[{args.scenario} {args.algorithm} s{args.train_seed}] wad={entry.seed} {name}: {mean:.3f}")
            finally:
                env.close()

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "scenario": args.scenario,
        "algorithm": args.algorithm,
        "train_seed": int(args.train_seed),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_steps": int(steps),
        "checkpoint_sha256": policy.checkpoint_sha256,
        "config_sha256": policy.config_sha256,
        "heldout_seeds": list(wad_seeds),
        "wad_sha256": wad_sha256,
        # Whole-pass reruns no longer exist (kept empty for schema compatibility with earlier v2 JSONs).
        "hidden_resets": [],
        # Episodes interrupted by a ViZDoom crash; each was discarded and replaced by a fresh episode.
        "crashed_episodes": crashed_episodes,
        # Simulator randomness is not re-seeded per pass: passes run in the fixed PASS_SPECS order in one live game.
        "pass_order": [name for name, _, _ in PASS_SPECS if name in pass_records],
        "both_episodes_per_wad": int(args.both_episodes_per_wad),
        "seat_episodes_per_wad": int(args.seat_episodes_per_wad),
        "eval_deterministic": False,
    }
    if len(pass_records) == len(PASS_SPECS):
        result.update(summarize_passes(pass_records))
    else:
        result["passes"] = summarize_passes({**{n: [] for n, _, _ in PASS_SPECS}, **pass_records})["passes"]

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp{os.getpid()}")
    tmp_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)  # atomic: a half-written JSON is never mistaken for a finished run
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
