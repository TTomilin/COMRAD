"""
Zero-shot evaluations for Reviewer kd6J rebuttal experiments on the five held-out DoomGen layouts
(`HELDOUT_SEEDS = (1009, 2027, 3037, 4051, 5099)`):

  1. `--mode=random`: Uniform random action policy across all 13 benchmark scenarios (5 seeds x 5 WADs x 20 eps)
     to establish the random lower bound `s_rand` for normalized Table 3, optionally collecting `labels_buffer`
     and 3D frustum visibility telemetry.
  2. `--mode=visibility`: Frozen 100M-step checkpoint evaluation with `enable_labels_buffer=True` to log
     `teammate_in_frustum_rate`, `teammate_visible_rate`, `teammate_obscured_rate`, `mutual_visibility_rate`,
     and `target_obscured_rate`.
  3. `--mode=partner_occlusion`: Frozen 100M-step checkpoint evaluation with `partner_occlusion_mode={sprite,cues,both}`
     masking out rendered `DoomPlayer` sprites and partner-transmitted visual/scalar cues.
  4. `--mode=blind_eval`: Frozen 100M-step checkpoint evaluation with `obs_modality=blind` (zeroed RGB + step counter
     `t/H`) to test open-loop temporal memorization vs. egocentric visual perception.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from comrad.envs.doom_params import default_doom_cfg  # noqa: E402
from comrad.train import (  # noqa: E402
    configure_batch_env_and_agents,
    register_model_factory,
    register_vizdoom_components,
)
from comrad.utils.recording_actions import reshape_deterministic_actions  # noqa: E402
from heldout_wad_evaluation import (  # noqa: E402
    DEFAULT_ARCHIVE_ROOT,
    DEFAULT_TARGET_STEPS,
    GOAL_TRACKED_SCENARIOS,
    HELDOUT_SEEDS,
    VALUE_BASED_ALGOS,
    GoalRelocationCounter,
    PolicySpec,
    WadManifestEntry,
    action_head_sizes,
    build_eval_cfg,
    discover_policy_specs,
    discover_seeded_policy_specs,
    fresh_episode_infos,
    generate_heldout_batch,
    greedy_q_actions,
    joint_true_objective,
    load_manifest,
    parse_csv_argument,
    singleton_batch_dir,
    write_raw_csv,
)
from sample_factory.algo.learning.learner import Learner  # noqa: E402
from sample_factory.algo.sampling.batched_sampling import preprocess_actions  # noqa: E402
from sample_factory.algo.utils.action_distributions import argmax_actions  # noqa: E402
from sample_factory.algo.utils.env_info import extract_env_info  # noqa: E402
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs  # noqa: E402
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor  # noqa: E402
from sample_factory.enjoy import make_env as make_eval_env  # noqa: E402
from sample_factory.envs.create_env import create_env  # noqa: E402
from sample_factory.model.actor_critic import create_actor_critic  # noqa: E402
from sample_factory.model.model_utils import get_rnn_size  # noqa: E402
from sample_factory.utils.attr_dict import AttrDict  # noqa: E402


ALL_13_SCENARIOS = (
    "stag_hunt_arena",
    "rhythm_sync_dense",
    "foraging_commons",
    "coop_puzzle",
    "platform_chain",
    "armory_siege",
    "coop_health_gathering",
    "lavapit",
    "smart_enemies",
    "dumb_enemies",
    "stealth_labyrinth",
    "ammo_carrier",
    "lava_maze",
)
DEFAULT_TRAIN_SEEDS = (42, 68, 81, 97, 154)
STUDENT_T_95 = {1: 0.0, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}

VISIBILITY_KEYS = (
    "teammate_in_frustum_rate",
    "teammate_visible_rate",
    "teammate_obscured_rate",
    "teammate_out_of_view_rate",
    "mutual_visibility_rate",
    "target_visible_rate",
    "target_obscured_rate",
    "mean_teammate_pixel_fraction",
    # Per-viewpoint rates (agent 0 observing agent 1 and vice versa), needed for asymmetric roles such as Lava Maze
    "agent0_teammate_in_frustum_rate",
    "agent0_teammate_visible_rate",
    "agent0_teammate_obscured_rate",
    "agent1_teammate_in_frustum_rate",
    "agent1_teammate_visible_rate",
    "agent1_teammate_obscured_rate",
)


def student_t_ci95_half_width(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    n = len(arr)
    if n <= 1:
        return 0.0
    t_crit = STUDENT_T_95.get(n, 1.96)
    return float(t_crit * np.std(arr, ddof=1) / math.sqrt(n))


def ensure_heldout_batch(scenario: str, batch_root: Path, seeds: tuple[int, ...] = HELDOUT_SEEDS) -> list[WadManifestEntry]:
    batch_dir = batch_root / scenario
    manifest_file = batch_dir / "batch_registry.json"
    if not manifest_file.is_file():
        entries = generate_heldout_batch(scenario, batch_dir, HELDOUT_SEEDS)
    else:
        entries = load_manifest(batch_dir)
    seed_set = set(int(s) for s in seeds)
    return [entry for entry in entries if int(entry.seed) in seed_set]


def sample_uniform_random_action(
    action_space: gym.Space,
    rng: np.random.Generator,
    action_mask: np.ndarray | None = None,
) -> Any:
    """Sample a uniform valid action per discrete/tuple head, respecting `action_mask` when present."""
    if isinstance(action_space, gym.spaces.Discrete):
        n = int(action_space.n)
        if action_mask is not None and len(action_mask) == n:
            valid = np.flatnonzero(action_mask > 0.0)
            if len(valid) > 0:
                return int(rng.choice(valid))
        return int(rng.integers(0, n))

    if isinstance(action_space, gym.spaces.Tuple):
        out = []
        offset = 0
        for subspace in action_space.spaces:
            n = int(subspace.n)
            if action_mask is not None and offset + n <= len(action_mask):
                sub_mask = action_mask[offset : offset + n]
                valid = np.flatnonzero(sub_mask > 0.0)
                if len(valid) > 0:
                    out.append(int(rng.choice(valid)))
                else:
                    out.append(int(rng.integers(0, n)))
            else:
                out.append(int(rng.integers(0, n)))
            offset += n
        return tuple(out)

    return action_space.sample()


def make_random_eval_env(
    scenario: str,
    temp_batch: str,
    enable_labels_buffer: bool = True,
    partner_occlusion_mode: str = "none",
):
    cfg = default_doom_cfg(env=scenario)
    cfg.algo = "MAPPO"
    cfg.num_agents = 2
    cfg.num_bots = 0
    cfg.num_humans = 0
    cfg.env_frameskip = 4
    cfg.wandb_record_every = 0
    cfg.wad_batch = temp_batch
    cfg.curriculum = "uniform"
    cfg.enable_labels_buffer = bool(enable_labels_buffer)
    cfg.partner_occlusion_mode = partner_occlusion_mode
    configure_batch_env_and_agents(cfg)
    return create_env(cfg.env, cfg=cfg, env_config=AttrDict({"worker_index": 0, "vector_index": 0, "env_id": 0}))


def _extract_visibility_from_infos(infos: Any, env: Any) -> dict[str, float]:
    if isinstance(infos, (list, tuple)) and infos:
        for info in infos:
            if isinstance(info, dict) and "episode_visibility_stats" in info:
                stats = info["episode_visibility_stats"]
                return {k: float(stats.get(k, 0.0)) for k in VISIBILITY_KEYS}
    # Fallback if episode was truncated early by max_steps_per_episode
    cur = env
    while cur is not None:
        if "current_stats" in getattr(cur, "__dict__", {}) and hasattr(cur.current_stats, "to_dict"):
            d = cur.current_stats.to_dict()
            return {k: float(d.get(k, 0.0)) for k in VISIBILITY_KEYS}
        cur = getattr(cur, "env", None)
    return {k: 0.0 for k in VISIBILITY_KEYS}


def evaluate_random_on_wad(
    scenario: str,
    batch_dir: Path,
    entry: WadManifestEntry,
    eval_seeds: int | Iterable[int],
    episodes: int,
    enable_labels_buffer: bool = True,
    max_steps_per_episode: int = 0,
) -> list[dict[str, Any]]:
    seed_list = [int(eval_seeds)] if isinstance(eval_seeds, int) else [int(s) for s in eval_seeds]
    track_goals = any(scenario.startswith(prefix) for prefix in GOAL_TRACKED_SCENARIOS)
    records: list[dict[str, Any]] = []

    with singleton_batch_dir(batch_dir, entry) as temp_batch:
        env = make_random_eval_env(scenario, temp_batch, enable_labels_buffer=enable_labels_buffer)
        try:
            num_agents = int(getattr(env, "num_agents", 2))
            goals = GoalRelocationCounter()
            # ViZDoom is seeded once per WAD; later eval seeds only re-seed the action RNG and keep sampling fresh
            # episodes from the same auto-resetting stream, so every (eval_seed, episode) pair is a distinct episode.
            obs, infos = env.reset(seed=int(seed_list[0]))
            goals.observe(infos)

            for eval_seed in seed_list:
                rng = np.random.default_rng(int(eval_seed) * 100_003 + int(entry.seed))
                step_in_ep = 0
                ep_reward_sum = np.zeros(num_agents, dtype=np.float64)
                ep_idx = 0

                while ep_idx < episodes:
                    actions = []
                    for i in range(num_agents):
                        obs_i = obs[i] if isinstance(obs, (list, tuple)) else obs
                        mask_i = obs_i.get("action_mask") if isinstance(obs_i, dict) else None
                        actions.append(sample_uniform_random_action(env.action_space, rng, mask_i))

                    obs, rewards, terminated, truncated, infos = env.step(actions)
                    step_in_ep += 1
                    if rewards is not None:
                        ep_reward_sum += np.asarray(rewards, dtype=np.float64)

                    force_done = max_steps_per_episode > 0 and step_in_ep >= max_steps_per_episode
                    dones = [bool(t) or bool(tr) or force_done for t, tr in zip(terminated, truncated)]
                    if all(dones):
                        goals.observe(infos)
                        vis_stats = _extract_visibility_from_infos(infos, env)
                        if isinstance(infos, (list, tuple)) and any(
                            isinstance(inf, dict) and "true_objective" in inf for inf in infos
                        ):
                            true_obj = float(
                                np.mean(
                                    [
                                        float(inf["true_objective"])
                                        for inf in infos
                                        if isinstance(inf, dict) and "true_objective" in inf
                                    ]
                                )
                            )
                        elif scenario in ("foraging_commons", "ammo_carrier"):
                            true_obj = float(step_in_ep * 4)
                        elif scenario == "coop_health_gathering":
                            true_obj = float(step_in_ep)
                        else:
                            true_obj = float(np.mean(ep_reward_sum))
                        if track_goals:
                            true_obj = float(goals.count)

                        record: dict[str, Any] = {
                            "scenario": scenario,
                            "algorithm": "Random",
                            "mode": "random",
                            "train_seed": int(eval_seed),
                            "wad_id": entry.wad_id,
                            "wad_seed": int(entry.seed),
                            "episode_index": int(ep_idx),
                            "episode_steps": int(step_in_ep),
                            "true_objective": true_obj,
                            **vis_stats,
                        }
                        if track_goals:
                            record["goals_reached"] = int(goals.count)
                        records.append(record)
                        ep_idx += 1
                        goals.reset()
                        step_in_ep = 0
                        ep_reward_sum.fill(0.0)
                        if force_done and not all(
                            bool(t) or bool(tr) for t, tr in zip(terminated, truncated)
                        ):
                            obs, infos = env.reset()
                            goals.observe(infos)
                        else:
                            goals.observe(fresh_episode_infos(infos))
                    else:
                        goals.observe(infos)
        finally:
            env.close()

    return records


def evaluate_checkpoint_diagnostic_on_wad(
    policy: PolicySpec,
    batch_dir: Path,
    entry: WadManifestEntry,
    episodes: int,
    device: str,
    deterministic: bool,
    *,
    mode: str,
    enable_labels_buffer: bool = False,
    partner_occlusion_mode: str = "none",
    obs_modality_override: str | None = None,
) -> list[dict[str, Any]]:
    with singleton_batch_dir(batch_dir, entry) as temp_batch:
        cfg = build_eval_cfg(policy, Path(temp_batch), device, deterministic)
        cfg.enable_labels_buffer = bool(enable_labels_buffer)
        cfg.partner_occlusion_mode = partner_occlusion_mode
        if obs_modality_override is not None:
            cfg.obs_modality = obs_modality_override
        configure_batch_env_and_agents(cfg)
        register_model_factory(cfg)

        torch_device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
        env = make_eval_env(cfg, render_mode=None)
        try:
            env_info = extract_env_info(env, cfg)
            actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
            actor_critic.eval()
            actor_critic.model_to_device(torch_device)
            checkpoint_dict = Learner.load_checkpoint([policy.checkpoint], torch_device)
            if checkpoint_dict is None:
                raise RuntimeError(f"Could not load checkpoint: {policy.checkpoint}")
            actor_critic.load_state_dict(checkpoint_dict["model"])

            value_based = str(cfg.algo).upper() in VALUE_BASED_ALGOS
            head_sizes = action_head_sizes(env_info.action_space) if value_based else []
            track_goals = any(policy.scenario.startswith(prefix) for prefix in GOAL_TRACKED_SCENARIOS)
            goals = GoalRelocationCounter()

            obs, infos = env.reset()
            goals.observe(infos)
            action_mask = obs.pop("action_mask").to(torch_device) if "action_mask" in obs else None
            rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=torch_device)
            episode_records: list[dict[str, Any]] = []
            with torch.no_grad():
                while len(episode_records) < episodes:
                    normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
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
                    action_mask = obs.pop("action_mask").to(torch_device) if "action_mask" in obs else None
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
                    else:
                        goals.observe(infos)
            return episode_records
        finally:
            env.close()


def aggregate_records_by_scenario_and_algo(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (str(rec["scenario"]), str(rec["algorithm"]), str(rec.get("mode", "eval")))
        grouped.setdefault(key, []).append(rec)

    summaries: list[dict[str, Any]] = []
    for (scenario, algorithm, mode), recs in sorted(grouped.items()):
        by_seed: dict[int, list[float]] = {}
        for r in recs:
            s = int(r.get("train_seed") if r.get("train_seed") is not None else 0)
            by_seed.setdefault(s, []).append(float(r["true_objective"]))
        seed_means = [float(np.mean(vals)) for _, vals in sorted(by_seed.items())]
        vis_agg = {
            vk: float(np.mean([float(r.get(vk, 0.0)) for r in recs]))
            for vk in VISIBILITY_KEYS
        }
        summaries.append(
            {
                "scenario": scenario,
                "algorithm": algorithm,
                "mode": mode,
                "seeds": sorted(by_seed.keys()),
                "num_seeds": len(seed_means),
                "episodes_total": len(recs),
                "seed_means": seed_means,
                "mean_true_objective": float(np.mean(seed_means)) if seed_means else 0.0,
                "std_over_seeds": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
                "ci95_half_width": student_t_ci95_half_width(seed_means),
                **vis_agg,
            }
        )
    return summaries


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("random", "visibility", "partner_occlusion", "blind_eval"),
        default="random",
    )
    parser.add_argument("--archive-root", default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--batch-root", default="results/heldout_wad_batches")
    parser.add_argument("--output-dir", default="results/rebuttal_kd6j")
    parser.add_argument("--scenarios", default=",".join(ALL_13_SCENARIOS))
    parser.add_argument("--algorithms", default="MAPPO,QMIX")
    parser.add_argument("--seeds", default=",".join(map(str, HELDOUT_SEEDS)))
    parser.add_argument("--train-seeds", default=",".join(map(str, DEFAULT_TRAIN_SEEDS)))
    parser.add_argument("--episodes-per-wad", type=int, default=20)
    parser.add_argument("--max-steps-per-episode", type=int, default=0)
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS)
    parser.add_argument("--partner-occlusion-mode", choices=("sprite", "cues", "both"), default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    scenarios = parse_csv_argument(args.scenarios)
    heldout_seeds = tuple(int(v) for v in parse_csv_argument(args.seeds))
    train_seeds = tuple(int(v) for v in parse_csv_argument(args.train_seeds)) if args.train_seeds else DEFAULT_TRAIN_SEEDS
    batch_root = Path(args.batch_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "per_scenario"
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, list[WadManifestEntry]] = {}
    for scenario in scenarios:
        manifests[scenario] = ensure_heldout_batch(scenario, batch_root, heldout_seeds)

    if args.generate_only:
        print(
            json.dumps(
                {
                    "heldout_seeds": list(heldout_seeds),
                    "scenarios": list(scenarios),
                    "batch_root": str(batch_root.resolve()),
                },
                indent=2,
            )
        )
        return

    register_vizdoom_components()
    all_records: list[dict[str, Any]] = []

    if args.mode == "random":
        for scenario in scenarios:
            cache_file = cache_dir / f"random_{scenario}.json"
            if cache_file.is_file() and not args.overwrite:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                all_records.extend(cached)
                print(f"[cached] {scenario}: {len(cached)} episodes", flush=True)
                continue
            scen_records: list[dict[str, Any]] = []
            for entry in manifests[scenario]:
                recs = evaluate_random_on_wad(
                    scenario=scenario,
                    batch_dir=batch_root / scenario,
                    entry=entry,
                    eval_seeds=train_seeds,
                    episodes=int(args.episodes_per_wad),
                    enable_labels_buffer=True,
                    max_steps_per_episode=int(args.max_steps_per_episode),
                )
                scen_records.extend(recs)
            cache_file.write_text(json.dumps(scen_records, indent=2) + "\n", encoding="utf-8")
            all_records.extend(scen_records)
            print(f"[done] {scenario}: {len(scen_records)} episodes", flush=True)
    else:
        algorithms = parse_csv_argument(args.algorithms)
        archive_root = Path(args.archive_root)
        try:
            policies = discover_seeded_policy_specs(
                archive_root, scenarios, algorithms, train_seeds, int(args.target_steps)
            )
        except FileNotFoundError:
            policies = discover_policy_specs(archive_root, scenarios, algorithms, int(args.target_steps))

        enable_labels = args.mode in ("visibility", "partner_occlusion")
        occ_mode = args.partner_occlusion_mode if args.mode == "partner_occlusion" else "none"
        obs_mod = "blind" if args.mode == "blind_eval" else None

        for policy in policies:
            for entry in manifests[policy.scenario]:
                recs = evaluate_checkpoint_diagnostic_on_wad(
                    policy=policy,
                    batch_dir=batch_root / policy.scenario,
                    entry=entry,
                    episodes=int(args.episodes_per_wad),
                    device=args.device,
                    deterministic=bool(args.deterministic),
                    mode=args.mode,
                    enable_labels_buffer=enable_labels,
                    partner_occlusion_mode=occ_mode,
                    obs_modality_override=obs_mod,
                )
                all_records.extend(recs)

    summaries = aggregate_records_by_scenario_and_algo(all_records)
    raw_csv_path = output_dir / f"{args.mode}_episodes.csv"
    summary_json_path = output_dir / f"{args.mode}_summary.json"
    if all_records:
        write_raw_csv(raw_csv_path, all_records)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "heldout_layout_seeds": list(heldout_seeds),
        "train_or_eval_seeds": list(train_seeds),
        "episodes_per_wad": int(args.episodes_per_wad),
        "summaries": summaries,
    }
    summary_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {summary_json_path}")


if __name__ == "__main__":
    main()
