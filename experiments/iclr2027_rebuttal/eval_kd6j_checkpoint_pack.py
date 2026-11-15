"""
Evaluate a trained COMRAD checkpoint directory on the 5 held-out WADs (`HELDOUT_SEEDS`)
and optionally also run Partner-Occlusion (`--partner_occlusion_mode=both`) and Zero-Shot Blind
(`--obs_modality=blind`) interventions inside the same live `MultiAgentEnv` per WAD.

Every pass goes through `eval_frozen_partner.run_pass`: the intervention is configured before the pass's explicit
`env.reset()` (so step 0 is already masked or unmasked as intended), RNN state and goal counter start from zero, the
RNG is seeded by (wad_seed, pass index), and episodes hit by a hidden crash-recovery reset are discarded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

from comrad.train import (
    configure_batch_env_and_agents,
    register_model_factory,
    register_vizdoom_components,
)
from comrad.utils.doom_utils import CLOCK_EGO, batch_base_env_name, doom_env_by_name, observation_clock_horizon
from experiments.iclr2027_rebuttal.eval_frozen_partner import pass_rng_seed, run_pass
from experiments.iclr2027_rebuttal.kd6j_protocol import KD6J_PACK_PROTOCOL_VERSION
from experiments.iclr2027_rebuttal.rebuttal_kd6j_eval import (
    _extract_visibility_from_infos,
    ensure_heldout_batch,
)
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.enjoy import make_env as make_eval_env
from sample_factory.model.actor_critic import create_actor_critic
from scripts.heldout_wad_evaluation import (
    HELDOUT_SEEDS,
    PolicySpec,
    WadManifestEntry,
    build_eval_cfg,
    select_checkpoint,
    sha256_file,
    singleton_batch_dir,
)

def _find_visibility_wrapper(env: Any) -> Any:
    cur = env
    while cur is not None:
        if "partner_occlusion_mode" in getattr(cur, "__dict__", {}):
            return cur
        cur = getattr(cur, "env", None)
    raise RuntimeError("no VisibilityDiagnosticsWrapper in the env stack; build the env with enable_labels_buffer=True")


def blind_observation(horizon: float) -> Any:
    """Zero-shot blind intervention: what `BlindObservationWrapper` feeds a blind-trained policy (black frame whose
    first pixel encodes the episode progress step / horizon)."""

    def transform(obs: dict[str, torch.Tensor], episode_step: int) -> dict[str, torch.Tensor]:
        zeroed = torch.zeros_like(obs["obs"])
        progress = min(1.0, float(episode_step) / float(horizon))
        flat = zeroed.view(zeroed.shape[0], -1)
        flat[:, 0] = int(round(progress * 255.0)) if zeroed.dtype == torch.uint8 else float(progress)
        return {**obs, "obs": zeroed}

    return transform


def run_mode(
    env: Any,
    actor_critic: Any,
    env_info: Any,
    cfg: Any,
    policy: PolicySpec,
    entry: WadManifestEntry,
    episodes: int,
    torch_device: torch.device,
    *,
    mode: str,
    pass_index: int,
    partner_occlusion_mode: str,
    blind_horizon: float | None,
) -> list[dict[str, Any]]:
    _find_visibility_wrapper(env).partner_occlusion_mode = partner_occlusion_mode
    return run_pass(
        env, actor_critic, env_info, cfg, policy, entry, episodes, torch_device,
        pass_name=mode, partner_policy="learned", partner_idx=None, rng_seed=pass_rng_seed(entry.seed, pass_index),
        obs_transform=None if blind_horizon is None else blind_observation(blind_horizon),
        record_extras=lambda infos: {"mode": mode, "checkpoint": policy.checkpoint, **_extract_visibility_from_infos(infos)},
    )


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

                common = (env, actor_critic, env_info, cfg, policy, entry, int(args.episodes_per_wad), torch_device)
                normal_records.extend(
                    run_mode(*common, mode=args.cell_tag, pass_index=0, partner_occlusion_mode="none", blind_horizon=None)
                )
                if args.run_interventions:
                    occ_records.extend(
                        run_mode(*common, mode="partner_occlusion", pass_index=1, partner_occlusion_mode="both",
                                 blind_horizon=None)
                    )
                    blind_records.extend(
                        run_mode(*common, mode="blind_eval", pass_index=2, partner_occlusion_mode="none",
                                 blind_horizon=observation_clock_horizon(
                                     doom_env_by_name(batch_base_env_name(cfg)), cfg, CLOCK_EGO))
                    )
            finally:
                env.close()

    result: dict[str, Any] = {
        "protocol_version": KD6J_PACK_PROTOCOL_VERSION,
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
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    tmp_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)
    print(f"Saved {out_path} ({len(normal_records)} normal eps)")


if __name__ == "__main__":
    main()
