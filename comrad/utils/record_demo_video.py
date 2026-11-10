"""Faithful high-res videos: the policy plays at its training resolution while the host writes a ViZDoom .lmp demo,
which is then replayed at the video resolution from every player's viewpoint.

Rendering the policy env directly at e.g. 1920x1080 changes Doom's HUD/status-bar layout, so the observations no longer
match training and the agents behave much worse; demo replay avoids that while keeping native high-res frames.
"""
import glob
import json
import os
import shutil
import struct
import sys
import tempfile
from typing import List, Tuple

import imageio
import numpy as np
import torch
import vizdoom as vzd

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import experiment_dir, log

from comrad.utils.record_video import checkpoint_paths_for_recording, parse_recording_args
from comrad.train import register_model_factory, register_vizdoom_components
from comrad.utils.recording_actions import reshape_deterministic_actions


VALUE_BASED_ALGOS = ("IDQN", "QMIX", "VDN", "QPLEX")


def action_head_sizes(action_space) -> List[int]:
    if hasattr(action_space, "n"):
        return [int(action_space.n)]
    return [int(space.n) for space in action_space.spaces if hasattr(space, "n")]


def greedy_q_actions(action_logits: torch.Tensor, head_sizes: List[int]) -> torch.Tensor:
    """Per-head argmax over Q-values (epsilon=0)."""
    if len(head_sizes) == 1:
        return action_logits.argmax(dim=1)
    return torch.stack([q.argmax(dim=1) for q in torch.split(action_logits, head_sizes, dim=1)], dim=1)


# ACS HudMessage overlays drawn for player 0 (e.g. the Lava Maze navigator's S_BLACK blindfold) are rendered for the
# console player in *every* replay viewpoint; these lumps are blanked in the other players' replays.
PLAYER0_ONLY_HUD_LUMPS = {
    "lava_maze": ["S_BLACK"],
}


def write_transparent_lumps_wad(path: str, lump_names: List[str]) -> str:
    """PWAD replacing each lump with a fully transparent 1x1 Doom patch (one column, no posts)."""
    patch = struct.pack("<hhhhI", 1, 1, 0, 0, 12) + b"\xff"
    data = patch * len(lump_names)
    directory = b"".join(
        struct.pack("<ii8s", 12 + i * len(patch), len(patch), name.encode()[:8]) for i, name in enumerate(lump_names)
    )
    with open(path, "wb") as f:
        f.write(b"PWAD" + struct.pack("<ii", len(lump_names), 12 + len(data)) + data + directory)
    return path


def hidden_lumps_for(config_path: str) -> List[str]:
    name = os.path.basename(config_path)
    return next((lumps for prefix, lumps in PLAYER0_ONLY_HUD_LUMPS.items() if name.startswith(prefix)), [])


def replay_game_args(host_game_args: str) -> str:
    """Host args minus the networking ones: a replay is a local single-process game."""
    tokens = host_game_args.split()
    kept, skip_next = [], False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in ("-host", "-port", "-join"):
            skip_next = True
            continue
        if tok.startswith("+viz_connect_timeout"):
            skip_next = True
            continue
        kept.append(tok)
    return " ".join(kept)


def latest_demo(demo_dir: str) -> str:
    demos = sorted(glob.glob(os.path.join(demo_dir, "*.lmp")), key=os.path.getmtime)
    if not demos:
        raise FileNotFoundError(f"No .lmp demo was written to {demo_dir}")
    return demos[-1]


def play_and_record_demo(cfg: Config, demo_dir: str) -> Tuple[List[float], List, List]:
    """Runs one episode with the policy at training resolution; returns per-agent rewards / true objectives."""
    cfg.demo_record_to = demo_dir
    cfg.num_envs = 1
    env = make_env_func_batched(cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=None)
    env_info = extract_env_info(env, cfg)
    if hasattr(env.unwrapped, "reset_on_init"):
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)
    checkpoint_dict = Learner.load_checkpoint(checkpoint_paths_for_recording(cfg, cfg.policy_index), device)
    if not checkpoint_dict:
        raise RuntimeError("Could not load checkpoint")
    actor_critic.load_state_dict(checkpoint_dict["model"])

    value_based = str(cfg.algo).upper() in VALUE_BASED_ALGOS
    head_sizes = action_head_sizes(env_info.action_space) if value_based else []
    num_agents = env.num_agents
    obs, _ = env.reset()
    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
    rnn_states = torch.zeros([num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = torch.zeros(num_agents)
    rewards, objectives, successes = [None] * num_agents, [None] * num_agents, [None] * num_agents
    num_steps = 0

    with torch.no_grad():
        while not all(r is not None for r in rewards) and num_steps < cfg.max_num_frames:
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask, sample_actions=not value_based)
            if value_based:
                actions = greedy_q_actions(policy_outputs["action_logits"], head_sizes)
            else:
                actions = policy_outputs["actions"]
            if cfg.eval_deterministic and not value_based:
                actions = reshape_deterministic_actions(actions, argmax_actions(actor_critic.action_distribution()))
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)
            rnn_states = policy_outputs["new_rnn_states"]

            obs, rew, terminated, truncated, infos = env.step(actions)
            action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
            infos = [{} for _ in range(num_agents)] if infos is None else infos
            episode_reward += rew.float().cpu()
            num_steps += 1
            for i, done in enumerate(make_dones(terminated, truncated).cpu().numpy()):
                if done and rewards[i] is None:
                    rewards[i] = episode_reward[i].item()
                    objectives[i] = infos[i].get("true_objective") if i < len(infos) else None
                    successes[i] = bool(infos[i].get("success", False)) if i < len(infos) else False
                    log.info("Agent %d finished after %d steps. Reward: %.3f, true_objective: %s, success: %s",
                             i, num_steps, rewards[i], objectives[i], successes[i])
                    rnn_states[i] = 0
    env.close()  # finalizes the demo
    return rewards, objectives, successes


def replay_demo_to_video(demo_path: str, meta: dict, resolution: str, out_prefix: str, fps: int, quality: int) -> List[str]:
    """Replays the demo from each player's view in lockstep; writes per-player and side-by-side mp4s."""
    width, height = map(int, resolution.split("x"))
    num_players = int(meta.get("max_num_players", 1))
    games = []
    hidden = hidden_lumps_for(meta["config_path"])
    override_wad = write_transparent_lumps_wad(f"{out_prefix}_hide.wad", hidden) if hidden else None
    for player in range(1, num_players + 1):
        game = vzd.DoomGame()
        game.load_config(meta["config_path"])
        game.set_screen_resolution(getattr(vzd.ScreenResolution, f"RES_{width}X{height}"))
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_window_visible(False)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_seed(int(meta.get("seed", 0)))
        game.add_game_args(replay_game_args(meta.get("game_args", "")))
        if override_wad is not None and player > 1:
            game.add_game_args(f"-file {override_wad}")
        game.init()
        game.replay_episode(demo_path, player)
        games.append(game)

    paths = [f"{out_prefix}_p{p}_{resolution}.mp4" for p in range(1, num_players + 1)]
    writers = [imageio.get_writer(p, fps=fps, quality=quality, macro_block_size=1) for p in paths]
    combined_path = f"{out_prefix}_side_by_side_{resolution}.mp4" if num_players > 1 else None
    combined = imageio.get_writer(combined_path, fps=fps, quality=quality, macro_block_size=1) if combined_path else None

    num_frames = 0
    last = [np.zeros((height, width, 3), dtype=np.uint8) for _ in games]
    while not all(g.is_episode_finished() for g in games):
        for i, game in enumerate(games):
            if game.is_episode_finished():
                continue
            state = game.get_state()
            if state is not None and state.screen_buffer is not None:
                last[i] = state.screen_buffer
            game.advance_action()
        for w, frame in zip(writers, last):
            w.append_data(frame)
        if combined is not None:
            combined.append_data(np.concatenate(last, axis=1))
        num_frames += 1

    for w in writers:
        w.close()
    if combined is not None:
        combined.close()
        paths.append(combined_path)
    for g in games:
        g.close()
    if override_wad is not None:
        os.remove(override_wad)
    log.info("Replayed %d tics (%.1f s) into %s", num_frames, num_frames / fps, paths)
    return paths


def record_demo_video(cfg: Config) -> Tuple[StatusCode, float]:
    cfg = load_from_checkpoint(cfg)
    register_model_factory(cfg)
    output_dir = cfg.output_dir or experiment_dir(cfg)
    os.makedirs(output_dir, exist_ok=True)

    # ViZDoom silently refuses demo paths longer than ~100 chars
    demo_dir = tempfile.mkdtemp(prefix="comrad_demo_", dir="/tmp")
    try:
        rewards, objectives, successes = play_and_record_demo(cfg, demo_dir)
        if cfg.replay_only_success and not any(successes):
            log.info("Episode was not a success; skipping replay (--replay_only_success)")
            valid = [r for r in rewards if r is not None]
            return ExperimentStatus.SUCCESS, float(np.mean(valid)) if valid else 0.0
        demo = latest_demo(demo_dir)
        with open(f"{demo}.json") as f:
            meta = json.load(f)
        out_prefix = os.path.join(output_dir, cfg.video_prefix)
        shutil.copy(demo, f"{out_prefix}.lmp")
        with open(f"{out_prefix}.json", "w") as f:
            json.dump({**meta, "rewards": rewards, "true_objective": objectives, "success": successes}, f, indent=2)
        replay_demo_to_video(demo, meta, cfg.resolution, out_prefix, cfg.video_fps, cfg.video_quality)
    finally:
        shutil.rmtree(demo_dir, ignore_errors=True)
    valid = [r for r in rewards if r is not None]
    return ExperimentStatus.SUCCESS, float(np.mean(valid)) if valid else 0.0


def main():
    register_vizdoom_components()
    cfg = parse_recording_args()
    status, avg_reward = record_demo_video(cfg)
    log.info(f"Average reward: {avg_reward:.3f}")
    return status


if __name__ == "__main__":
    sys.exit(main())
