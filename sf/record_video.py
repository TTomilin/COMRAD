"""
High-resolution video recording script for trained MARL models.

Records videos from trained checkpoints at configurable resolutions.
Supports multi-agent environments with grid-based rendering.

================================================================================
HOW TO USE
================================================================================

1. BASIC USAGE (record 1 episode at 720p):

    python -m sf.record_video --env=doom_pitfall --experiment=my_experiment

2. FULL HD (1080p) RECORDING:

    python -m sf.record_video --env=parallel --experiment=exp_name --resolution=1920x1080

3. RECORD MULTIPLE EPISODES:

    python -m sf.record_video --env=armory_siege --experiment=exp_name --max_num_episodes=5

4. USE BEST CHECKPOINT (instead of latest):

    python -m sf.record_video --env=doom_mwh --experiment=exp_name --load_checkpoint_kind=best

5. DETERMINISTIC ACTIONS (argmax instead of sampling):

    python -m sf.record_video --env=doom_pitfall --experiment=exp_name --eval_deterministic=True

6. CUSTOM OUTPUT LOCATION:

    python -m sf.record_video --env=parallel --experiment=exp_name --output_dir=./videos --video_prefix=demo

================================================================================
KEY ARGUMENTS
================================================================================

    --env               Environment name (doom_pitfall, parallel, armory_siege, etc.)
    --experiment        Experiment folder name in train_dir
    --resolution        Video resolution (default: 1280x720)
                        Options: 640x360, 640x480, 800x450, 800x600, 1024x576,
                                 1024x768, 1280x720, 1280x960, 1600x900,
                                 1600x1200, 1920x1080
    --max_num_episodes  Number of episodes to record (default: 1)
    --max_num_frames    Max frames to record (default: 10000)
    --video_fps         Video FPS (default: 35, VizDoom native)
    --load_checkpoint_kind  "latest" or "best" (default: latest)
    --eval_deterministic    Use argmax actions (default: False)
    --output_dir        Output directory (default: experiment folder)
    --video_prefix      Video filename prefix (default: "recording")
    --overwrite_video   Overwrite existing video files

================================================================================
OUTPUT
================================================================================

Videos are saved as: {output_dir}/{video_prefix}_{ENV}_{ALGO}_{resolution}.mp4
Example: train_dir/my_experiment/recording_DP_APPO_1280x720.mp4

"""

import copy
import os
import sys
import time
from collections import deque
from typing import Optional, Tuple

import imageio
import numpy as np
import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint, parse_full_cfg, parse_sf_args
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import experiment_dir, log

from sf.train import register_vizdoom_components
from sf.doom.doom_params import add_doom_env_args, add_doom_env_eval_args, doom_override_defaults, add_wandb_args
from sf.doom.wrappers.observation_space import resolutions as VIZDOOM_RESOLUTIONS


# Available high resolutions for recording (subset of VizDoom supported resolutions)
AVAILABLE_RESOLUTIONS = [
    "640x360", "640x480", "800x450", "800x600",
    "1024x576", "1024x768", "1280x720", "1280x960",
    "1600x900", "1600x1200", "1920x1080"
]


def add_recording_args(parser):
    """Add recording-specific arguments."""
    parser.add_argument(
        "--resolution",
        default="1280x720",
        type=str,
        choices=AVAILABLE_RESOLUTIONS,
        help="Video recording resolution (width x height)"
    )
    parser.add_argument(
        "--video_fps",
        default=35,
        type=int,
        help="Video FPS (VizDoom runs at 35 fps natively)"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory for videos (defaults to experiment directory)"
    )
    parser.add_argument(
        "--video_prefix",
        default="recording",
        type=str,
        help="Prefix for video filename"
    )
    parser.add_argument(
        "--overwrite_video",
        action="store_true",
        help="Overwrite existing video files"
    )


def parse_recording_args(argv=None):
    """Parse arguments for recording script."""
    parser, _ = parse_sf_args(argv=argv, evaluation=True)
    add_doom_env_args(parser)
    add_doom_env_eval_args(parser)
    add_wandb_args(parser)
    add_recording_args(parser)
    doom_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)
    return cfg


def save_video(frames, file_path, fps=35):
    """Save frames as MP4 video using imageio."""
    if not frames:
        log.warning("No frames to save!")
        return

    writer = imageio.get_writer(file_path, fps=fps, quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    log.info(f"Video saved to {file_path} ({len(frames)} frames at {fps} fps)")


def make_env_for_recording(cfg: Config, resolution: str, render_mode: str = "rgb_array"):
    """
    Create environment configured for high-resolution recording.

    This creates a separate config copy with high-res settings to avoid
    affecting the policy environment's configuration.
    """
    # Create a deep copy of config for the render environment
    render_cfg = copy.deepcopy(cfg)

    # Parse target resolution
    width, height = map(int, resolution.split("x"))

    # Update config for high-res rendering
    render_cfg.res_w = width
    render_cfg.res_h = height
    render_cfg.wide_aspect_ratio = width / height > 1.5  # Use wide aspect for 16:9 and wider

    # Disable wandb recording for the render environment
    render_cfg.wandb_record_every = 0
    render_cfg.with_wandb = False

    env = make_env_func_batched(
        render_cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=1),
        render_mode=render_mode
    )

    return env


def record_video(cfg: Config) -> Tuple[StatusCode, float]:
    """
    Record high-resolution video from trained checkpoint.

    Creates two environments:
    - Policy env: at training resolution for inference
    - Render env: at high resolution for video recording
    """
    verbose = True

    # Preserve user-specified settings before loading checkpoint
    train_dir = cfg.train_dir
    resolution = cfg.resolution
    output_dir = getattr(cfg, 'output_dir', None)
    video_prefix = getattr(cfg, 'video_prefix', 'recording')
    video_fps = getattr(cfg, 'video_fps', 35)
    overwrite_video = getattr(cfg, 'overwrite_video', False)
    max_num_episodes = cfg.max_num_episodes
    max_num_frames = cfg.max_num_frames
    eval_deterministic = cfg.eval_deterministic
    load_checkpoint_kind = cfg.load_checkpoint_kind

    # Load config from checkpoint (this overwrites cfg with saved training config)
    cfg = load_from_checkpoint(cfg)

    # Restore user-specified settings
    cfg.train_dir = train_dir
    cfg.resolution = resolution
    cfg.output_dir = output_dir
    cfg.video_prefix = video_prefix
    cfg.video_fps = video_fps
    cfg.overwrite_video = overwrite_video
    cfg.max_num_episodes = max_num_episodes
    cfg.max_num_frames = max_num_frames
    cfg.eval_deterministic = eval_deterministic
    cfg.load_checkpoint_kind = load_checkpoint_kind

    # Generate video file path
    output_dir = cfg.output_dir if cfg.output_dir else experiment_dir(cfg)
    os.makedirs(output_dir, exist_ok=True)

    env_initials = ''.join(word[0].upper() for word in cfg.env.split('_'))
    video_file_path = os.path.join(
        output_dir,
        f"{cfg.video_prefix}_{env_initials}_{cfg.algo}_{resolution}.mp4"
    )

    if os.path.exists(video_file_path) and not overwrite_video:
        log.info(f"Video already exists: {video_file_path}. Use --overwrite_video to replace.")
        return ExperimentStatus.SUCCESS, 0

    # Setup frameskip for smooth rendering
    eval_env_frameskip = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert cfg.env_frameskip % eval_env_frameskip == 0, \
        f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for recording")

    cfg.num_envs = 1

    # Create policy environment (at training resolution)
    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=None
    )

    # Create rendering environment (at high resolution)
    env_render = make_env_for_recording(cfg, resolution, render_mode="rgb_array")

    env_info = extract_env_info(env, cfg)

    # Disable reset on init if available (for VizDoom demo recording)
    if hasattr(env.unwrapped, "reset_on_init"):
        env.unwrapped.reset_on_init = False
    if hasattr(env_render.unwrapped, "reset_on_init"):
        env_render.unwrapped.reset_on_init = False

    # Create and load actor-critic model
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    # Load checkpoint
    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)

    if checkpoint_dict:
        actor_critic.load_state_dict(checkpoint_dict["model"])
        log.info(f"Loaded checkpoint from {Learner.checkpoint_dir(cfg, policy_id)}")
    else:
        raise RuntimeError("Could not load checkpoint")

    # Initialize tracking variables
    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0
    video_frames = []
    num_episodes = 0

    def max_frames_reached(frames):
        return cfg.max_num_frames is not None and frames > cfg.max_num_frames

    # Reset environments
    obs, infos = env.reset()
    _, _ = env_render.reset()

    # Handle action mask if present
    action_mask = obs.pop("action_mask").to(device) if isinstance(obs, dict) and "action_mask" in obs else None

    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    finished_episode = [False for _ in range(env.num_agents)]

    log.info(f"Starting video recording at {resolution} resolution...")
    log.info(f"Recording up to {cfg.max_num_episodes} episodes or {cfg.max_num_frames} frames")
    log.info(f"Environment has {env.num_agents} agents")

    with torch.no_grad():
        while not max_frames_reached(num_frames):
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)

            # Get actions (deterministic or sampled)
            actions = policy_outputs["actions"]
            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]

            for _ in range(render_action_repeat):
                # Capture frame from high-res render environment
                frame = env_render.render()
                if frame is not None:
                    video_frames.append(frame.copy())

                # Step both environments with same actions
                obs, rew, terminated, truncated, infos = env.step(actions)
                _, _, _, _, _ = env_render.step(actions)

                # Handle action mask if present
                action_mask = obs.pop("action_mask").to(device) if isinstance(obs, dict) and "action_mask" in obs else None

                dones = make_dones(terminated, truncated)
                infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                if episode_reward is None:
                    episode_reward = rew.float().clone()
                else:
                    episode_reward += rew.float()

                num_frames += 1
                if num_frames % 500 == 0:
                    log.debug(f"Recorded {num_frames} frames, {len(video_frames)} video frames...")

                dones = dones.cpu().numpy()
                for agent_i, done_flag in enumerate(dones):
                    if done_flag:
                        finished_episode[agent_i] = True
                        rew_val = episode_reward[agent_i].item()
                        episode_rewards[agent_i].append(rew_val)

                        if verbose:
                            log.info(
                                f"Episode finished for agent {agent_i} at {num_frames} frames. "
                                f"Reward: {episode_reward[agent_i]:.3f}"
                            )
                        rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                        episode_reward[agent_i] = 0

                if all(dones):
                    # Capture final frame
                    frame = env_render.render()
                    if frame is not None:
                        video_frames.append(frame.copy())
                    time.sleep(0.05)

                if all(finished_episode):
                    num_agents = env.num_agents if hasattr(env, 'num_agents') else env.get_wrapper_attr('num_agents')
                    finished_episode = [False] * num_agents
                    num_episodes += 1

                    avg_rewards = [np.mean(episode_rewards[i]) for i in range(env.num_agents)]
                    log.info(f"Completed episode {num_episodes}. Avg rewards: {avg_rewards}")

            if num_episodes >= cfg.max_num_episodes:
                break

    env.close()
    env_render.close()

    # Save video
    if video_frames:
        save_video(video_frames, video_file_path, fps=video_fps)
    else:
        log.warning("No frames captured!")

    # Calculate average reward
    total_rewards = sum([sum(episode_rewards[i]) for i in range(env.num_agents)])
    total_episodes = sum([len(episode_rewards[i]) for i in range(env.num_agents)])
    avg_reward = total_rewards / total_episodes if total_episodes > 0 else 0

    log.info(f"Recording complete. Average reward: {avg_reward:.3f}")

    return ExperimentStatus.SUCCESS, avg_reward


def main():
    """Script entry point."""
    register_vizdoom_components()
    cfg = parse_recording_args()

    # Set reasonable defaults for recording
    if not hasattr(cfg, 'max_num_episodes') or cfg.max_num_episodes >= 1e9:
        cfg.max_num_episodes = 1
    if not hasattr(cfg, 'max_num_frames') or cfg.max_num_frames >= 1e9:
        cfg.max_num_frames = 10000

    status, avg_reward = record_video(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
