import copy
import os
import sys
import time
from collections import deque
from typing import Tuple

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

from comrad.train import register_vizdoom_components, register_model_factory
from comrad.envs.doom_params import add_doom_env_args, add_doom_env_eval_args, doom_override_defaults, add_wandb_args
from comrad.utils.recording_actions import reshape_deterministic_actions

AVAILABLE_RESOLUTIONS = [
    "640x360", "640x480", "800x450", "800x600",
    "1024x576", "1024x768", "1280x720", "1280x960",
    "1600x900", "1600x1200", "1920x1080"
]


def add_recording_args(parser):
    parser.add_argument("--resolution", default="1280x720", type=str, choices=AVAILABLE_RESOLUTIONS,
                        help="Video recording resolution (width x height)")
    parser.add_argument("--video_fps", default=35, type=int, help="Video FPS (VizDoom runs at 35 fps natively)")
    parser.add_argument("--output_dir", default=None, type=str, help="Output directory for videos (defaults to experiment directory)")
    parser.add_argument("--video_prefix", default="recording", type=str, help="Prefix for video filename")
    parser.add_argument("--overwrite_video", action="store_true", help="Overwrite existing video files")
    parser.set_defaults(max_num_episodes=1, max_num_frames=10000)


def parse_recording_args(argv=None):
    parser, _ = parse_sf_args(argv=argv, evaluation=True)
    add_doom_env_args(parser)
    add_doom_env_eval_args(parser)
    add_wandb_args(parser)
    add_recording_args(parser)
    doom_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)
    return cfg


def save_video(frames, file_path, fps=35):
    # IMAGEIO FFMPEG_WRITER WARNING occurs when resolution like 640x360 has height 360, which isn't divisible by 16 (the macro block size for H.264)
    # Thus here must set macro_block_size here
    writer = imageio.get_writer(file_path, fps=fps, quality=8, macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    log.info(f"Video saved to {file_path} ({len(frames)} frames at {fps} fps)")


def make_env_for_recording(cfg: Config, resolution: str):
    """Separate config copy at high-res to avoid affecting policy env."""
    render_cfg = copy.deepcopy(cfg)

    width, height = map(int, resolution.split("x"))
    render_cfg.res_w = width
    render_cfg.res_h = height
    render_cfg.wide_aspect_ratio = width / height > 1.5

    render_cfg.wandb_record_every = 0
    render_cfg.with_wandb = False

    return make_env_func_batched(
        render_cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=1),
        render_mode="rgb_array"
    )


def record_video(cfg: Config) -> Tuple[StatusCode, float]:
    verbose = True

    cfg = load_from_checkpoint(cfg)
    register_model_factory(cfg)

    output_dir = cfg.output_dir or experiment_dir(cfg)
    os.makedirs(output_dir, exist_ok=True)

    env_initials = ''.join(word[0].upper() for word in cfg.env.split('_'))
    video_file_path = os.path.join(output_dir,f"{cfg.video_prefix}_{env_initials}_{cfg.algo}_{cfg.resolution}.mp4")

    if os.path.exists(video_file_path) and not cfg.overwrite_video:
        log.info(f"Video already exists: {video_file_path}. Use --overwrite_video to replace.")
        return ExperimentStatus.SUCCESS, 0

    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for recording")

    cfg.num_envs = 1

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=None
    )
    env_render = make_env_for_recording(cfg, cfg.resolution)
    env_info = extract_env_info(env, cfg)

    # reset call ruins the demo recording for VizDoom
    if hasattr(env.unwrapped, "reset_on_init"):
        env.unwrapped.reset_on_init = False
    if hasattr(env_render.unwrapped, "reset_on_init"):
        env_render.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    if checkpoint_dict:
        actor_critic.load_state_dict(checkpoint_dict["model"])
    else:
        raise RuntimeError("Could not load checkpoint")

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0
    video_frames = []
    num_episodes = 0

    def max_frames_reached(frames):
        return cfg.max_num_frames is not None and frames > cfg.max_num_frames

    obs, infos = env.reset()
    _, _ = env_render.reset()

    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    finished_episode = [False for _ in range(env.num_agents)]

    log.info(f"Recording at {cfg.resolution}, up to {cfg.max_num_episodes} episodes, {env.num_agents} agents")

    with torch.no_grad():
        while not max_frames_reached(num_frames):
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)

            actions = policy_outputs["actions"]
            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = reshape_deterministic_actions(actions, argmax_actions(action_distribution))

            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]

            for _ in range(render_action_repeat):
                frame = env_render.render()
                if frame is not None:
                    video_frames.append(frame.copy())

                obs, rew, terminated, truncated, infos = env.step(actions)
                _, _, _, _, _ = env_render.step(actions)

                action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
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
                        episode_rewards[agent_i].append(episode_reward[agent_i].item())

                        if verbose:
                            log.info(
                                "Episode finished for agent %d at %d frames. Reward: %.3f",
                                agent_i, num_frames, episode_reward[agent_i],
                            )
                        rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                        episode_reward[agent_i] = 0

                if all(dones):
                    frame = env_render.render()
                    if frame is not None:
                        video_frames.append(frame.copy())
                    time.sleep(0.05)

                if all(finished_episode):
                    finished_episode = [False] * env.num_agents
                    num_episodes += 1

                    avg_rewards = [np.mean(episode_rewards[i]) for i in range(env.num_agents)]
                    log.info(f"Completed episode {num_episodes}. Avg rewards: {avg_rewards}")

            if num_episodes >= cfg.max_num_episodes:
                break

    env.close()
    env_render.close()

    if video_frames:
        save_video(video_frames, video_file_path, fps=cfg.video_fps)
    else:
        log.warning("No frames captured!")

    total_rewards = sum([sum(episode_rewards[i]) for i in range(env.num_agents)])
    total_episodes = sum([len(episode_rewards[i]) for i in range(env.num_agents)])
    avg_reward = total_rewards / total_episodes if total_episodes > 0 else 0

    log.info(f"Recording complete. Average reward: {avg_reward:.3f}")

    return ExperimentStatus.SUCCESS, avg_reward


def main():
    register_vizdoom_components()
    cfg = parse_recording_args()
    status, avg_reward = record_video(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
