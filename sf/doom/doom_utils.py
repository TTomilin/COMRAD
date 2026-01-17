import datetime
import os
from os.path import join
from typing import Optional

from sample_factory.envs.env_wrappers import (
    PixelFormatChwWrapper,
    RecordingWrapper,
    ResizeWrapper,
    RewardScalingWrapper,
    TimeLimitWrapper,
)
from sample_factory.utils.utils import debug_log_every_n, ensure_dir_exists, experiment_dir
from sf.doom.action_space import (
    doom_action_space_pitfall,
    doom_action_space_parallel,
    doom_action_space_armory_siege,
    doom_action_space_safe_ground2,
)
from sf.doom.doom_gym import VizdoomEnv
from sf.doom.wrappers.additional_input import DoomAdditionalInput
from sf.doom.wrappers.observation_space import SetResolutionWrapper, resolutions
from sf.doom.wrappers.scenario_wrappers import DoomPitfallRewardShaping, DoomMWHRewardShaping, ParallelReward, ArmorySiegeRewardShaping, DoomSafeGround2RewardShaping
from sf.doom.wrappers.video_recorder import VideoLoggerWrapper

class DoomSpec:
    def __init__(
        self,
        name,
        env_spec_file,
        action_space,
        reward_scaling=1.0,
        default_timeout=-1,
        num_agents=1,
        num_bots=0,
        forcerespawn=1,
        respawn_delay=0,
        timelimit=4.0,
        extra_wrappers=None,
    ):
        self.name = name
        self.env_spec_file = env_spec_file
        self.action_space = action_space
        self.reward_scaling = reward_scaling
        self.default_timeout = default_timeout

        # 1 for singleplayer, >1 otherwise
        self.num_agents = num_agents

        self.num_bots = num_bots

        self.forcerespawn = forcerespawn
        self.respawn_delay = respawn_delay
        self.timelimit = timelimit

        # expect list of tuples (wrapper_cls, wrapper_kwargs)
        self.extra_wrappers = extra_wrappers

ADDITIONAL_INPUT = (DoomAdditionalInput, {})  # health, ammo, etc. as input vector
DOOM_ENVS = [
    DoomSpec(
        "safe_ground2",
        "safe_ground2.cfg",
        doom_action_space_safe_ground2(),
        1.0,
        1000,
        num_agents=2,
        extra_wrappers=[(DoomSafeGround2RewardShaping, {})],
    ),

    DoomSpec(
        "doom_pitfall",
        "pitfall.cfg",
        doom_action_space_pitfall(),
        1.0,
        1000,
        num_agents=2,
        extra_wrappers=[(DoomPitfallRewardShaping, {})],
    ),

    DoomSpec(
        "doom_mwh",
        "my_way_home_multi.cfg",
        doom_action_space_parallel(),
        num_agents=2, # reward shaping is set only for 2 agents, dont increase
        extra_wrappers=[(DoomMWHRewardShaping, {})],
    ),

    DoomSpec(
        "parallel",
        "prot_beta_long.cfg",
        doom_action_space_parallel(),
        1.0,
        1200,
        num_agents=2,
        forcerespawn=0,
        extra_wrappers=[(ParallelReward, {})],
    ),

    DoomSpec(
        "armory_siege",
        "armory_siege.cfg",
        doom_action_space_armory_siege(),
        1.0,
        2100,
        num_agents=3,
        respawn_delay=1,
        extra_wrappers=[(ArmorySiegeRewardShaping, {})],
    ),
]


def doom_env_by_name(name):
    for cfg in DOOM_ENVS:
        if cfg.name == name:
            return cfg
    raise RuntimeError("Unknown Doom env")

def get_num_agents(cfg, env_name):
    spec = doom_env_by_name(env_name)
    return spec.num_agents if cfg.num_agents <= 0 else cfg.num_agents

# noinspection PyUnusedLocal
def make_doom_env_impl(
    doom_spec,
    cfg=None,
    env_config=None,
    skip_frames=None,
    episode_horizon=None,
    player_id=None,
    num_agents=None,
    max_num_players=None,
    num_bots=0,  # for multi-agent
    custom_resolution=None,
    render_mode: Optional[str] = None,
    **kwargs,
):
    assert cfg is not None

    skip_frames = skip_frames if skip_frames is not None else cfg.env_frameskip

    fps = cfg.fps if "fps" in cfg else None
    async_mode = fps == 0

    if player_id is None:
        env = VizdoomEnv(
            doom_spec.action_space,
            doom_spec.env_spec_file,
            skip_frames=skip_frames,
            async_mode=async_mode,
            render_mode=render_mode,
        )
    else:
        timelimit = cfg.timelimit if cfg.timelimit is not None else doom_spec.timelimit

        from sf.doom.multiplayer.doom_multiagent import VizdoomEnvMultiplayer

        env = VizdoomEnvMultiplayer(
            doom_spec.action_space,
            doom_spec.env_spec_file,
            player_id=player_id,
            num_agents=num_agents,
            max_num_players=max_num_players,
            num_bots=num_bots,
            skip_frames=skip_frames,
            async_mode=async_mode,
            forcerespawn=doom_spec.forcerespawn,
            respawn_delay=doom_spec.respawn_delay,
            timelimit=timelimit,
            render_mode=render_mode,
        )

    record_to = cfg.record_to if "record_to" in cfg else None
    should_record = False
    if env_config is None:
        should_record = True
    elif env_config.worker_index == 0 and env_config.vector_index == 0 and (player_id is None or player_id == 0):
        should_record = True

    if record_to is not None and should_record:
        env = RecordingWrapper(env, record_to, player_id)

    resolution = custom_resolution
    if resolution is None:
        resolution = "256x144" if cfg.wide_aspect_ratio else "160x120"
        # resolution = "640x480" # Custom resolution

    assert resolution in resolutions
    env = SetResolutionWrapper(env, resolution)  # default (wide aspect ratio)

    assert env.observation_space.shape is not None
    h, w, channels = env.observation_space.shape
    if w != cfg.res_w or h != cfg.res_h:
        env = ResizeWrapper(env, cfg.res_w, cfg.res_h, grayscale=False)

    debug_log_every_n(50, "Doom resolution: %s, resize resolution: %r", resolution, (cfg.res_w, cfg.res_h))

    # randomly vary episode duration to somewhat decorrelate the experience
    timeout = doom_spec.default_timeout
    if episode_horizon is not None and episode_horizon > 0:
        timeout = episode_horizon
    if timeout > 0:
        env = TimeLimitWrapper(env, limit=timeout, random_variation_steps=0)

    pixel_format = cfg.pixel_format if "pixel_format" in cfg else "HWC"
    if pixel_format == "CHW":
        env = PixelFormatChwWrapper(env)

    if doom_spec.extra_wrappers is not None:
        for wrapper_cls, wrapper_kwargs in doom_spec.extra_wrappers:
            env = wrapper_cls(env, **wrapper_kwargs)

    if doom_spec.reward_scaling != 1.0:
        env = RewardScalingWrapper(env, doom_spec.reward_scaling)

    if getattr(cfg, "wandb_record_every", 0) and getattr(cfg, "with_wandb", False) and player_id is None:
        root = ensure_dir_exists(join(experiment_dir(cfg=cfg), "wandb_videos"))
        if env_config is not None:
            worker_id = getattr(env_config, "worker_index", "main")
            vec_id = getattr(env_config, "vector_index", "main")
        else:
            worker_id = "main"
            vec_id = "main"
        dirr = ensure_dir_exists(join(root, f"worker_{worker_id}_vec_{vec_id}"))
        env = VideoLoggerWrapper(
            env,
            record_every=getattr(cfg, "wandb_record_every", 0),
            fps=getattr(cfg, "wandb_video_fps", 35),
            is_multi=False,
            output_dir=dirr,
        )

    return env


def make_doom_multiplayer_env(doom_spec, cfg=None, env_config=None, render_mode: Optional[str] = None, **kwargs):
    assert cfg is not None

    skip_frames = cfg.env_frameskip

    if cfg.num_bots < 0:
        num_bots = doom_spec.num_bots
    else:
        num_bots = cfg.num_bots

    num_agents = doom_spec.num_agents if cfg.num_agents <= 0 else cfg.num_agents
    max_num_players = num_agents + cfg.num_humans

    is_multiagent = num_agents > 1

    def make_env_func(player_id):
        return make_doom_env_impl(
            doom_spec,
            cfg=cfg,
            player_id=player_id,
            num_agents=num_agents,
            max_num_players=max_num_players,
            num_bots=num_bots,
            skip_frames=1 if is_multiagent else skip_frames,  # multi-agent skipped frames are handled by the wrapper
            env_config=env_config,
            render_mode=render_mode,
            **kwargs,
        )

    if is_multiagent:
        # create a wrapper that treats multiple game instances as a single multi-agent environment

        from sf.doom.multiplayer.doom_multiagent_wrapper import MultiAgentEnv

        env = MultiAgentEnv(
            num_agents=num_agents,
            make_env_func=make_env_func,
            env_config=env_config,
            skip_frames=skip_frames,
            render_mode=render_mode,
            is_pitfall=doom_spec.name == "doom_pitfall",
        )
    else:
        # if we have only one agent, there's no need for multi-agent wrapper
        from sf.doom.multiplayer.doom_multiagent_wrapper import init_multiplayer_env

        env = init_multiplayer_env(make_env_func, player_id=0, env_config=env_config)

    if getattr(cfg, "wandb_record_every", 0) and getattr(cfg, "with_wandb", False):
        root = ensure_dir_exists(join(experiment_dir(cfg=cfg), "wandb_videos"))
        if env_config is not None:
            worker_id = getattr(env_config, "worker_index", "main")
            vec_id = getattr(env_config, "vector_index", "main")
        else:
            worker_id = "main"
            vec_id = "main"
        dirr = ensure_dir_exists(join(root, f"worker_{worker_id}_vec_{vec_id}"))
        env = VideoLoggerWrapper(
            env,
            record_every=getattr(cfg, "wandb_record_every", 0),
            fps=getattr(cfg, "wandb_video_fps", 35),
            is_multi=True,
            output_dir=dirr,
        )

    return env


def make_doom_env(env_name, cfg, env_config, render_mode: Optional[str] = None, **kwargs):
    spec = doom_env_by_name(env_name)
    return make_doom_env_from_spec(spec, env_name, cfg, env_config, render_mode, **kwargs)


def make_doom_env_from_spec(spec, _env_name, cfg, env_config, render_mode: Optional[str] = None, **kwargs):
    """
    Makes a Doom environment from a DoomSpec instance.
    _env_name is unused but we keep it, so functools.partial(make_doom_env_from_spec, env_spec) can registered
    in Sample Factory (first argument in make_env_func is expected to be the env_name).
    """

    if "record_to" in cfg and cfg.record_to:
        tstamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        cfg.record_to = join(cfg.record_to, f"{cfg.experiment}", tstamp)
        if not os.path.isdir(cfg.record_to):
            os.makedirs(cfg.record_to, exist_ok=True)
    else:
        cfg.record_to = None

    if spec.num_agents > 1 or spec.num_bots > 0:
        # requires multiplayer setup (e.g. at least a host, not a singleplayer game)
        return make_doom_multiplayer_env(spec, cfg=cfg, env_config=env_config, render_mode=render_mode, **kwargs)
    else:
        return make_doom_env_impl(spec, cfg=cfg, env_config=env_config, render_mode=render_mode, **kwargs)
