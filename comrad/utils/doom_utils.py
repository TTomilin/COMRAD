import datetime
import os
from dataclasses import dataclass
from os.path import join
from typing import Optional

from comrad.envs.wad_catalog import WadBatch
from comrad.envs.multi_wad_env import MultiWADEnv

from sample_factory.envs.env_wrappers import (
    PixelFormatChwWrapper,
    RecordingWrapper,
    ResizeWrapper,
    RewardScalingWrapper,
    TimeLimitWrapper,
)
from sample_factory.utils.utils import debug_log_every_n, ensure_dir_exists, experiment_dir, log
from comrad.envs.action_space import (
    doom_action_space_coop_puzzle,
    doom_action_space_armory_siege,
    doom_action_space_lavapit,
    doom_action_space_platform_chain,
    doom_action_space_ammo_carrier,
    doom_action_space_lava_maze,
    doom_action_space_lava_maze_simple,
    doom_action_space_smart_enemies,
    doom_action_space_dumb_enemies,
    doom_action_space_stealth_labyrinth,
    doom_action_space_stag_hunt,
    doom_action_space_stag_hunt_continuous,
    doom_action_space_stag_hunt_continuous_full,
    doom_action_space_coop_health_gathering,
    doom_action_space_foraging_commons,
    doom_action_space_rhythm_sync,
)
from comrad.envs.doom_gym import VizdoomEnv
from comrad.wrappers.additional_input import DoomAdditionalInput
from comrad.wrappers.action_mask import RoleActionMaskWrapper
from comrad.wrappers.multiplayer_stats import MultiplayerStatsWrapper
from comrad.wrappers.observation_space import SetResolutionWrapper, resolutions
from comrad.wrappers.scenario_wrappers import (
    CoopPuzzleRewardShaping, ArmorySiegeRewardShaping, ArmorySiegeAdditionalInput, AmmoCarrierAdditionalInput, AmmoCarrierRewardShaping, LavapitAdditionalInput, LavapitRewardShaping, LavaMazeRewardShaping, LavaMazeAdditionalInput, LavaMazeSimpleRewardShaping, LavaMazeSimpleAdditionalInput, CoopHealthGatheringRewardShaping, ForagingCommonsAdditionalInput, ForagingCommonsRewardShaping, PlatformChainRewardShaping, RhythmSyncAdditionalInput, RhythmSyncRewardShaping, RhythmSyncRewardShapingDense, SmartEnemiesRewardShaping, DumbEnemiesRewardShaping, StagHuntArenaRewardShaping, StealthLabyrinthRewardShaping
)
from comrad.wrappers.shared_reward import SharedRewardWrapper
from comrad.wrappers.video_recorder import VideoLoggerWrapper

OFF_POLICY = {"IDQN", "VDN", "QMIX", "QPLEX"}
ON_POLICY = {"APPO", "IPPO", "MAPPO", "HAPPO"}
_WARNED_UNSUPPORTED_ADDITIONAL_INPUT_ENVS = set()

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
        nofreelook=1,
        respawn_delay=0,
        timelimit=10.0,
        additional_input_wrapper=None,
        use_additional_input=False,
        extra_wrappers=None,
        shared_reward_alpha=0.0,
        shared_reward_scalarisation="sum",
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
        self.nofreelook = nofreelook
        self.respawn_delay = respawn_delay
        self.timelimit = timelimit

        self.additional_input_wrapper = additional_input_wrapper
        self.use_additional_input = use_additional_input
        # expect list of tuples (wrapper_cls, wrapper_kwargs)
        self.extra_wrappers = extra_wrappers
        # reward wrappers before rewards are collected into joint env step
        # unused for off-policy variants
        self.shared_reward_alpha = shared_reward_alpha
        self.shared_reward_scalarisation = shared_reward_scalarisation

ADDITIONAL_INPUT = (DoomAdditionalInput, {})  # health, ammo, etc. as input vector
ARMORY_SIEGE_ADDITIONAL_INPUT = (ArmorySiegeAdditionalInput, {})  # health, ammo, weapons, core_hp
LAVA_MAZE_ADDITIONAL_INPUT = (LavaMazeAdditionalInput, {})
LAVA_MAZE_SIMPLE_ADDITIONAL_INPUT = (LavaMazeSimpleAdditionalInput, {})
LAVA_MAZE_ACTION_MASK = (
    RoleActionMaskWrapper,
    {
        "masks_by_player_id": {
            0: [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            1: [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        },
    },
)
LAVA_MAZE_SIMPLE_ACTION_MASK = (
    RoleActionMaskWrapper,
    {
        "masks_by_player_id": {
            0: [1, 1, 1, 1, 0, 1, 0, 0],
            1: [1, 0, 0, 1, 1, 1, 1, 1],
        },
    },
)
FORAGING_COMMONS_ADDITIONAL_INPUT = (ForagingCommonsAdditionalInput, {})
AMMO_CARRIER_ADDITIONAL_INPUT = (AmmoCarrierAdditionalInput, {})
LAVAPIT_ADDITIONAL_INPUT = (LavapitAdditionalInput, {})
DOOM_ENVS = [
    DoomSpec(
        "lavapit",
        "lavapit.cfg",
        doom_action_space_lavapit(),
        1.0,
        1000,
        num_agents=2,
        forcerespawn=0,
        additional_input_wrapper=LAVAPIT_ADDITIONAL_INPUT,
        extra_wrappers=[(LavapitRewardShaping, {})],
        shared_reward_alpha=1.0,
    ),

    DoomSpec(
        "platform_chain_easy",
        "platform_chain_easy.cfg",
        doom_action_space_platform_chain(),
        1.0,
        5250,
        num_agents=2,
        forcerespawn=0,
        extra_wrappers=[(PlatformChainRewardShaping, {})],
        shared_reward_alpha=1.0,
    ),

    DoomSpec(
        "platform_chain",
        "platform_chain.cfg",
        doom_action_space_platform_chain(),
        1.0,
        5250,
        num_agents=2,
        forcerespawn=0,
        extra_wrappers=[(PlatformChainRewardShaping, {})],
        shared_reward_alpha=1.0,
    ),

    DoomSpec(
        "coop_puzzle",
        "coop_puzzle.cfg",
        doom_action_space_coop_puzzle(),
        1.0,
        2000,
        num_agents=2,
        forcerespawn=0,
        extra_wrappers=[(CoopPuzzleRewardShaping, {})],
        shared_reward_alpha=1.0,
    ),

    DoomSpec(
        "armory_siege",
        "armory_siege.cfg",
        doom_action_space_armory_siege(),
        1.0,
        4500,
        num_agents=2, # I find 2 agents learn better than 3 agents
        respawn_delay=1,
        additional_input_wrapper=ARMORY_SIEGE_ADDITIONAL_INPUT,
        extra_wrappers=[(ArmorySiegeRewardShaping, {})],
    ),

    DoomSpec(
        "lava_maze_simple",
        "lava_maze_simple.cfg",
        doom_action_space_lava_maze_simple(),
        1.0,
        5250,
        num_agents=2,
        forcerespawn=0,
        additional_input_wrapper=LAVA_MAZE_SIMPLE_ADDITIONAL_INPUT,
        extra_wrappers=[(LavaMazeSimpleRewardShaping, {}), LAVA_MAZE_SIMPLE_ACTION_MASK],
        shared_reward_alpha=1.0,
    ),

    DoomSpec(
        "lava_maze",
        "lava_maze.cfg",
        doom_action_space_lava_maze(),
        1.0,
        5250,
        num_agents=2,
        forcerespawn=0,
        nofreelook=0,
        additional_input_wrapper=LAVA_MAZE_ADDITIONAL_INPUT,
        extra_wrappers=[(LavaMazeRewardShaping, {}), LAVA_MAZE_ACTION_MASK],
        shared_reward_alpha=1.0,
    ),

    DoomSpec(
        "smart_enemies",
        "smart_enemies.cfg",
        doom_action_space_smart_enemies(),
        1.0,
        4500,
        num_agents=2,
        forcerespawn=0,
        extra_wrappers=[(SmartEnemiesRewardShaping, {})],
    ),

    DoomSpec(
        "dumb_enemies",
        "dumb_enemies.cfg",
        doom_action_space_dumb_enemies(),
        1.0,
        4500,
        num_agents=2,
        forcerespawn=0,
        extra_wrappers=[(DumbEnemiesRewardShaping, {})],
    ),

    DoomSpec(
        "stealth_labyrinth",
        "stealth_labyrinth.cfg",
        doom_action_space_stealth_labyrinth(),
        1.0,
        5250,
        num_agents=2,
        forcerespawn=0,
        extra_wrappers=[(StealthLabyrinthRewardShaping, {})],
        shared_reward_alpha=1.0,
    ),

    DoomSpec(
        "stag_hunt_arena",
        "stag_hunt_arena.cfg",
        doom_action_space_stag_hunt(),
        1.0,
        4500,
        num_agents=2,
        forcerespawn=1,
        extra_wrappers=[(StagHuntArenaRewardShaping, {})],
        shared_reward_alpha=0.0,
    ),

    DoomSpec(
        "stag_hunt_arena_continuous",
        "stag_hunt_arena_continuous.cfg",
        doom_action_space_stag_hunt_continuous(),
        1.0,
        4500,
        num_agents=2,
        forcerespawn=1,
        extra_wrappers=[(StagHuntArenaRewardShaping, {})],
        shared_reward_alpha=0.0,
    ),

    DoomSpec(
        "stag_hunt_arena_continuous_full",
        "stag_hunt_arena_continuous_full.cfg",
        doom_action_space_stag_hunt_continuous_full(),
        1.0,
        4500,
        num_agents=2,
        forcerespawn=1,
        extra_wrappers=[(StagHuntArenaRewardShaping, {})],
        shared_reward_alpha=0.0,
    ),

    DoomSpec(
        "ammo_carrier",
        "ammo_carrier.cfg",
        doom_action_space_ammo_carrier(),
        1.0,
        2100,
        num_agents=2,
        forcerespawn=0,
        additional_input_wrapper=AMMO_CARRIER_ADDITIONAL_INPUT,
        extra_wrappers=[(AmmoCarrierRewardShaping, {})],
    ),

    DoomSpec(
        "foraging_commons",
        "foraging_commons.cfg",
        doom_action_space_foraging_commons(),
        1.0,
        5250,
        num_agents=2,
        forcerespawn=0,
        additional_input_wrapper=FORAGING_COMMONS_ADDITIONAL_INPUT,
        extra_wrappers=[(ForagingCommonsRewardShaping, {})],
    ),

    DoomSpec(
        "coop_health_gathering",
        "coop_health_gathering.cfg",
        doom_action_space_coop_health_gathering(),
        1.0,
        2100,
        num_agents=2,
        forcerespawn=0,
        extra_wrappers=[(CoopHealthGatheringRewardShaping, {})],
    ),

    DoomSpec(
        "rhythm_sync", # NEXP
        "rhythm_sync.cfg",
        doom_action_space_rhythm_sync(),
        1.0,
        1750,
        num_agents=2,
        forcerespawn=0,
        additional_input_wrapper=(RhythmSyncAdditionalInput, {"feature_set": "partial"}),
        extra_wrappers=[(RhythmSyncRewardShaping, {})],
        shared_reward_alpha=1.0,
    ),

    DoomSpec(
        # Dense shaping learns better than sparse shaping
        "rhythm_sync_dense",
        "rhythm_sync.cfg",
        doom_action_space_rhythm_sync(),
        1.0,
        1750,
        num_agents=2,
        forcerespawn=0,
        additional_input_wrapper=(RhythmSyncAdditionalInput, {"feature_set": "partial"}),
        extra_wrappers=[(RhythmSyncRewardShapingDense, {})],
        shared_reward_alpha=1.0,
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

def get_alpha(cfg, doom_spec) -> float:
    override = getattr(cfg, "shared_reward_alpha", None)
    if override is None:
        return doom_spec.shared_reward_alpha
    return override


def get_scalarisation(cfg, doom_spec) -> str:
    override = getattr(cfg, "shared_reward_scalarisation", None)
    if override is None: return doom_spec.shared_reward_scalarisation
    return override


def get_use_additional_input(cfg, doom_spec) -> bool:
    override = getattr(cfg, "use_additional_input", None)
    if override is None:
        return doom_spec.use_additional_input
    return override


def get_extra_wrappers(cfg, doom_spec):
    wrappers = []
    use_additional_input = get_use_additional_input(cfg, doom_spec)

    if doom_spec.additional_input_wrapper is not None and use_additional_input:
        wrappers.append(doom_spec.additional_input_wrapper)
    elif use_additional_input and doom_spec.additional_input_wrapper is None:
        if doom_spec.name not in _WARNED_UNSUPPORTED_ADDITIONAL_INPUT_ENVS:
            log.warning(
                "Scenario %s does not define an additional input wrapper, ignoring --use_additional_input=True",
                doom_spec.name,
            )
            _WARNED_UNSUPPORTED_ADDITIONAL_INPUT_ENVS.add(doom_spec.name)

    if doom_spec.extra_wrappers is not None:
        wrappers.extend(doom_spec.extra_wrappers)

    return wrappers

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

        from comrad.envs.multiagent.doom_multiagent import VizdoomEnvMultiplayer

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
            nofreelook=doom_spec.nofreelook,
            respawn_delay=doom_spec.respawn_delay,
            timelimit=timelimit,
            render_mode=render_mode,
            host_ip=getattr(cfg, "host_ip", "127.0.0.1"),
        )

    record_to = cfg.record_to if "record_to" in cfg else None
    should_record = False
    if env_config is None:
        should_record = True
    elif env_config.worker_index == 0 and env_config.vector_index == 0 and (player_id is None or player_id == 0):
        should_record = True

    if record_to is not None and should_record:
        env = RecordingWrapper(env, record_to, player_id)

    env = MultiplayerStatsWrapper(env)

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
        #TODO: for TimeLimitWrapper, random_variation_steps may be set to a proper value
        env = TimeLimitWrapper(env, limit=timeout, random_variation_steps=0)

    pixel_format = cfg.pixel_format if "pixel_format" in cfg else "HWC"
    if pixel_format == "CHW":
        env = PixelFormatChwWrapper(env)

    for wrapper_cls, wrapper_kwargs in get_extra_wrappers(cfg, doom_spec):
        env = wrapper_cls(env, **wrapper_kwargs)

    if doom_spec.reward_scaling != 1.0:
        env = RewardScalingWrapper(env, doom_spec.reward_scaling)

    # This is for HAPPO. Read agent_id_wrapper.py
    # Skip temp env (those with player_id=-1) used by MultiAgentEnv.__init__() for obs_space query
    # Should be fine without it as that setting is only for player testing.
    if str(getattr(cfg, 'algo', 'APPO')).upper() == 'HAPPO' and player_id is not None and player_id >= 0:
        from comrad.wrappers.agent_id_wrapper import AgentIDWrapper
        _num_agents = num_agents if num_agents is not None else doom_spec.num_agents
        env = AgentIDWrapper(env, agent_index=player_id, num_agents=_num_agents)

    # Only record video from worker_0, vec_0
    if getattr(cfg, "wandb_record_every", 0) and getattr(cfg, "with_wandb", False) and player_id is None:
        # root = ensure_dir_exists(join(experiment_dir(cfg=cfg), "wandb_videos"))
        # if env_config is not None:
        #     worker_id = getattr(env_config, "worker_index", "main")
        #     vec_id = getattr(env_config, "vector_index", "main")
        # else:
        #     worker_id = "main"
        #     vec_id = "main"
        # dirr = ensure_dir_exists(join(root, f"worker_{worker_id}_vec_{vec_id}"))
        worker_id = getattr(env_config, "worker_index", 0) if env_config else 0
        vec_id = getattr(env_config, "vector_index", 0) if env_config else 0
        if worker_id == 0 and vec_id == 0:
            dirr = ensure_dir_exists(join(experiment_dir(cfg=cfg), "wandb_videos"))
            env = VideoLoggerWrapper(
                env,
                record_every=getattr(cfg, "wandb_record_every", 0),
                fps=getattr(cfg, "wandb_video_fps", 35) // getattr(cfg, "env_frameskip", 1),
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

        from comrad.envs.multiagent.doom_multiagent_wrapper import MultiAgentEnv

        env = MultiAgentEnv(
            num_agents=num_agents,
            make_env_func=make_env_func,
            env_config=env_config,
            skip_frames=skip_frames,
            render_mode=render_mode,
        )

        # For HAPPO, obs_space with temp env (player_id=-1) doesn't get AgentIDWrapper, so MultiAgentEnv.observation_space is missing 'agent_id'
        # SF uses this obs_space to create the model, so we augment it manually here
        # But that setting is only for player testing so shouldn't matter much
        if str(getattr(cfg, 'algo', 'APPO')).upper() == 'HAPPO':
            import gymnasium as gym
            import numpy as np
            spaces = dict(env.observation_space.spaces) if isinstance(env.observation_space, gym.spaces.Dict) else {'obs': env.observation_space}
            spaces['agent_id'] = gym.spaces.Box(low=0.0, high=1.0, shape=(num_agents,), dtype=np.float32)
            env.observation_space = gym.spaces.Dict(spaces)

        shared_reward_alpha = get_alpha(cfg, doom_spec)
        shared_reward_scalarisation = get_scalarisation(cfg, doom_spec)
        if shared_reward_alpha > 0 and str(getattr(cfg, "algo", "MAPPO")).upper() not in OFF_POLICY:
            env = SharedRewardWrapper(
                env,
                alpha=shared_reward_alpha,
                scalarisation=shared_reward_scalarisation,
            )
    else:
        # if we have only one agent, there's no need for multi-agent wrapper
        from comrad.envs.multiagent.doom_multiagent_wrapper import init_multiplayer_env

        env = init_multiplayer_env(make_env_func, player_id=0, env_config=env_config)

    # Only record video from worker_0, vec_0
    if getattr(cfg, "wandb_record_every", 0) and getattr(cfg, "with_wandb", False):
        # root = ensure_dir_exists(join(experiment_dir(cfg=cfg), "wandb_videos"))
        # if env_config is not None:
        #     worker_id = getattr(env_config, "worker_index", "main")
        #     vec_id = getattr(env_config, "vector_index", "main")
        # else:
        #     worker_id = "main"
        #     vec_id = "main"
        # dirr = ensure_dir_exists(join(root, f"worker_{worker_id}_vec_{vec_id}"))
        worker_id = getattr(env_config, "worker_index", 0) if env_config else 0
        vec_id = getattr(env_config, "vector_index", 0) if env_config else 0
        if worker_id == 0 and vec_id == 0:
            dirr = ensure_dir_exists(join(experiment_dir(cfg=cfg), "wandb_videos"))
            algo_name = str(getattr(cfg, "algo", "APPO")).upper()
            env = VideoLoggerWrapper(
                env,
                record_every=getattr(cfg, "wandb_record_every", 0),
                fps=getattr(cfg, "wandb_video_fps", 35) // getattr(cfg, "env_frameskip", 1),
                is_multi=is_multiagent,
                done_mode="any" if is_multiagent and algo_name in ON_POLICY else "all",
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

@dataclass
class DoomBatchSpec:
    base: DoomSpec
    batch_dir: str
    swap_every: int = 1


def make_doom_env_from_batch(batch_spec: DoomBatchSpec, curriculum,
                              _env_name, cfg, env_config,
                              render_mode: Optional[str] = None, **kwargs):
    batch = WadBatch.from_dir(batch_spec.batch_dir)
    base_cfg = _resolve_scenario_cfg(batch_spec.base.env_spec_file)
    base_env = make_doom_env_from_spec(
        batch_spec.base, _env_name, cfg, env_config, render_mode, **kwargs
    )
    seed = env_config.worker_index * 1000 + env_config.vector_index if env_config else 0
    return MultiWADEnv(
        env=base_env,
        batch=batch,
        base_cfg=base_cfg,
        swap_every=batch_spec.swap_every,
        seed=seed,
        curriculum=curriculum,
    )

def _resolve_scenario_cfg(env_spec_file: str) -> str:
    if os.path.isabs(env_spec_file):
        return env_spec_file
    scenarios_dir = join(os.path.dirname(__file__), os.pardir, "scenarios")
    return os.path.normpath(join(scenarios_dir, env_spec_file))
