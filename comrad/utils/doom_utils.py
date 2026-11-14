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
from sample_factory.utils.attr_dict import AttrDict
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
from comrad.wrappers.scenario_wrappers.lava_maze import LavaMazeOracleHintInput, LavaMazeOracleMessageWrapper
from comrad.wrappers.scripted_partner import NoopPartnerPolicy, ScriptedPartnerWrapper
from comrad.wrappers.shared_reward import SharedRewardWrapper
from comrad.wrappers.entity_state import EntityStateWrapper
from comrad.wrappers.global_state import GlobalStateWrapper
from comrad.wrappers.shared_vision import SharedVisionWrapper
from comrad.wrappers.state_observation import BlindObservationWrapper, StateObservationWrapper
from comrad.wrappers.visibility_diagnostics import (
    CUE_LAYOUT_NO_CUE_CHANNEL,
    CUE_LAYOUT_SCENARIO_MEASUREMENTS,
    VisibilityDiagnosticsWrapper,
)
from comrad.scripted.lava_maze_navigator import LavaMazeNavigatorPolicy
from comrad.scripted.lava_maze_observer import LavaMazeObserverPolicy
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
        joint_wrappers=None,
    ):
        self.name = name
        self.env_spec_file = env_spec_file
        self.action_space = action_space
        self.reward_scaling = reward_scaling
        self.default_timeout = default_timeout

        # players in the ViZDoom game: 1 for singleplayer, >1 otherwise
        self.num_players = num_agents
        # players driven by a scripted policy inside a joint wrapper, hidden from the learner
        self.num_scripted_players = sum(getattr(w_cls, "scripted_players", 0) for w_cls, _ in (joint_wrappers or []))
        # learner agents exposed by the final wrapped env (what the models and the centralized critic see)
        self.num_agents = self.num_players - self.num_scripted_players

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
        # list of tuples (wrapper_cls, wrapper_kwargs) applied around MultiAgentEnv (cross-agent logic)
        self.joint_wrappers = joint_wrappers

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
# Oracle-message variant: the observer's ATTACK is masked so the ACS flash never fires; head 3 is the message
LAVA_MAZE_ORACLE_MSG_ACTION_MASK = (
    RoleActionMaskWrapper,
    {
        "masks_by_player_id": {
            0: [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            1: [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
        },
    },
)
LAVA_MAZE_ORACLE_HINT_INPUT = (LavaMazeOracleHintInput, {})
LAVA_MAZE_SCRIPTED_OBSERVER = (
    ScriptedPartnerWrapper,
    {"learner_index": 0, "partner_policy": LavaMazeObserverPolicy},
)
# converse reference: scripted navigator (fixed colour -> direction convention), learned observer
LAVA_MAZE_SCRIPTED_NAVIGATOR = (
    ScriptedPartnerWrapper,
    {"learner_index": 1, "partner_policy": LavaMazeNavigatorPolicy},
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

    # Learnability references on the original Lava Maze (paper App. very_difficult)
    DoomSpec(
        # scripted BFS observer (hidden from SF), learned navigator
        "lava_maze_scripted_observer",
        "lava_maze_scripted_observer.cfg",
        doom_action_space_lava_maze(),
        1.0,
        5250,
        num_agents=2,
        forcerespawn=0,
        nofreelook=0,
        additional_input_wrapper=LAVA_MAZE_ADDITIONAL_INPUT,
        use_additional_input=True,
        extra_wrappers=[(LavaMazeRewardShaping, {}), LAVA_MAZE_ACTION_MASK],
        shared_reward_alpha=1.0,
        joint_wrappers=[LAVA_MAZE_SCRIPTED_OBSERVER],
    ),

    DoomSpec(
        # converse: scripted navigator (hidden from SF) obeys the fixed colour convention, learned observer
        "lava_maze_scripted_navigator",
        "lava_maze.cfg",
        doom_action_space_lava_maze(),
        1.0,
        5250,
        num_agents=2,
        forcerespawn=0,
        nofreelook=0,
        additional_input_wrapper=LAVA_MAZE_ADDITIONAL_INPUT,
        use_additional_input=True,
        extra_wrappers=[(LavaMazeRewardShaping, {}), LAVA_MAZE_ACTION_MASK],
        shared_reward_alpha=1.0,
        joint_wrappers=[LAVA_MAZE_SCRIPTED_NAVIGATOR],
    ),

    DoomSpec(
        # lossless discrete message head: observer head 3 -> navigator measurements next step
        "lava_maze_oracle_msg",
        "lava_maze.cfg",
        doom_action_space_lava_maze(),
        1.0,
        5250,
        num_agents=2,
        forcerespawn=0,
        nofreelook=0,
        additional_input_wrapper=LAVA_MAZE_ADDITIONAL_INPUT,
        use_additional_input=True,
        extra_wrappers=[(LavaMazeRewardShaping, {}), LAVA_MAZE_ORACLE_MSG_ACTION_MASK],
        shared_reward_alpha=1.0,
        joint_wrappers=[(LavaMazeOracleMessageWrapper, {})],
    ),

    DoomSpec(
        # communication removed: navigator measurements carry the BFS next-step direction
        "lava_maze_oracle_hint",
        "lava_maze.cfg",
        doom_action_space_lava_maze(),
        1.0,
        5250,
        num_agents=2,
        forcerespawn=0,
        nofreelook=0,
        additional_input_wrapper=LAVA_MAZE_ORACLE_HINT_INPUT,
        use_additional_input=True,
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

    # Learnability references on the original Stealth Labyrinth (paper App. very_difficult)
    DoomSpec(
        # ACS-driven scout (player 0, hidden from SF), learned gunner (player 1)
        "stealth_labyrinth_scripted_scout",
        "stealth_labyrinth_scripted_scout.cfg",
        doom_action_space_stealth_labyrinth(),
        1.0,
        5250,
        num_agents=2,
        forcerespawn=0,
        extra_wrappers=[(StealthLabyrinthRewardShaping, {})],
        shared_reward_alpha=1.0,
        joint_wrappers=[(ScriptedPartnerWrapper, {"learner_index": 1, "partner_policy": NoopPartnerPolicy})],
    ),

    DoomSpec(
        # converse: ACS-driven gunner (player 1, hidden from SF) trails the scout and shoots lit turrets; learned scout
        "stealth_labyrinth_scripted_gunner",
        "stealth_labyrinth_scripted_gunner.cfg",
        doom_action_space_stealth_labyrinth(),
        1.0,
        5250,
        num_agents=2,
        forcerespawn=0,
        extra_wrappers=[(StealthLabyrinthRewardShaping, {})],
        shared_reward_alpha=1.0,
        joint_wrappers=[(ScriptedPartnerWrapper, {"learner_index": 0, "partner_policy": NoopPartnerPolicy})],
    ),

    DoomSpec(
        # rooms/corridors permanently lit; turrets still invulnerable until the scout lights the active room
        "stealth_labyrinth_lit",
        "stealth_labyrinth_lit.cfg",
        doom_action_space_stealth_labyrinth(),
        1.0,
        5250,
        num_agents=2,
        forcerespawn=0,
        extra_wrappers=[(StealthLabyrinthRewardShaping, {})],
        shared_reward_alpha=1.0,
    ),

    DoomSpec(
        # each agent observes [own frame, partner frame] stacked along channels
        "stealth_labyrinth_shared_vision",
        "stealth_labyrinth.cfg",
        doom_action_space_stealth_labyrinth(),
        1.0,
        5250,
        num_agents=2,
        forcerespawn=0,
        extra_wrappers=[(StealthLabyrinthRewardShaping, {})],
        shared_reward_alpha=1.0,
        joint_wrappers=[(SharedVisionWrapper, {})],
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


# One-line description and benchmark score (the episode `true_objective`) for every scenario in DOOM_ENVS.
# Used by comrad.list_scenarios() and the docs.
SCENARIO_METADATA = {
    "stag_hunt_arena": ("Stag-hunt dilemma: safe solo rabbits vs. a larger stag that needs joint commitment.", "Stags killed"),
    "stag_hunt_arena_continuous": ("Stag Hunt Arena with a continuous turning action.", "Stags killed"),
    "stag_hunt_arena_continuous_full": ("Stag Hunt Arena with continuous turning and the full button set.", "Stags killed"),
    "rhythm_sync": ("Separated agents must hit paired switches inside shared timing windows.", "Stages completed"),
    "rhythm_sync_dense": ("Rhythm Sync with dense reward shaping.", "Stages completed"),
    "foraging_commons": ("Harvesting depletes a shared resource; cleanup restores it at a personal cost.", "Survival time"),
    "coop_puzzle": ("Paired lanes where players alternately unlock each other's path.", "Completed puzzle pairs"),
    "platform_chain": ("Tethered players climb a long vertical route together.", "Joint checkpoint progress"),
    "platform_chain_easy": ("Shorter, easier Platform Chain variant for debugging and warm-up.", "Joint checkpoint progress"),
    "armory_siege": ("Defend a central core while rotating through gated supply rooms.", "Shaped episodic return"),
    "coop_health_gathering": ("Tethered survival in a toxic maze with shared health kits.", "Survival time"),
    "lavapit": ("Reciprocal bridge handoffs: one agent holds support while the other crosses.", "Joint bridges cleared"),
    "smart_enemies": ("Fleeing enemies get harder to finish when agents crowd the same target.", "Enemies killed"),
    "dumb_enemies": ("Same arena as Smart Enemies, but coordinated pressure makes enemies easier.", "Enemies killed"),
    "ammo_carrier": ("Fixed roles: an immobile defender kept supplied with ammo by a runner.", "Survival time"),
    "stealth_labyrinth": ("A non-attacking scout lights rooms for a gunner who depends on its vision.", "Enemies destroyed / total"),
    "lava_maze": ("A navigator crosses lava guided by colour signals from an overhead observer.", "Maze levels cleared"),
    "lava_maze_simple": ("Smaller, easier Lava Maze variant for debugging and warm-up.", "Maze levels cleared"),
    # learnability references (paper App. very_difficult): one role scripted and hidden, or coordination removed
    "lava_maze_scripted_observer": ("Lava Maze with a scripted BFS observer; only the navigator learns.", "Maze levels cleared"),
    "lava_maze_scripted_navigator": ("Lava Maze with a scripted navigator obeying a fixed colour convention; only the observer learns.", "Maze levels cleared"),
    "lava_maze_oracle_msg": ("Lava Maze with a lossless discrete message channel from observer to navigator.", "Maze levels cleared"),
    "lava_maze_oracle_hint": ("Lava Maze where the navigator receives the next safe direction directly.", "Maze levels cleared"),
    "stealth_labyrinth_scripted_scout": ("Stealth Labyrinth with a scripted scout; only the gunner learns.", "Enemies destroyed / total"),
    "stealth_labyrinth_scripted_gunner": ("Stealth Labyrinth with a scripted gunner that trails the scout; only the scout learns.", "Enemies destroyed / total"),
    "stealth_labyrinth_lit": ("Stealth Labyrinth with permanently lit rooms; turrets stay scout-gated.", "Enemies destroyed / total"),
    "stealth_labyrinth_shared_vision": ("Stealth Labyrinth where each agent also sees its partner's frame.", "Enemies destroyed / total"),
}


def doom_env_by_name(name):
    for cfg in DOOM_ENVS:
        if cfg.name == name:
            return cfg
    raise RuntimeError("Unknown Doom env")

def batch_base_env_name(cfg) -> str:
    base_env = getattr(cfg, "_batch_base_env", None)
    if base_env:
        return base_env

    env_name = str(cfg.env)
    if env_name.endswith("_batch"):
        return env_name[: -len("_batch")]
    return env_name


def get_num_agents(cfg, env_name):
    spec = doom_env_by_name(env_name)
    return spec.num_agents if cfg.num_agents <= 0 else cfg.num_agents


def get_num_players(cfg, env_name) -> int:
    """Players in the ViZDoom game: the learner agents plus the scenario's scripted players."""
    return get_num_agents(cfg, env_name) + doom_env_by_name(env_name).num_scripted_players

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
    if override is not None:
        return override
    # Only the 1D state modality pulls in the scenario's task scalars by default. Full observability must NOT
    # enable them implicitly, otherwise Pixel/Full gets hand-crafted features that Pixel/Partial lacks
    # (pass --use_additional_input=True explicitly to reproduce that variant).
    if doom_spec.additional_input_wrapper is not None and str(getattr(cfg, "obs_modality", "pixel")).lower() == "state":
        return True
    return doom_spec.use_additional_input


FULL_OBS_SHARED_VISION = "shared_vision"
FULL_OBS_GLOBAL_STATE = "global_state"


def get_full_obs_mode(cfg) -> Optional[str]:
    """None when full observability is off, else the mode (archived configs without the key are shared vision)."""
    if not bool(getattr(cfg, "full_observability", False)):
        return None
    return str(getattr(cfg, "full_obs_mode", None) or FULL_OBS_SHARED_VISION).lower()


def get_global_state_version(cfg) -> int:
    """Encoding version of --full_obs_mode=global_state (see doom_params; archived runs are resolved by the evaluator)."""
    return int(getattr(cfg, "global_state_version", None) or 2)


def restrict_global_state_input_normalization(cfg) -> None:
    """Global-state v2 training normalizes only the image: the state vector is already scaled, and running statistics
    of inputs that are constant on the training layouts (empty object slots, static walls) turn any held-out change
    into a saturated input. An explicit --normalize_input_keys is kept."""
    if get_full_obs_mode(cfg) != FULL_OBS_GLOBAL_STATE or get_global_state_version(cfg) < 2:
        return
    if getattr(cfg, "normalize_input", False) and getattr(cfg, "normalize_input_keys", None) is None:
        log.info("global_state v2: normalizing only the 'obs' image, not the global state in 'measurements'")
        cfg.normalize_input_keys = ["obs"]


STATE_ENCODING_EGO = "ego"
STATE_ENCODING_ENTITY = "entity"


def get_state_encoding(cfg) -> str:
    """State-vector encoding for --obs_modality=state (archived configs without the key are 'ego')."""
    return str(getattr(cfg, "state_encoding", None) or STATE_ENCODING_EGO).lower()


def uses_entity_state_encoding(cfg) -> bool:
    obs_modality = str(getattr(cfg, "obs_modality", "pixel") or "pixel").lower()
    return obs_modality == "state" and get_state_encoding(cfg) == STATE_ENCODING_ENTITY


DOOM_TICRATE = 35
CLOCK_EGO = "ego"
CLOCK_JOINT = "joint"
OBSERVATION_CLOCKS = (CLOCK_EGO, CLOCK_JOINT)


def get_observation_clock_version(cfg) -> int:
    """Observation clock version (see doom_params; archived runs are resolved by resolve_archived_cfg)."""
    version = getattr(cfg, "observation_clock_version", None)
    if version is None:
        raise ValueError(
            "cfg has no observation_clock_version: build it with the comrad parser, or load archived configs through "
            "load_from_checkpoint so resolve_archived_cfg can pin their clock"
        )
    return int(version)


def uses_multiplayer_game(doom_spec) -> bool:
    """Whether the env runs a ViZDoom multiplayer game (host + joiners) rather than a singleplayer game."""
    return doom_spec.num_players > 1 or doom_spec.num_bots > 0


def scenario_cfg_episode_timeout(doom_spec) -> int:
    """`episode_timeout` (tics) of the scenario cfg file; 0 when the file sets none."""
    timeout = 0
    with open(doom_spec_config_path(doom_spec), "r") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if "=" in line and line.split("=", 1)[0].strip().lower().replace("_", "") == "episodetimeout":
                timeout = int(float(line.split("=", 1)[1].strip()))
    return timeout


def _timelimit_minutes(doom_spec, cfg) -> float:
    return cfg.timelimit if getattr(cfg, "timelimit", None) is not None else doom_spec.timelimit


def _frameskip(cfg) -> int:
    frameskip = int(cfg.env_frameskip)
    if frameskip < 1:
        raise ValueError(f"env_frameskip must be >= 1, got {frameskip}")
    return frameskip


def episode_horizon_steps(doom_spec, cfg) -> int:
    """Joint env steps after which an env built with `cfg` truncates the episode.

    The earlier of the TimeLimitWrapper limit (`default_timeout` joint steps) and the engine timeout in tics over
    the frameskip. Multiplayer games replace the scenario cfg `episode_timeout` with `timelimit` minutes
    (VizdoomEnvMultiplayer); singleplayer games keep the cfg value.
    """
    limits = []
    if doom_spec.default_timeout > 0:
        limits.append(int(doom_spec.default_timeout))
    if uses_multiplayer_game(doom_spec):
        engine_tics = int(_timelimit_minutes(doom_spec, cfg) * 60 * DOOM_TICRATE)
    else:
        engine_tics = scenario_cfg_episode_timeout(doom_spec)
    if engine_tics > 0:
        limits.append(-(-engine_tics // _frameskip(cfg)))
    if not limits:
        raise ValueError(f"{doom_spec.name}: episodes have neither a step limit nor an engine timeout")
    return min(limits)


def _clock_horizon_v1(doom_spec, cfg, clock: str) -> float:
    frameskip = _frameskip(cfg)
    if clock == CLOCK_EGO:
        if doom_spec.default_timeout <= 0:
            raise ValueError(f"{doom_spec.name}: observation clock version 1 needs a step limit (default_timeout)")
        num_agents = doom_spec.num_agents if cfg.num_agents <= 0 else cfg.num_agents
        # the per-player wrappers counted every frame the multi-agent wrapper skips
        frames_per_step = frameskip if num_agents + doom_spec.num_scripted_players > 1 else 1
        return doom_spec.default_timeout / frames_per_step
    tics = [scenario_cfg_episode_timeout(doom_spec), int(_timelimit_minutes(doom_spec, cfg) * 60 * DOOM_TICRATE)]
    tics = [t for t in tics if t > 0]
    if not tics:
        raise ValueError(f"{doom_spec.name}: observation clock version 1 needs an episode_timeout or a timelimit")
    return float(-(-min(tics) // frameskip))


def observation_clock_horizon(doom_spec, cfg, clock: str) -> float:
    """Joint env steps at which an observation clock `min(1, step / horizon)` reaches 1.

    `clock` is CLOCK_EGO (StateObservationWrapper, BlindObservationWrapper) or CLOCK_JOINT (GlobalStateWrapper,
    EntityStateWrapper). Version 2 clocks run on `episode_horizon_steps`; version 1 reproduces the clocks archived
    checkpoints were trained with.
    """
    if clock not in OBSERVATION_CLOCKS:
        raise ValueError(f"Unknown observation clock {clock!r}, expected one of {OBSERVATION_CLOCKS}")
    version = get_observation_clock_version(cfg)
    if version == 2:
        return float(episode_horizon_steps(doom_spec, cfg))
    if version == 1:
        return _clock_horizon_v1(doom_spec, cfg, clock)
    raise ValueError(f"Unknown observation_clock_version {version}")


def cue_observation_layout(cfg, doom_spec) -> str:
    """Layout of the per-agent observation the partner-cue masking of VisibilityDiagnosticsWrapper acts on.

    Pixel and blind observations of a single agent carry partner cues only in the scenario `measurements` (or none
    without them); global-state, state and shared-vision observations re-encode them under another layout.
    """
    obs_modality = str(getattr(cfg, "obs_modality", "pixel") or "pixel").lower()
    full_obs_mode = get_full_obs_mode(cfg)
    if any(issubclass(w_cls, SharedVisionWrapper) for w_cls, _ in (doom_spec.joint_wrappers or [])):
        full_obs_mode = FULL_OBS_SHARED_VISION
    has_scenario_measurements = doom_spec.additional_input_wrapper is not None and get_use_additional_input(cfg, doom_spec)
    if obs_modality in ("pixel", "blind") and full_obs_mode is None:
        return CUE_LAYOUT_SCENARIO_MEASUREMENTS if has_scenario_measurements else CUE_LAYOUT_NO_CUE_CHANNEL
    return f"obs_modality={obs_modality}, full_obs_mode={full_obs_mode}, scenario_measurements={has_scenario_measurements}"


def doom_spec_config_path(doom_spec) -> str:
    if os.path.isabs(doom_spec.env_spec_file):
        return doom_spec.env_spec_file
    return join(os.path.dirname(__file__), os.pardir, "scenarios", doom_spec.env_spec_file)


def resolve_archived_use_additional_input(archived_cfg: dict):
    """Value of --use_additional_input that reproduces an archived run's observation space.

    Runs trained before full observability stopped implicitly enabling the scenario's task scalars stored
    ``use_additional_input=None`` together with ``full_observability=True``; their checkpoints expect the wider
    measurement vector, so ``None`` must be resolved to ``True`` when evaluating them. The clean runs pass the flag
    explicitly, so an explicit value is always returned unchanged.
    """
    value = archived_cfg.get("use_additional_input")
    if value is not None:
        return bool(value)
    if bool(archived_cfg.get("full_observability", False)):
        return True
    return None


def resolve_archived_num_agents(archived_cfg: dict):
    """Value of --num_agents (learner agents) that reproduces an archived run.

    Runs of scripted-partner scenarios trained before learner and physical player counts were separated stored the
    physical player count; it is resolved to the learner count. Any other value is returned unchanged.
    """
    num_agents = archived_cfg.get("num_agents")
    if "env" not in archived_cfg:
        return num_agents
    env_name = batch_base_env_name(AttrDict(archived_cfg))
    spec = next((s for s in DOOM_ENVS if s.name == env_name), None)
    if spec is not None and spec.num_scripted_players > 0 and num_agents == spec.num_players:
        return spec.num_agents
    return num_agents


def resolve_archived_cfg(archived_cfg: dict) -> dict:
    """Archived config.json resolved to the current config semantics (applied by load_from_checkpoint)."""
    resolved = dict(archived_cfg)
    if "num_agents" in resolved:
        resolved["num_agents"] = resolve_archived_num_agents(archived_cfg)
    if resolved.get("observation_clock_version") is None:
        resolved["observation_clock_version"] = 1
    return resolved


def get_extra_wrappers(cfg, doom_spec):
    wrappers = []
    use_additional_input = get_use_additional_input(cfg, doom_spec)

    if doom_spec.additional_input_wrapper is not None and use_additional_input:
        w_cls, w_kwargs = doom_spec.additional_input_wrapper
        if w_cls is RhythmSyncAdditionalInput and get_full_obs_mode(cfg) == FULL_OBS_SHARED_VISION:
            w_kwargs = dict(w_kwargs)
            w_kwargs["feature_set"] = "full"
        wrappers.append((w_cls, w_kwargs))
    elif use_additional_input and doom_spec.additional_input_wrapper is None:
        if doom_spec.name not in _WARNED_UNSUPPORTED_ADDITIONAL_INPUT_ENVS and getattr(cfg, "use_additional_input", None) is True:
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

    enable_labels_buffer = bool(
        getattr(cfg, "enable_labels_buffer", False)
        or str(getattr(cfg, "partner_occlusion_mode", "none")).lower() in ("sprite", "both")
    )
    # objects and layout are read from a single (synchronized) game instance: the host
    global_state_mode = get_full_obs_mode(cfg) == FULL_OBS_GLOBAL_STATE
    entity_state = uses_entity_state_encoding(cfg)
    enable_global_state_info = (global_state_mode or entity_state) and player_id in (None, 0)
    # the entity encoding only needs objects; wall lines and sector heights stay with the global-state mode
    global_state_geometry = global_state_mode
    enable_depth_row = entity_state

    if player_id is None:
        env = VizdoomEnv(
            doom_spec.action_space,
            doom_spec.env_spec_file,
            skip_frames=skip_frames,
            async_mode=async_mode,
            render_mode=render_mode,
            enable_labels_buffer=enable_labels_buffer,
            enable_global_state_info=enable_global_state_info,
            global_state_geometry=global_state_geometry,
            enable_depth_row=enable_depth_row,
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
            # set by comrad.utils.record_demo_video: host writes a native ViZDoom .lmp demo for high-res replay
            record_to=cfg.get("demo_record_to") if hasattr(cfg, "get") else getattr(cfg, "demo_record_to", None),
            render_mode=render_mode,
            host_ip=getattr(cfg, "host_ip", "127.0.0.1"),
            enable_labels_buffer=enable_labels_buffer,
            enable_global_state_info=enable_global_state_info,
            global_state_geometry=global_state_geometry,
            enable_depth_row=enable_depth_row,
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
        # set by comrad.record_video so the render env draws natively at the video resolution instead of upscaling
        resolution = cfg.get("native_render_resolution") if hasattr(cfg, "get") else getattr(cfg, "native_render_resolution", None)
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
    if doom_spec.default_timeout > 0:
        #TODO: for TimeLimitWrapper, random_variation_steps may be set to a proper value
        env = TimeLimitWrapper(env, limit=doom_spec.default_timeout, random_variation_steps=0)

    pixel_format = cfg.pixel_format if "pixel_format" in cfg else "HWC"
    if pixel_format == "CHW":
        env = PixelFormatChwWrapper(env)

    for wrapper_cls, wrapper_kwargs in get_extra_wrappers(cfg, doom_spec):
        env = wrapper_cls(env, **wrapper_kwargs)

    obs_modality = str(getattr(cfg, "obs_modality", "pixel") or "pixel").lower()
    if obs_modality == "state":
        state_encoding = get_state_encoding(cfg)
        env = StateObservationWrapper(
            env,
            scenario_name=doom_spec.name,
            include_global_state=bool(getattr(cfg, "full_observability", False)),
            horizon=None if state_encoding == STATE_ENCODING_ENTITY else observation_clock_horizon(doom_spec, cfg, CLOCK_EGO),
            state_encoding=state_encoding,
        )
    elif obs_modality == "blind":
        env = BlindObservationWrapper(env, horizon=observation_clock_horizon(doom_spec, cfg, CLOCK_EGO))

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


def make_doom_multiplayer_env(
    doom_spec, cfg=None, env_config=None, render_mode: Optional[str] = None, auto_reset: bool = True, **kwargs
):
    assert cfg is not None

    skip_frames = cfg.env_frameskip

    if cfg.num_bots < 0:
        num_bots = doom_spec.num_bots
    else:
        num_bots = cfg.num_bots

    num_agents = doom_spec.num_agents if cfg.num_agents <= 0 else cfg.num_agents
    num_players = num_agents + doom_spec.num_scripted_players
    max_num_players = num_players + cfg.num_humans

    is_multiagent = num_players > 1

    def make_env_func(player_id):
        return make_doom_env_impl(
            doom_spec,
            cfg=cfg,
            player_id=player_id,
            num_agents=num_players,
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
            num_agents=num_players,
            make_env_func=make_env_func,
            env_config=env_config,
            skip_frames=skip_frames,
            render_mode=render_mode,
            auto_reset=auto_reset,
        )

        shared_reward_alpha = get_alpha(cfg, doom_spec)
        shared_reward_scalarisation = get_scalarisation(cfg, doom_spec)
        use_shared_reward = shared_reward_alpha > 0 and str(getattr(cfg, "algo", "MAPPO")).upper() not in OFF_POLICY
        shared_reward_applied = False

        for wrapper_cls, wrapper_kwargs in (doom_spec.joint_wrappers or []):
            if getattr(wrapper_cls, "scripted_players", 0) > 0 and use_shared_reward and not shared_reward_applied:
                # blend rewards while every player is still visible, so a single learner behind a scripted partner
                # receives exactly the benchmark reward (team reward for on-policy algos)
                env = SharedRewardWrapper(env, alpha=shared_reward_alpha, scalarisation=shared_reward_scalarisation)
                shared_reward_applied = True
            env = wrapper_cls(env, **wrapper_kwargs)
        if env.num_agents != num_agents:
            raise ValueError(
                f"{doom_spec.name}: the wrapped env exposes {env.num_agents} learner agents but the config expects "
                f"{num_agents}; --num_agents counts learner agents only (scripted players: {doom_spec.num_scripted_players})"
            )

        has_spec_shared_vision = any(
            issubclass(w_cls, SharedVisionWrapper) for w_cls, _ in (doom_spec.joint_wrappers or [])
        )
        full_obs_mode = get_full_obs_mode(cfg)
        if uses_entity_state_encoding(cfg):
            if full_obs_mode is not None:
                raise ValueError("--state_encoding=entity cannot be combined with --full_observability=True")
            env = EntityStateWrapper(
                env,
                scenario_name=doom_spec.name,
                horizon=observation_clock_horizon(doom_spec, cfg, CLOCK_JOINT),
                sight_range=float(getattr(cfg, "state_sight_range", 512.0) or 512.0),
            )
        if full_obs_mode == FULL_OBS_GLOBAL_STATE:
            env = GlobalStateWrapper(
                env,
                horizon=observation_clock_horizon(doom_spec, cfg, CLOCK_JOINT),
                game_variables=VizdoomEnv._parse_variable_indices(doom_spec_config_path(doom_spec)).keys(),
                version=get_global_state_version(cfg),
            )
        elif full_obs_mode == FULL_OBS_SHARED_VISION and not has_spec_shared_vision and num_agents == 2:
            env = SharedVisionWrapper(
                env,
                stack_measurements=True,
                include_relative_pose=True,
            )

        partner_occlusion_mode = str(getattr(cfg, "partner_occlusion_mode", "none") or "none").lower()
        if (bool(getattr(cfg, "enable_labels_buffer", False)) or partner_occlusion_mode != "none") and num_agents == 2:
            env = VisibilityDiagnosticsWrapper(
                env,
                scenario_name=doom_spec.name,
                partner_occlusion_mode=partner_occlusion_mode,
                cue_layout=cue_observation_layout(cfg, doom_spec),
            )

        # For HAPPO, obs_space with temp env (player_id=-1) doesn't get AgentIDWrapper, so MultiAgentEnv.observation_space is missing 'agent_id'
        # SF uses this obs_space to create the model, so we augment it manually here
        # But that setting is only for player testing so shouldn't matter much
        if str(getattr(cfg, 'algo', 'APPO')).upper() == 'HAPPO':
            import gymnasium as gym
            import numpy as np
            spaces = dict(env.observation_space.spaces) if isinstance(env.observation_space, gym.spaces.Dict) else {'obs': env.observation_space}
            spaces['agent_id'] = gym.spaces.Box(low=0.0, high=1.0, shape=(num_players,), dtype=np.float32)
            env.observation_space = gym.spaces.Dict(spaces)

        if use_shared_reward and not shared_reward_applied and num_agents > 1:
            env = SharedRewardWrapper(
                env,
                alpha=shared_reward_alpha,
                scalarisation=shared_reward_scalarisation,
            )
    else:
        if uses_entity_state_encoding(cfg):
            raise ValueError("--state_encoding=entity needs a multi-agent env (partner and host object info)")
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
                done_mode="any" if is_multiagent and (algo_name in ON_POLICY or getattr(env, "num_agents", 1) == 1) else "all",
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

    if uses_multiplayer_game(spec):
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
