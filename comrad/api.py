"""Framework-agnostic public API for COMRAD.

    import comrad
    env = comrad.make("stag_hunt_arena")            # Gymnasium env, per-agent lists
    env = comrad.parallel_env("stag_hunt_arena")    # PettingZoo ParallelEnv, per-agent dicts

Both build the exact same environment stack as `python -m comrad.train` (same scenario wrappers, reward shaping and
default shared-reward blend), but without a Sample Factory config and with explicit-reset episode semantics.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from comrad.utils.doom_utils import DOOM_ENVS, SCENARIO_METADATA, doom_env_by_name, make_doom_multiplayer_env

RENDER_MODES = (None, "human", "rgb_array")
PIXEL_FORMATS = ("HWC", "CHW")


@dataclass(frozen=True)
class ScenarioInfo:
    name: str
    num_agents: int
    description: str
    score: str


def list_scenarios() -> List[ScenarioInfo]:
    """All registered COMRAD scenarios with their default number of agents and benchmark score metric."""
    scenarios = []
    for spec in DOOM_ENVS:
        description, score = SCENARIO_METADATA.get(spec.name, ("", "true_objective"))
        scenarios.append(ScenarioInfo(spec.name, spec.num_agents, description, score))
    return scenarios


def _scenario_spec(scenario: str):
    try:
        return doom_env_by_name(scenario)
    except RuntimeError:
        names = ", ".join(spec.name for spec in DOOM_ENVS)
        raise ValueError(f"Unknown COMRAD scenario {scenario!r}. Available scenarios: {names}") from None


def _build_cfg(
    scenario: str,
    num_agents: Optional[int] = None,
    frameskip: int = 4,
    resolution: Tuple[int, int] = (128, 72),
    pixel_format: str = "HWC",
    shared_reward: Optional[float] = None,
    additional_input: bool = False,
):
    """Translate public keyword options into the config object consumed by the COMRAD env factory."""
    from comrad.envs.doom_params import default_doom_cfg

    spec = _scenario_spec(scenario)

    if num_agents is None:
        num_agents = spec.num_agents
    if int(num_agents) < 2:
        raise ValueError(f"COMRAD scenarios are multi-agent, num_agents must be >= 2 (got {num_agents})")
    if int(frameskip) < 1:
        raise ValueError(f"frameskip must be >= 1 (got {frameskip})")
    if len(resolution) != 2 or min(resolution) < 1:
        raise ValueError(f"resolution must be a (width, height) pair of positive ints (got {resolution!r})")
    if pixel_format not in PIXEL_FORMATS:
        raise ValueError(f"pixel_format must be one of {PIXEL_FORMATS} (got {pixel_format!r})")
    if shared_reward is not None and not 0.0 <= float(shared_reward) <= 1.0:
        raise ValueError(f"shared_reward must be None or in [0, 1] (got {shared_reward})")

    # APPO is on-policy and not HAPPO: SharedRewardWrapper follows `shared_reward_alpha`, no agent-id observation.
    cfg = default_doom_cfg(algo="APPO", env=scenario, experiment="comrad_api")
    cfg.num_agents = int(num_agents)
    cfg.env_frameskip = int(frameskip)
    cfg.res_w, cfg.res_h = int(resolution[0]), int(resolution[1])
    cfg.pixel_format = pixel_format
    cfg.shared_reward_alpha = None if shared_reward is None else float(shared_reward)
    cfg.use_additional_input = bool(additional_input)
    cfg.with_wandb = False
    cfg.wandb_record_every = 0
    cfg.record_to = None
    return spec, cfg


def _build_env(scenario: str, render_mode: Optional[str] = None, **options):
    if render_mode not in RENDER_MODES:
        raise ValueError(f"render_mode must be one of {RENDER_MODES} (got {render_mode!r})")
    spec, cfg = _build_cfg(scenario, **options)
    env = make_doom_multiplayer_env(spec, cfg=cfg, env_config=None, render_mode=render_mode, auto_reset=False)
    if render_mode is not None and hasattr(env.unwrapped, "enable_rendering"):
        env.unwrapped.enable_rendering = True  # keep the latest frames so render() works right after reset()
    return env


def make(
    scenario: str,
    *,
    num_agents: Optional[int] = None,
    frameskip: int = 4,
    resolution: Tuple[int, int] = (128, 72),
    pixel_format: str = "HWC",
    shared_reward: Optional[float] = None,
    additional_input: bool = False,
    render_mode: Optional[str] = None,
):
    """Create a COMRAD scenario as a Gymnasium env.

    `reset()` returns `(obs_list, info_list)` and `step(action_list)` returns
    `(obs_list, reward_list, terminated_list, truncated_list, info_list)`, one entry per agent.
    `observation_space` / `action_space` describe a single agent. The env never resets itself:
    call `reset()` once every agent is terminated or truncated.
    """
    return _build_env(
        scenario,
        render_mode=render_mode,
        num_agents=num_agents,
        frameskip=frameskip,
        resolution=resolution,
        pixel_format=pixel_format,
        shared_reward=shared_reward,
        additional_input=additional_input,
    )


def parallel_env(
    scenario: str,
    *,
    num_agents: Optional[int] = None,
    frameskip: int = 4,
    resolution: Tuple[int, int] = (128, 72),
    pixel_format: str = "HWC",
    shared_reward: Optional[float] = None,
    additional_input: bool = False,
    render_mode: Optional[str] = None,
):
    """Create a COMRAD scenario as a PettingZoo `ParallelEnv` with agents `player_0 .. player_{N-1}`.

    Takes the same options as `make()`.
    """
    from comrad.envs.pettingzoo_env import ComradParallelEnv

    env = make(
        scenario,
        num_agents=num_agents,
        frameskip=frameskip,
        resolution=resolution,
        pixel_format=pixel_format,
        shared_reward=shared_reward,
        additional_input=additional_input,
        render_mode=render_mode,
    )
    return ComradParallelEnv(env, scenario=scenario)
