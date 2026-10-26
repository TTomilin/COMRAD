"""COMRAD: cooperative multi-agent RL benchmark in ViZDoom."""

from importlib.metadata import PackageNotFoundError, version

from comrad.api import ScenarioInfo, list_scenarios, make, parallel_env

try:
    __version__ = version("comrad-bench")
except PackageNotFoundError:  # running from a source checkout without an install
    __version__ = "unknown"

__all__ = ["ScenarioInfo", "list_scenarios", "make", "parallel_env"]


def _register_gymnasium_envs():
    import gymnasium as gym

    for scenario in list_scenarios():
        env_id = f"comrad/{scenario.name}-v0"
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point="comrad.api:make",
            kwargs={"scenario": scenario.name},
            disable_env_checker=True,  # observations are per-agent lists, not single-agent arrays
        )


_register_gymnasium_envs()
