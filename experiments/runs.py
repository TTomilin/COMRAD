"""Sample Factory launcher entry point for the experiment profiles (benchmark profiles live in comrad.train_all).

    COMRAD_EXPERIMENT_PROFILE=<profile> python -m sample_factory.launcher.run --run=experiments.runs \
        --backend=processes --max_parallel=1 --pause_between=1
"""
import os

from sample_factory.launcher.run_description import RunDescription

from experiments.iclr2027_rebuttal import profiles as rebuttal_profiles
from experiments.very_difficult import profiles as very_difficult_profiles

PROFILES = {**very_difficult_profiles.PROFILES, **rebuttal_profiles.PROFILES}


def _active_profile() -> str:
    profile = os.environ.get("COMRAD_EXPERIMENT_PROFILE", "").strip().lower()
    if profile not in PROFILES:
        raise ValueError(f"Unsupported COMRAD_EXPERIMENT_PROFILE={profile!r}. Expected one of {sorted(PROFILES)}.")
    return profile


ACTIVE_PROFILE = _active_profile()
BENCHMARK_NAME, _build_experiments = PROFILES[ACTIVE_PROFILE]
_experiments = _build_experiments()

TOTAL_RUNS = sum(max(1, len(experiment.params)) for experiment in _experiments)

RUN_DESCRIPTION = RunDescription(
    os.environ.get("COMRAD_RUN_NAME", BENCHMARK_NAME),
    experiments=_experiments,
)
