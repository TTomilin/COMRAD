"""Training profiles of the ICLR 2027 rebuttal studies (reviewers kd6J and C2K6, see README.md).

Launched through `experiments.runs` (see experiments/README.md).
"""
import os

from sample_factory.launcher.run_description import Experiment, ParamGrid

from comrad.train_all import BENCHMARK_SCENARIOS, AlgorithmSpec, BenchmarkScenario, _command_for_scenario
from experiments.common import CANONICAL_STEALTH_QMIX_REPLAY_BUFFER, algorithm_by_name, with_train_steps

FULL_OBSERVABILITY_NAME = "comrad_full_observability"
OBS_MODALITY_2X2_NAME = "comrad_obs_modality_2x2"
BLIND_ABLATION_NAME = "comrad_blind_ablation"
VISIBILITY_TELEMETRY_NAME = "comrad_visibility_telemetry"
FULL_OBS_CLEAN_NAME = "comrad_full_obs_clean"
FULL_OBS_GLOBAL_NAME = "comrad_full_obs_global"
FULL_OBS_GLOBAL_V2_NAME = "comrad_full_obs_global_v2"
STATE_ENTITY_NAME = "comrad_state_entity"
FROZEN_PARTNER_WINNERS_NAME = "comrad_frozen_partner_winners"
MASK_PARTNER_TRAINING_NAME = "comrad_mask_partner_training"

FULL_OBSERVABILITY_ROOT_DIR = "full_observability"
OBS_MODALITY_2X2_ROOT_DIR = "obs_modality_2x2"
BLIND_ABLATION_ROOT_DIR = "blind_ablation"
REBUTTAL_KD6J_SEEDS = [
    int(s.strip())
    # 2 seeds to save GPU hours; aggregation reports mean +/- 95% Student-t CI with df=1 (t=12.706).
    for s in os.environ.get("COMRAD_KD6J_SEEDS", "42,68").split(",")
    if s.strip()
]
# Every kd6J cell (incl. the retrained Pixel-Partial baseline) shares this budget, so comparisons stay paired.
REBUTTAL_KD6J_TRAIN_FOR_ENV_STEPS = int(os.environ.get("COMRAD_KD6J_TRAIN_STEPS", "50000000"))
REBUTTAL_KD6J_ALGORITHMS = ["MAPPO", "QMIX"]
# SMACv2-style blind diagnostic uses a single (on-policy) learner, as in Ellis et al.
BLIND_ABLATION_ALGORITHMS = ["MAPPO"]
VISIBILITY_TELEMETRY_ROOT_DIR = "visibility_telemetry"
VISIBILITY_TELEMETRY_SEEDS = [42]
REPRESENTATIVE_2X2_ENV_NAMES = (
    "rhythm_sync_dense",
    "armory_siege",
    "ammo_carrier",
    "stealth_labyrinth",
)
REPRESENTATIVE_2X2_SCENARIOS = [
    scenario for scenario in BENCHMARK_SCENARIOS if scenario.env in REPRESENTATIVE_2X2_ENV_NAMES
]
OBS_MODALITY_2X2_CELLS = (
    ("pixel", False, "pixel_partial"),
    ("pixel", True, "pixel_full"),
    ("state", False, "state_partial"),
    ("state", True, "state_full"),
)


def _kd6j_params_for_scenario(scenario: BenchmarkScenario, algorithm: AlgorithmSpec, seeds=None):
    seed_list = REBUTTAL_KD6J_SEEDS if seeds is None else list(seeds)
    if algorithm.uses_shared_reward:
        shared_reward_alpha = 1.0 if scenario.actor_critic_shared_reward else 0.0
        return ParamGrid([("shared_reward_alpha", [shared_reward_alpha]), ("seed", seed_list)])
    return ParamGrid([("seed", seed_list)])


def _kd6j_base_cmd(
    scenario: BenchmarkScenario,
    algorithm: AlgorithmSpec,
    *,
    obs_modality: str = "pixel",
    full_observability: bool = False,
    train_for_env_steps: int = REBUTTAL_KD6J_TRAIN_FOR_ENV_STEPS,
) -> str:
    cmd = with_train_steps(_command_for_scenario(scenario, algorithm), train_for_env_steps)
    extra = [
        f"--obs_modality={obs_modality}",
        f"--full_observability={str(bool(full_observability))}",
    ]
    if obs_modality == "state":
        extra.append("--obs_scale=1.0")
    # Pin every kd6J QMIX cell to the canonical 3-channel capacity so replay size never confounds the ablation.
    if algorithm.name == "QMIX":
        extra.append(f"--replay_buffer_size={CANONICAL_STEALTH_QMIX_REPLAY_BUFFER}")
    return f"{cmd} {' '.join(extra)}"


# Pixel-Partial vs Pixel-Full at the same 100M budget for the scenarios not already covered by the 2x2 profile
# (whose pixel_partial / pixel_full cells complete the 13-scenario Full-Obs column of normalized Table 3).
FULL_OBSERVABILITY_SCENARIOS = [
    scenario for scenario in BENCHMARK_SCENARIOS if scenario.env not in REPRESENTATIVE_2X2_ENV_NAMES
]
FULL_OBSERVABILITY_CELLS = (
    (False, "pixel_partial"),
    (True, "pixel_full"),
)


def full_observability_experiments():
    experiments = []
    for scenario in FULL_OBSERVABILITY_SCENARIOS:
        for algorithm in (algorithm_by_name(name) for name in REBUTTAL_KD6J_ALGORITHMS):
            params = _kd6j_params_for_scenario(scenario, algorithm)
            for full_obs, cell_tag in FULL_OBSERVABILITY_CELLS:
                cmd = _kd6j_base_cmd(
                    scenario,
                    algorithm,
                    obs_modality="pixel",
                    full_observability=full_obs,
                )
                experiments.append(
                    Experiment(
                        f"{algorithm.name}_{cell_tag}_env_{scenario.env}",
                        cmd,
                        params.generate_params(randomize=False),
                        root_dir_name=FULL_OBSERVABILITY_ROOT_DIR,
                    )
                )
    return experiments


def obs_modality_2x2_experiments():
    experiments = []
    for scenario in REPRESENTATIVE_2X2_SCENARIOS:
        for algorithm in (algorithm_by_name(name) for name in REBUTTAL_KD6J_ALGORITHMS):
            params = _kd6j_params_for_scenario(scenario, algorithm)
            for obs_modality, full_obs, cell_tag in OBS_MODALITY_2X2_CELLS:
                cmd = _kd6j_base_cmd(
                    scenario,
                    algorithm,
                    obs_modality=obs_modality,
                    full_observability=full_obs,
                )
                experiments.append(
                    Experiment(
                        f"{algorithm.name}_{cell_tag}_env_{scenario.env}",
                        cmd,
                        params.generate_params(randomize=False),
                        root_dir_name=OBS_MODALITY_2X2_ROOT_DIR,
                    )
                )
    return experiments


def visibility_telemetry_experiments():
    """One MAPPO Pixel-Partial checkpoint per non-2x2 scenario, used only for visibility/occlusion telemetry."""
    algorithm = algorithm_by_name("MAPPO")
    experiments = []
    for scenario in FULL_OBSERVABILITY_SCENARIOS:
        cmd = _kd6j_base_cmd(scenario, algorithm, obs_modality="pixel", full_observability=False)
        params = _kd6j_params_for_scenario(scenario, algorithm, seeds=VISIBILITY_TELEMETRY_SEEDS)
        experiments.append(
            Experiment(
                f"{algorithm.name}_pixel_partial_env_{scenario.env}",
                cmd,
                params.generate_params(randomize=False),
                root_dir_name=VISIBILITY_TELEMETRY_ROOT_DIR,
            )
        )
    return experiments


# Unconfounded Q1 test (MAPPO, 3 seeds): Pixel/Partial = own RGB only; Pixel/Full = [own, partner] RGB + 5D relative
# pose, with no AdditionalInput task scalars in either cell. Pixel/Partial runs already trained by the obs_modality_2x2
# (s42, s68) and visibility_telemetry (coop_puzzle s42) profiles are behaviourally identical (spec default disables
# AdditionalInput, and Partial never enabled it) and are reused, so only
# the missing seeds are trained here.
FULL_OBS_CLEAN_ROOT_DIR = "full_obs_clean"
FULL_OBS_CLEAN_ENV_NAMES = ("rhythm_sync_dense", "armory_siege", "coop_puzzle")
FULL_OBS_CLEAN_SEEDS = [
    int(s.strip()) for s in os.environ.get("COMRAD_KD6J_CLEAN_SEEDS", "42,68,81").split(",") if s.strip()
]
FULL_OBS_CLEAN_EXISTING_PARTIAL_SEEDS = {
    "rhythm_sync_dense": (42, 68),
    "armory_siege": (42, 68),
    "coop_puzzle": (42,),
}


def full_obs_clean_experiments():
    algorithm = algorithm_by_name("MAPPO")
    experiments = []
    for scenario in (s for s in BENCHMARK_SCENARIOS if s.env in FULL_OBS_CLEAN_ENV_NAMES):
        existing = set(FULL_OBS_CLEAN_EXISTING_PARTIAL_SEEDS.get(scenario.env, ()))
        cells = (
            (False, "pixel_partial", [s for s in FULL_OBS_CLEAN_SEEDS if s not in existing]),
            (True, "pixel_full_clean", list(FULL_OBS_CLEAN_SEEDS)),
        )
        for full_obs, cell_tag, seeds in cells:
            if not seeds:
                continue
            cmd = _kd6j_base_cmd(scenario, algorithm, obs_modality="pixel", full_observability=full_obs)
            cmd += " --use_additional_input=False"
            params = _kd6j_params_for_scenario(scenario, algorithm, seeds=seeds)
            experiments.append(
                Experiment(
                    f"{algorithm.name}_{cell_tag}_env_{scenario.env}",
                    cmd,
                    params.generate_params(randomize=False),
                    root_dir_name=FULL_OBS_CLEAN_ROOT_DIR,
                )
            )
    return experiments


# True full observability (MAPPO, 3 seeds): own RGB plus the global environment state (all agents, task variables,
# every non-player object, layout grid, time) in `measurements`, built by GlobalStateWrapper. No AdditionalInput,
# so it differs from the reused Pixel/Partial runs only by the global state. Seeds already trained to 50M frames
# (mode3_comrad_full_obs_global_20261115_191501) are skipped, so a resubmission trains only the missing runs.
FULL_OBS_GLOBAL_ROOT_DIR = "full_obs_global"
FULL_OBS_GLOBAL_EXISTING_SEEDS = {
    "rhythm_sync_dense": (42, 68, 81),
    "armory_siege": (81,),
}


def full_obs_global_experiments():
    algorithm = algorithm_by_name("MAPPO")
    experiments = []
    for scenario in (s for s in BENCHMARK_SCENARIOS if s.env in FULL_OBS_CLEAN_ENV_NAMES):
        cmd = _kd6j_base_cmd(scenario, algorithm, obs_modality="pixel", full_observability=True)
        existing = set(FULL_OBS_GLOBAL_EXISTING_SEEDS.get(scenario.env, ()))
        seeds = [s for s in FULL_OBS_CLEAN_SEEDS if s not in existing]
        if not seeds:
            continue
        cmd += " --use_additional_input=False --full_obs_mode=global_state"
        params = _kd6j_params_for_scenario(scenario, algorithm, seeds=seeds)
        experiments.append(
            Experiment(
                f"{algorithm.name}_pixel_full_global_env_{scenario.env}",
                cmd,
                params.generate_params(randomize=False),
                root_dir_name=FULL_OBS_GLOBAL_ROOT_DIR,
            )
        )
    return experiments


# True full observability, fixed encoding (global_state_version=2), compared with MAPPO in Table 3 (tab:baseline_results):
# same 5 seeds and 100M-frame budget, trained on a layout-only WAD pool (default generator config, layout seeds disjoint
# from the held-out seeds, experiments/iclr2027_rebuttal/generate_layout_training_batch.py) like the held-out batches. Only the image is
# normalized: running statistics of the already scaled state saturated held-out inputs in the v1 runs.
FULL_OBS_GLOBAL_V2_ROOT_DIR = "full_obs_global_v2"
FULL_OBS_GLOBAL_V2_TRAIN_FOR_ENV_STEPS = 100_000_000
FULL_OBS_GLOBAL_V2_SEEDS = [
    int(s.strip()) for s in (os.environ.get("COMRAD_FULL_OBS_V2_SEEDS") or "42,68,81,97,154").split(",") if s.strip()
]
FULL_OBS_GLOBAL_V2_NUM_WORKERS = int(os.environ.get("COMRAD_FULL_OBS_V2_NUM_WORKERS") or 16)
FULL_OBS_GLOBAL_V2_ENVS_PER_WORKER = int(os.environ.get("COMRAD_FULL_OBS_V2_ENVS_PER_WORKER") or 8)
# A WAD swap restarts the multiplayer game (host/client sync); swapping after every episode cut FPS ~3x on gpu_mig.
# The 128 envs of a job sample maps independently, so the job still covers the pool at any time.
FULL_OBS_GLOBAL_V2_WAD_SWAP_EVERY = int(os.environ.get("COMRAD_FULL_OBS_V2_WAD_SWAP_EVERY") or 10)


def layout_batch_dir(env_name: str) -> str:
    return f"experiments/iclr2027_rebuttal/layouts/batch_{env_name}_layouts"


def full_obs_global_v2_experiments(envs=None):
    envs = _csv_env("COMRAD_FULL_OBS_V2_ENVS") if envs is None else set(envs)
    algorithm = algorithm_by_name("MAPPO")
    experiments = []
    for scenario in (s for s in BENCHMARK_SCENARIOS if s.env in FULL_OBS_CLEAN_ENV_NAMES):
        if envs is not None and scenario.env not in envs:
            continue
        cmd = _kd6j_base_cmd(
            scenario,
            algorithm,
            obs_modality="pixel",
            full_observability=True,
            train_for_env_steps=FULL_OBS_GLOBAL_V2_TRAIN_FOR_ENV_STEPS,
        )
        cmd = cmd.replace("--num_workers=16", f"--num_workers={FULL_OBS_GLOBAL_V2_NUM_WORKERS}").replace(
            "--num_envs_per_worker=8", f"--num_envs_per_worker={FULL_OBS_GLOBAL_V2_ENVS_PER_WORKER}"
        )
        cmd += " " + " ".join(
            [
                "--use_additional_input=False",
                "--full_obs_mode=global_state",
                "--global_state_version=2",
                "--normalize_input_keys obs",
                f"--wad_batch={layout_batch_dir(scenario.env)}",
                f"--wad_swap_every={FULL_OBS_GLOBAL_V2_WAD_SWAP_EVERY}",
                "--curriculum=uniform",
            ]
        )
        params = _kd6j_params_for_scenario(scenario, algorithm, seeds=FULL_OBS_GLOBAL_V2_SEEDS)
        experiments.append(
            Experiment(
                f"{algorithm.name}_pixel_full_global_v2_env_{scenario.env}",
                cmd,
                params.generate_params(randomize=False),
                root_dir_name=FULL_OBS_GLOBAL_V2_ROOT_DIR,
            )
        )
    return experiments


# State vs pixel with a standard partially observable state (Table tab:po_state, 3 seeds): the SMAC-style `entity`
# encoding (heading-relative partner and nearest objects within a sight radius, 16 depth rays, radius-gated task
# variables) built by EntityStateWrapper. Pixel-Partial seeds 42 and 68 are reused from obs_modality_2x2, so only
# seed 81 is trained here, with the identical obs_modality_2x2 command. State cells come first and Armory Siege
# first, so a partial submission still yields usable cells.
STATE_ENTITY_ROOT_DIR = "state_entity"
STATE_ENTITY_SEEDS = [
    # `or`: comrad/send.sh forwards the variable as '' when unset
    int(s.strip()) for s in (os.environ.get("COMRAD_STATE_ENTITY_SEEDS") or "42,68,81").split(",") if s.strip()
]
STATE_ENTITY_EXISTING_PIXEL_SEEDS = (42, 68)
STATE_ENTITY_CELLS = (
    ("armory_siege", "MAPPO"),
    ("armory_siege", "QMIX"),
    ("ammo_carrier", "MAPPO"),
    ("ammo_carrier", "QMIX"),
    ("rhythm_sync_dense", "MAPPO"),
)


def _csv_env(name: str):
    """Optional comma-separated filter from the environment (forwarded by comrad/send.sh); None means no filter."""
    values = [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]
    return set(values) or None


def state_entity_experiments(envs=None, modalities=None, algos=None):
    """`envs` / `modalities` / `algos` (or COMRAD_STATE_ENTITY_ENVS / _MODALITIES / _ALGOS) submit a subset when
    the SBU budget is short."""
    envs = _csv_env("COMRAD_STATE_ENTITY_ENVS") if envs is None else set(envs)
    algos = _csv_env("COMRAD_STATE_ENTITY_ALGOS") if algos is None else set(algos)
    modalities = _csv_env("COMRAD_STATE_ENTITY_MODALITIES") if modalities is None else set(modalities)
    scenarios = {scenario.env: scenario for scenario in BENCHMARK_SCENARIOS}
    pixel_seeds = [s for s in STATE_ENTITY_SEEDS if s not in STATE_ENTITY_EXISTING_PIXEL_SEEDS]
    experiments = []
    for obs_modality, cell_tag, seeds in (("state", "state_entity", STATE_ENTITY_SEEDS), ("pixel", "pixel_partial", pixel_seeds)):
        if not seeds or (modalities is not None and obs_modality not in modalities):
            continue
        for env_name, algorithm_name in STATE_ENTITY_CELLS:
            if envs is not None and env_name not in envs:
                continue
            if algos is not None and algorithm_name not in algos:
                continue
            scenario = scenarios[env_name]
            algorithm = algorithm_by_name(algorithm_name)
            cmd = _kd6j_base_cmd(scenario, algorithm, obs_modality=obs_modality, full_observability=False)
            if obs_modality == "state":
                cmd += " --state_encoding=entity"
            params = _kd6j_params_for_scenario(scenario, algorithm, seeds=list(seeds))
            experiments.append(
                Experiment(
                    f"{algorithm.name}_{cell_tag}_env_{scenario.env}",
                    cmd,
                    params.generate_params(randomize=False),
                    root_dir_name=STATE_ENTITY_ROOT_DIR,
                )
            )
    return experiments


def blind_ablation_experiments():
    experiments = []
    for scenario in REPRESENTATIVE_2X2_SCENARIOS:
        for algorithm in (algorithm_by_name(name) for name in BLIND_ABLATION_ALGORITHMS):
            cmd = _kd6j_base_cmd(
                scenario,
                algorithm,
                obs_modality="blind",
                full_observability=False,
            )
            params = _kd6j_params_for_scenario(scenario, algorithm)
            experiments.append(
                Experiment(
                    f"{algorithm.name}_blind_env_{scenario.env}",
                    cmd,
                    params.generate_params(randomize=False),
                    root_dir_name=BLIND_ABLATION_ROOT_DIR,
                )
            )
    return experiments


# Reviewer C2K6 frozen-partner baseline: Table 3 winners that are not covered by the 50M kd6J Pixel-Partial
# checkpoints on Snellius. Training stops exactly at the 100M evaluation frontier so the 100M snapshot survives
# keep_checkpoints=2 (as for the very-difficult oracle runs). WandB is disabled to keep home-quota writes at zero.
FROZEN_PARTNER_WINNERS_ROOT_DIR = "frozen_partner_winners"
FROZEN_PARTNER_WINNERS_TRAIN_FOR_ENV_STEPS = 100_000_000
FROZEN_PARTNER_WINNERS = (
    ("coop_health_gathering", "IPPO", (42, 68)),
    ("dumb_enemies", "IPPO", (42, 68)),
    ("ammo_carrier", "IPPO", (42, 68)),
    ("stealth_labyrinth", "QMIX", (42, 68, 81)),
)


def frozen_partner_winners_experiments():
    scenarios = {scenario.env: scenario for scenario in BENCHMARK_SCENARIOS}
    experiments = []
    for env_name, algorithm_name, seeds in FROZEN_PARTNER_WINNERS:
        scenario = scenarios[env_name]
        algorithm = algorithm_by_name(algorithm_name)
        cmd = _kd6j_base_cmd(
            scenario,
            algorithm,
            obs_modality="pixel",
            full_observability=False,
            train_for_env_steps=FROZEN_PARTNER_WINNERS_TRAIN_FOR_ENV_STEPS,
        ).replace("--with_wandb=True", "--with_wandb=False")
        params = _kd6j_params_for_scenario(scenario, algorithm, seeds=seeds)
        experiments.append(
            Experiment(
                f"{algorithm.name}_pixel_partial_env_{scenario.env}",
                cmd,
                params.generate_params(randomize=False),
                root_dir_name=FROZEN_PARTNER_WINNERS_ROOT_DIR,
            )
        )
    return experiments


# Reviewer kd6J: MAPPO trained from scratch with the partner masked (`partner_occlusion_mode=both`: partner sprite
# replaced by background via labels_buffer + partner-transmitted cues zeroed), i.e. the same intervention as the
# zero-shot Mask-partner evaluation column. Same 50M budget and seeds as the kd6J Pixel-Partial MAPPO checkpoints
# (obs_modality_2x2 for the 4 representative scenarios, visibility_telemetry for the other 9), so comparisons stay paired.
MASK_PARTNER_TRAINING_ROOT_DIR = "mask_partner_training"
MASK_PARTNER_TRAINING_MODE = os.environ.get("COMRAD_MASK_PARTNER_MODE", "both")


def mask_partner_training_experiments():
    algorithm = algorithm_by_name("MAPPO")
    experiments = []
    for scenario in BENCHMARK_SCENARIOS:
        cmd = _kd6j_base_cmd(scenario, algorithm, obs_modality="pixel", full_observability=False).replace(
            "--with_wandb=True", "--with_wandb=False"
        )
        cmd += f" --partner_occlusion_mode={MASK_PARTNER_TRAINING_MODE}"
        params = _kd6j_params_for_scenario(scenario, algorithm)
        experiments.append(
            Experiment(
                f"{algorithm.name}_pixel_partial_maskpartner_env_{scenario.env}",
                cmd,
                params.generate_params(randomize=False),
                root_dir_name=MASK_PARTNER_TRAINING_ROOT_DIR,
            )
        )
    return experiments


PROFILES = {
    "full_observability": (FULL_OBSERVABILITY_NAME, full_observability_experiments),
    "obs_modality_2x2": (OBS_MODALITY_2X2_NAME, obs_modality_2x2_experiments),
    "blind_ablation": (BLIND_ABLATION_NAME, blind_ablation_experiments),
    "visibility_telemetry": (VISIBILITY_TELEMETRY_NAME, visibility_telemetry_experiments),
    "full_obs_clean": (FULL_OBS_CLEAN_NAME, full_obs_clean_experiments),
    "full_obs_global": (FULL_OBS_GLOBAL_NAME, full_obs_global_experiments),
    "full_obs_global_v2": (FULL_OBS_GLOBAL_V2_NAME, full_obs_global_v2_experiments),
    "state_entity": (STATE_ENTITY_NAME, state_entity_experiments),
    "frozen_partner_winners": (FROZEN_PARTNER_WINNERS_NAME, frozen_partner_winners_experiments),
    "mask_partner_training": (MASK_PARTNER_TRAINING_NAME, mask_partner_training_experiments),
}
