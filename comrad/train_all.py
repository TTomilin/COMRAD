'''
Run agent full benchmark experiment (default): export COMRAD_TRAIN_ALL_PROFILE=benchmark && python -m sample_factory.launcher.run
Run agent scaling experiment: export COMRAD_TRAIN_ALL_PROFILE=agent_scaling && python -m sample_factory.launcher.run
Run agent curriculum experiment: export COMRAD_TRAIN_ALL_PROFILE=platform_chain_curriculum && python -m sample_factory.launcher.run
'''
import os
from dataclasses import dataclass

from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription


@dataclass(frozen=True)
class BenchmarkScenario:
    label: str
    env: str
    actor_critic_shared_reward: bool


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    cmd: str
    uses_shared_reward: bool


@dataclass(frozen=True)
class CurriculumExperimentSpec:
    suffix: str
    extra_args: str = ""


FULL_BENCHMARK_NAME = "comrad_benchmark"
AGENT_SCALING_NAME = "comrad_agent_scaling"
PLATFORM_CHAIN_CURRICULUM_NAME = "platform_chain_curriculum_compare"

PROFILE_FULL_BENCHMARK = "benchmark"
PROFILE_AGENT_SCALING = "agent_scaling"
PROFILE_PLATFORM_CHAIN_CURRICULUM = "platform_chain_curriculum"
PROFILE_QMIX_LR_SENSITIVITY_PILOT = "qmix_lr_sensitivity_pilot"
PROFILE_QMIX_LR_SENSITIVITY_FULL = "qmix_lr_sensitivity_full"

SEEDS = [42]
# SEEDS = [42, 68, 81, 97, 154]
TRAIN_FOR_ENV_STEPS = 125000000
QMIX_LR_SENSITIVITY_PILOT_STEPS = 50000000
QMIX_LR_SENSITIVITY_FULL_STEPS = TRAIN_FOR_ENV_STEPS

BENCHMARK_SCENARIOS = [
    BenchmarkScenario("Stag Hunt Arena", "stag_hunt_arena", False),
    BenchmarkScenario("Rhythm Sync", "rhythm_sync_dense", True),
    BenchmarkScenario("Foraging Commons", "foraging_commons", False),
    BenchmarkScenario("Co-op Puzzle", "coop_puzzle", True),
    BenchmarkScenario("Platform Chain", "platform_chain", True),
    # BenchmarkScenario("Platform Chain Easy", "platform_chain_easy", True),
    BenchmarkScenario("Armory Siege", "armory_siege", False),
    BenchmarkScenario("Co-op Health Gathering", "coop_health_gathering", False),
    BenchmarkScenario("Lava Pit", "lavapit", True),
    BenchmarkScenario("Smart Enemies", "smart_enemies", False),
    BenchmarkScenario("Dumb Enemies", "dumb_enemies", False),
    BenchmarkScenario("Stealth Labyrinth", "stealth_labyrinth", False),
    BenchmarkScenario("Ammo Carrier", "ammo_carrier", False),
    BenchmarkScenario("Lava Maze", "lava_maze", False),
]

ALGORITHMS = [
    AlgorithmSpec(
        "IPPO",
        f"python -m comrad.train --algo=IPPO --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_workers=16 --num_envs_per_worker=8 --policy_workers_per_policy=2 --num_policies=1 --batch_size=4096 --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_record_every=10 --wandb_project=COMRAD --num_agents=2 --num_epochs=4 --rnn_type=lstm",
        True,
    ),
    AlgorithmSpec(
        "MAPPO",
        f"python -m comrad.train --algo=MAPPO --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_workers=16 --num_envs_per_worker=8 --policy_workers_per_policy=2 --num_policies=1 --batch_size=4096 --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_record_every=10 --wandb_project=COMRAD --num_agents=2 --num_epochs=4 --rnn_type=lstm",
        True,
    ),
    AlgorithmSpec(
        "HAPPO",
        f"python -m comrad.train --algo=HAPPO --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_workers=16 --num_envs_per_worker=8 --policy_workers_per_policy=2 --batch_size=4096 --env_frameskip=4 --use_rnn=True --happo_critic_rnn=True --max_policy_lag=3000 --lr_schedule=linear_decay --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_record_every=10 --wandb_project=COMRAD --num_agents=2",
        True,
    ),
    AlgorithmSpec(
        "IDQN",
        f"python -m comrad.train --algo=IDQN --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_workers=16 --num_envs_per_worker=8 --policy_workers_per_policy=2 --batch_size=4096 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_record_every=10 --wandb_project=COMRAD --num_agents=2 --dqn_max_updates_per_batch=4 --batched_sampling=True --use_rnn=False",
        False,
    ),
    AlgorithmSpec(
        "VDN",
        f"python -m comrad.train --algo=VDN --mixer=vdn --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_workers=16 --num_envs_per_worker=8 --policy_workers_per_policy=2 --batch_size=4096 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_record_every=10 --wandb_project=COMRAD --qmix_sequence_batch_size=64 --num_agents=2 --dqn_max_updates_per_batch=4 --batched_sampling=True",
        False,
    ),
    AlgorithmSpec(
        "QMIX",
        f"python -m comrad.train --algo=QMIX --mixer=qmix --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_workers=16 --num_envs_per_worker=8 --policy_workers_per_policy=2 --batch_size=4096 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_record_every=10 --wandb_project=COMRAD --qmix_sequence_batch_size=64 --num_agents=2 --dqn_max_updates_per_batch=4 --batched_sampling=True",
        False,
    ),
    AlgorithmSpec(
        "QPLEX_dmaq",
        f"python -m comrad.train --algo=QPLEX --mixer=dmaq --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_workers=16 --num_envs_per_worker=8 --policy_workers_per_policy=2 --batch_size=4096 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_project=COMRAD --wandb_record_every=10 --qmix_sequence_batch_size=16 --num_agents=2 --dqn_max_updates_per_batch=4 --batched_sampling=True --qplex_grad_accum_mini_bs=16 --qplex_state_bias=False",
        False,
    ),
    AlgorithmSpec(
        "QPLEX_dmaq_qatten",
        f"python -m comrad.train --algo=QPLEX --mixer=dmaq_qatten --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_workers=16 --num_envs_per_worker=8 --policy_workers_per_policy=2 --batch_size=4096 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_project=COMRAD --wandb_record_every=10 --qmix_sequence_batch_size=16 --num_agents=2 --dqn_max_updates_per_batch=4 --batched_sampling=True --qplex_grad_accum_mini_bs=16 --qplex_state_bias=False",
        False,
    ),
]

AGENT_SCALING_SCENARIO = BenchmarkScenario("Armory Siege", "armory_siege", False)
AGENT_SCALING_AGENT_COUNTS = [2, 3, 4, 6, 8]
AGENT_SCALING_ROOT_DIR = "armory_siege_agent_scaling"
AGENT_SCALING_BATCH_SIZE_BY_NUM_AGENTS = {
    2: 2304,
    3: 2304,
    4: 2048,
    6: 2304,
    8: 2048,
}

QMIX_LR_SENSITIVITY_PILOT_NAME = "comrad_qmix_lr_sensitivity_pilot"
QMIX_LR_SENSITIVITY_FULL_NAME = "comrad_qmix_lr_sensitivity_full"
QMIX_LR_SENSITIVITY_ROOT_DIR = "armory_siege_qmix_lr_sensitivity"
QMIX_LR_SENSITIVITY_NUM_AGENTS = 3
QMIX_LR_SENSITIVITY_BATCH_SIZE = 2304
QMIX_LR_SENSITIVITY_LR_CANDIDATES = [5e-5, 1e-4, 2e-4, 5e-4]
QMIX_LR_SENSITIVITY_REPRESENTATIVE_LR = 1e-4

PLATFORM_CHAIN_CURRICULUM_SCENARIO = BenchmarkScenario("Platform Chain", "platform_chain", True)
PLATFORM_CHAIN_CURRICULUM_ALGORITHM = AlgorithmSpec(
    "MAPPO",
    f"python -m comrad.train --algo=MAPPO --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_workers=16 --num_envs_per_worker=8 --policy_workers_per_policy=2 --num_policies=1 --batch_size=4096 --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_record_every=10 --wandb_project=COMRAD --num_agents=2 --num_epochs=4 --rnn_type=lstm",
    True,
)
PLATFORM_CHAIN_CURRICULUM_BATCH_DIR = "comrad/scenarios/batch_platform_chain_curriculum"
PLATFORM_CHAIN_CURRICULUM_INTERESTINGNESS_PATH = (
    "comrad/scenarios/batch_platform_chain_curriculum/platform_chain_interestingness.json"
)
PLATFORM_CHAIN_CURRICULUM_ROOT_DIR = "platform_chain_curriculum_compare"
PLATFORM_CHAIN_CURRICULUM_SEQ_THRESHOLD = 0.6
PLATFORM_CHAIN_CURRICULUM_SEQ_WINDOW = 50
PLATFORM_CHAIN_CURRICULUM_VARIANTS = [
    CurriculumExperimentSpec("baseline"),
    CurriculumExperimentSpec("uniform", f"--wad_batch={PLATFORM_CHAIN_CURRICULUM_BATCH_DIR} --curriculum=uniform"),
    CurriculumExperimentSpec(
        "sequential",
        " ".join(
            [
                f"--wad_batch={PLATFORM_CHAIN_CURRICULUM_BATCH_DIR}",
                "--curriculum=sequential",
                f"--seq_threshold={PLATFORM_CHAIN_CURRICULUM_SEQ_THRESHOLD}",
                f"--seq_window={PLATFORM_CHAIN_CURRICULUM_SEQ_WINDOW}",
            ]
        ),
    ),
    CurriculumExperimentSpec(
        "learning_progress",
        " ".join(
            [
                f"--wad_batch={PLATFORM_CHAIN_CURRICULUM_BATCH_DIR}",
                "--curriculum=learning_progress",
                "--lp_min_return=0.0",
                "--lp_max_return=1.0",
            ]
        ),
    ),
    CurriculumExperimentSpec(
        "plr",
        f"--wad_batch={PLATFORM_CHAIN_CURRICULUM_BATCH_DIR} --curriculum=plr --with_vtrace=False",
    ),
    CurriculumExperimentSpec(
        "omni",
        " ".join(
            [
                f"--wad_batch={PLATFORM_CHAIN_CURRICULUM_BATCH_DIR}",
                "--curriculum=omni",
                "--lp_min_return=0.0",
                "--lp_max_return=1.0",
                f"--interestingness_graph_path={PLATFORM_CHAIN_CURRICULUM_INTERESTINGNESS_PATH}",
            ]
        ),
    ),
]

# Rerun only the jobs that failed in main results (server error)
# FAILED_BENCHMARK_RUNS = {
#     ("rhythm_sync_dense", "IPPO"),
#     ("rhythm_sync_dense", "VDN"),
#     ("rhythm_sync_dense", "QPLEX_dmaq"),
#     ("coop_health_gathering", "QPLEX_dmaq"),
# }


def _params_for_scenario(scenario: BenchmarkScenario, algorithm: AlgorithmSpec):
    if algorithm.uses_shared_reward:
        shared_reward_alpha = 1.0 if scenario.actor_critic_shared_reward else 0.0
        return ParamGrid([("shared_reward_alpha", [shared_reward_alpha]), ("seed", SEEDS)])

    return ParamGrid([("seed", SEEDS)])


def _command_for_scenario(scenario: BenchmarkScenario, algorithm: AlgorithmSpec) -> str:
    return f"{algorithm.cmd} --env={scenario.env}"


def _experiment_name_for_scenario(scenario: BenchmarkScenario, algorithm: AlgorithmSpec) -> str:
    return f"{algorithm.name}_env_{scenario.env}"


def _armory_siege_qmix_algorithm(train_for_env_steps: int, batch_size: int, learning_rate: float | None) -> AlgorithmSpec:
    parts = [
        "python -m comrad.train",
        "--algo=QMIX",
        "--mixer=qmix",
        f"--train_for_env_steps={train_for_env_steps}",
        "--num_workers=16",
        "--num_envs_per_worker=8",
        "--policy_workers_per_policy=2",
        f"--batch_size={batch_size}",
        "--env_frameskip=4",
        "--wide_aspect_ratio=False",
        "--with_wandb=True",
        "--wandb_dir=.",
        "--wandb_record_every=10",
        "--wandb_project=COMRAD",
        "--qmix_sequence_batch_size=32",
        "--dqn_max_updates_per_batch=4",
        "--batched_sampling=True",
    ]
    if learning_rate is not None:
        parts.append(f"--learning_rate={learning_rate}")
    return AlgorithmSpec("QMIX", " ".join(parts), False)


def _full_benchmark_experiments():
    experiments = []
    for scenario in BENCHMARK_SCENARIOS:
        for algorithm in ALGORITHMS:
            # if (scenario.env, algorithm.name) not in FAILED_BENCHMARK_RUNS:
            #     continue

            experiments.append(
                Experiment(
                    _experiment_name_for_scenario(scenario, algorithm),
                    _command_for_scenario(scenario, algorithm),
                    _params_for_scenario(scenario, algorithm).generate_params(randomize=False),
                    root_dir_name=scenario.env,
                )
            )

    return experiments


def _agent_scaling_experiments():
    experiments = []
    for num_agents in AGENT_SCALING_AGENT_COUNTS:
        algorithm = _armory_siege_qmix_algorithm(
            TRAIN_FOR_ENV_STEPS,
            AGENT_SCALING_BATCH_SIZE_BY_NUM_AGENTS[num_agents],
            QMIX_LR_SENSITIVITY_REPRESENTATIVE_LR,
        )
        experiments.append(
            Experiment(
                f"QMIX_env_armory_siege_agent_scaling_n.age_{num_agents}",
                f"{_command_for_scenario(AGENT_SCALING_SCENARIO, algorithm)} --num_agents={num_agents}",
                ParamGrid([("seed", SEEDS)]).generate_params(randomize=False),
                root_dir_name=AGENT_SCALING_ROOT_DIR,
            )
        )

    return experiments


def _qmix_lr_sensitivity_pilot_experiments():
    algorithm = _armory_siege_qmix_algorithm(
        QMIX_LR_SENSITIVITY_PILOT_STEPS,
        QMIX_LR_SENSITIVITY_BATCH_SIZE,
        None,
    )
    return [
        Experiment(
            "QMIX_env_armory_siege_lr_sensitivity_pilot",
            f"{_command_for_scenario(AGENT_SCALING_SCENARIO, algorithm)} --num_agents={QMIX_LR_SENSITIVITY_NUM_AGENTS}",
            ParamGrid(
                [
                    ("learning_rate", QMIX_LR_SENSITIVITY_LR_CANDIDATES),
                    ("seed", SEEDS),
                ]
            ).generate_params(randomize=False),
            root_dir_name=QMIX_LR_SENSITIVITY_ROOT_DIR,
        )
    ]


def _qmix_lr_sensitivity_full_experiments():
    algorithm = _armory_siege_qmix_algorithm(
        QMIX_LR_SENSITIVITY_FULL_STEPS,
        QMIX_LR_SENSITIVITY_BATCH_SIZE,
        QMIX_LR_SENSITIVITY_REPRESENTATIVE_LR,
    )
    return [
        Experiment(
            "QMIX_env_armory_siege_lr_sensitivity_full",
            f"{_command_for_scenario(AGENT_SCALING_SCENARIO, algorithm)} --num_agents={QMIX_LR_SENSITIVITY_NUM_AGENTS}",
            ParamGrid([("seed", SEEDS)]).generate_params(randomize=False),
            root_dir_name=QMIX_LR_SENSITIVITY_ROOT_DIR,
        )
    ]


def _platform_chain_curriculum_experiments():
    params = _params_for_scenario(PLATFORM_CHAIN_CURRICULUM_SCENARIO, PLATFORM_CHAIN_CURRICULUM_ALGORITHM)
    baseline_cmd = _command_for_scenario(PLATFORM_CHAIN_CURRICULUM_SCENARIO, PLATFORM_CHAIN_CURRICULUM_ALGORITHM)
    experiments = []
    for variant in PLATFORM_CHAIN_CURRICULUM_VARIANTS:
        cmd = baseline_cmd
        if variant.extra_args:
            cmd = f"{cmd} {variant.extra_args}"
        experiments.append(
            Experiment(
                f"MAPPO_env_platform_chain_{variant.suffix}",
                cmd,
                params.generate_params(randomize=False),
                root_dir_name=PLATFORM_CHAIN_CURRICULUM_ROOT_DIR,
            )
        )
    return experiments


def _active_profile() -> str:
    profile = os.environ.get("COMRAD_TRAIN_ALL_PROFILE", PROFILE_FULL_BENCHMARK).strip().lower()
    valid_profiles = {
        PROFILE_FULL_BENCHMARK,
        PROFILE_AGENT_SCALING,
        PROFILE_PLATFORM_CHAIN_CURRICULUM,
        PROFILE_QMIX_LR_SENSITIVITY_PILOT,
        PROFILE_QMIX_LR_SENSITIVITY_FULL,
    }
    if profile not in valid_profiles:
        raise ValueError(
            f"Unsupported COMRAD_TRAIN_ALL_PROFILE={profile!r}. "
            f"Expected one of {sorted(valid_profiles)}."
        )
    return profile


ACTIVE_PROFILE = _active_profile()

if ACTIVE_PROFILE == PROFILE_AGENT_SCALING:
    BENCHMARK_NAME = AGENT_SCALING_NAME
    _experiments = _agent_scaling_experiments()
elif ACTIVE_PROFILE == PROFILE_PLATFORM_CHAIN_CURRICULUM:
    BENCHMARK_NAME = PLATFORM_CHAIN_CURRICULUM_NAME
    _experiments = _platform_chain_curriculum_experiments()
elif ACTIVE_PROFILE == PROFILE_QMIX_LR_SENSITIVITY_PILOT:
    BENCHMARK_NAME = QMIX_LR_SENSITIVITY_PILOT_NAME
    _experiments = _qmix_lr_sensitivity_pilot_experiments()
elif ACTIVE_PROFILE == PROFILE_QMIX_LR_SENSITIVITY_FULL:
    BENCHMARK_NAME = QMIX_LR_SENSITIVITY_FULL_NAME
    _experiments = _qmix_lr_sensitivity_full_experiments()
else:
    BENCHMARK_NAME = FULL_BENCHMARK_NAME
    _experiments = _full_benchmark_experiments()

TOTAL_RUNS = sum(max(1, len(experiment.params)) for experiment in _experiments)

RUN_DESCRIPTION = RunDescription(
    os.environ.get("COMRAD_RUN_NAME", BENCHMARK_NAME),
    experiments=_experiments,
)
