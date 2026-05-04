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


BENCHMARK_NAME = "comrad_benchmark"
SEEDS = [42]
TRAIN_FOR_ENV_STEPS = 125000000

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


_experiments = []
for scenario in BENCHMARK_SCENARIOS:
    for algorithm in ALGORITHMS:
        # if (scenario.env, algorithm.name) not in FAILED_BENCHMARK_RUNS:
        #     continue

        _experiments.append(
            Experiment(
                _experiment_name_for_scenario(scenario, algorithm),
                _command_for_scenario(scenario, algorithm),
                _params_for_scenario(scenario, algorithm).generate_params(randomize=False),
                root_dir_name=scenario.env,
            )
        )

TOTAL_RUNS = sum(max(1, len(experiment.params)) for experiment in _experiments)

RUN_DESCRIPTION = RunDescription(
    os.environ.get("COMRAD_RUN_NAME", BENCHMARK_NAME),
    experiments=_experiments,
)
