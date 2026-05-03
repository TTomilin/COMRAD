# python -m sample_factory.launcher.run --run=comrad.train_all --backend=processes --max_parallel=1 --pause_between=1

import os
from dataclasses import dataclass

from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription


@dataclass(frozen=True)
class BenchmarkScenario:
    label: str
    env: str
    actor_critic_shared_reward: bool


BENCHMARK_NAME = "comrad_benchmark"
SEEDS = [42]
TRAIN_FOR_ENV_STEPS = 125000000

BENCHMARK_SCENARIOS = [
    BenchmarkScenario("Stag Hunt Arena", "stag_hunt_arena", False),
    BenchmarkScenario("Rhythm Sync", "rhythm_sync_dense", True),
    BenchmarkScenario("Foraging Commons", "foraging_commons", False),
    BenchmarkScenario("Co-op Puzzle", "coop_puzzle", True),
    BenchmarkScenario("Platform Chain", "platform_chain", True),
    BenchmarkScenario("Platform Chain Easy", "platform_chain_easy", True),
    BenchmarkScenario("Armory Siege", "armory_siege", False),
    BenchmarkScenario("Co-op Health Gathering", "coop_health_gathering", False),
    BenchmarkScenario("Lava Pit", "lavapit", True),
    BenchmarkScenario("Smart Enemies", "smart_enemies", False),
    BenchmarkScenario("Dumb Enemies", "dumb_enemies", False),
    BenchmarkScenario("Stealth Labyrinth", "stealth_labyrinth", False),
    BenchmarkScenario("Ammo Carrier", "ammo_carrier", False),
    BenchmarkScenario("Lava Maze", "lava_maze", False),
]

_actor_critic_params = ParamGrid(
    [
        (
            ("env", "shared_reward_alpha"),
            [(scenario.env, 1.0 if scenario.actor_critic_shared_reward else 0.0) for scenario in BENCHMARK_SCENARIOS],
        ),
        ("seed", SEEDS),
    ]
)
_off_policy_params = ParamGrid(
    [
        ("env", [scenario.env for scenario in BENCHMARK_SCENARIOS]),
        ("seed", SEEDS),
    ]
)

ippo = (
    f"python -m comrad.train --algo=IPPO --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_envs_per_worker=8 --policy_workers_per_policy=2 --num_policies=1 --batch_size=4096 --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_record_every=10 --wandb_project=comrad_jr --num_agents=2 --num_epochs=4 --rnn_type=lstm"
)

mappo = (
    f"python -m comrad.train --algo=MAPPO --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_envs_per_worker=8 --policy_workers_per_policy=2 --num_policies=1 --batch_size=4096 --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_record_every=10 --wandb_project=comrad_jr --num_agents=2 --num_epochs=4 --rnn_type=lstm"
)

happo = (
    f"python -m comrad.train --algo=HAPPO --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_envs_per_worker=8 --policy_workers_per_policy=2 --batch_size=4096 --env_frameskip=4 --use_rnn=True --happo_critic_rnn=True --max_policy_lag=3000 --lr_schedule=linear_decay --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_record_every=10 --wandb_project=comrad_jr --num_agents=2"
)

idqn = (
    f"python -m comrad.train --algo=IDQN --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_workers=16 --num_envs_per_worker=8 --policy_workers_per_policy=2 --batch_size=4096 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_record_every=10 --wandb_project=comrad_jr --qmix_sequence_batch_size=64 --num_agents=2 --dqn_max_updates_per_batch=4 --batched_sampling=True"
)

vdn = (
    f"python -m comrad.train --algo=VDN --mixer=vdn --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_workers=16 --num_envs_per_worker=8 --policy_workers_per_policy=2 --batch_size=4096 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_record_every=10 --wandb_project=comrad_jr --qmix_sequence_batch_size=64 --num_agents=2 --dqn_max_updates_per_batch=4 --batched_sampling=True"
)

qmix = (
    f"python -m comrad.train --algo=QMIX --mixer=qmix --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_workers=16 --num_envs_per_worker=8 --policy_workers_per_policy=2 --batch_size=4096 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_record_every=10 --wandb_project=comrad_jr --qmix_sequence_batch_size=64 --num_agents=2 --dqn_max_updates_per_batch=4 --batched_sampling=True"
)

qplex_dmaq = (
    f"python -m comrad.train --algo=QPLEX --mixer=dmaq --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_workers=16 --num_envs_per_worker=8 --policy_workers_per_policy=2 --batch_size=4096 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_project=comrad_jr --wandb_record_every=10 --qmix_sequence_batch_size=16 --num_agents=2 --dqn_max_updates_per_batch=4 --batched_sampling=True --qplex_grad_accum_mini_bs=16 --qplex_state_bias=False"
)

qplex_qatten = (
    f"python -m comrad.train --algo=QPLEX --mixer=dmaq_qatten --train_for_env_steps={TRAIN_FOR_ENV_STEPS} --num_workers=16 --num_envs_per_worker=8 --policy_workers_per_policy=2 --batch_size=4096 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_project=comrad_jr --wandb_record_every=10 --qmix_sequence_batch_size=16 --num_agents=2 --dqn_max_updates_per_batch=4 --batched_sampling=True --qplex_grad_accum_mini_bs=16 --qplex_state_bias=False"
)

_experiments = [
    Experiment("IPPO", ippo, _actor_critic_params.generate_params(randomize=False)),
    Experiment("MAPPO", mappo, _actor_critic_params.generate_params(randomize=False)),
    Experiment("HAPPO", happo, _actor_critic_params.generate_params(randomize=False)),
    Experiment("IDQN", idqn, _off_policy_params.generate_params(randomize=False)),
    Experiment("VDN", vdn, _off_policy_params.generate_params(randomize=False)),
    Experiment("QMIX", qmix, _off_policy_params.generate_params(randomize=False)),
    Experiment("QPLEX_dmaq", qplex_dmaq, _off_policy_params.generate_params(randomize=False)),
    Experiment("QPLEX_dmaq_qatten", qplex_qatten, _off_policy_params.generate_params(randomize=False)),
]

TOTAL_RUNS = sum(max(1, len(experiment.params)) for experiment in _experiments)

RUN_DESCRIPTION = RunDescription(
    os.environ.get("COMRAD_RUN_NAME", BENCHMARK_NAME),
    experiments=_experiments,
)
