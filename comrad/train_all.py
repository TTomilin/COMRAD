# python -m sample_factory.launcher.run --run=sf.train_all --backend=processes --max_parallel=1 --pause_between=1

import os

from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

env = "doom_pitfall"
n_agents = 2
seed = [0]
time = int(os.environ.get("train_for_seconds", 3600))
wandb_project = "comrad_jr"

wandb = f"--with_wandb=True --wandb_dir=. --wandb_project={wandb_project}"
common = f"--env={env} --train_for_seconds={time} --env_frameskip=4 --wide_aspect_ratio=False --num_agents={n_agents} {wandb} --num_workers=8 --num_envs_per_worker=8 --batched_sampling=True"

#======================

mappo = f"python -m sf.train {common} --algo=MAPPO --use_rnn=True --num_policies=1 --batch_size=1024 --wandb_record_every=10"

happo = f"python -m sf.train {common} --algo=HAPPO --policy_workers_per_policy=2 --batch_size=2048 --use_rnn=True --happo_critic_rnn=True --max_policy_lag=3000 --lr_schedule=linear_decay --wandb_record_every=10"

qmix = f"python -m sf.train {common} --algo=QMIX --mixer=qmix --policy_workers_per_policy=2 --batch_size=3072 --use_rnn=True --rnn_type=gru --rnn_size=256 --rollout=32 --gamma=0.99 --learning_starts=50000 --qmix_buffer_batch_size=256 --qmix_sequence_batch_size=64 --replay_buffer_size=500000 --epsilon_decay_steps=20000000 --epsilon_end=0.005 --learning_rate=0.0001 --dqn_max_updates_per_batch=4 --target_update_tau=0.005 --use_huber_loss=True --q_value_clamp=100 --train_frequency=8"

# vdn = f"python -m sf.train {common} --algo=QMIX --mixer=vdn --policy_workers_per_policy=2 --batch_size=3072 --use_rnn=True --rnn_type=gru --rnn_size=256 --rollout=32 --gamma=0.99 --learning_starts=50000 --qmix_buffer_batch_size=256 --qmix_sequence_batch_size=64 --replay_buffer_size=500000 --epsilon_decay_steps=20000000 --epsilon_end=0.005 --learning_rate=0.0001 --dqn_max_updates_per_batch=4 --target_update_tau=0.005 --use_huber_loss=True --q_value_clamp=100 --train_frequency=8"


#===================================

_seed_grid = ParamGrid([("seed", seed)])
_experiments = [
    Experiment("MAPPO", mappo, _seed_grid.generate_params(randomize=False)),
    Experiment("HAPPO", happo, _seed_grid.generate_params(randomize=False)),
    Experiment("QMIX", qmix, _seed_grid.generate_params(randomize=False)),
    # Experiment("VDN", vdn, _seed_grid.generate_params(randomize=False)),
]

RUN_DESCRIPTION = RunDescription(os.environ.get("COMRAD_RUN_NAME", f"bench_{env}"), experiments=_experiments)
