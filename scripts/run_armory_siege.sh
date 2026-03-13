#!/bin/bash

# Ref: https://github.com/TTomilin/HASARD/blob/main/scripts/reproduce_paper_results.sh

methods=("QMIX")
envs=("armory_siege")
seeds=(1)

# Loop over parameter combinations and submit Slurm jobs
for algo in "${methods[@]}"; do
    for env in "${envs[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Submitting job for combination: Algo=$algo, Env=$env, Seed=$seed"

            # Create an SBATCH script
            cat <<EOF | sbatch
#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time 05:00:00
#SBATCH --gres gpu:1
#SBATCH --output=~/slurm/%j_"${algo}"_"${env}"_$(date +%Y-%m-%d-%H-%M-%S).out
#SBATCH --error=~/slurm/%j_"${algo}"_"${env}"_$(date +%Y-%m-%d-%H-%M-%S).err

module purge
module load uv
uv venv
source .venv/bin/activate
uv sync --active -p .venv

python -m comrad.train --env=armory_siege --algo=QMIX --mixer=qmix --train_for_env_steps=50000000 --num_workers=4 --num_envs_per_worker=8 --policy_workers_per_policy=1 --batch_size=2048 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_project=marl_vizdoom --use_rnn=True --rnn_type=gru --rnn_size=256 --rollout=32 --gamma=0.99 --learning_starts=50000 --qmix_sequence_batch_size=64 --replay_buffer_size=500000 --epsilon_decay_steps=20000000 --epsilon_end=0.005 --learning_rate=0.0001 --num_agents=2 --dqn_max_updates_per_batch=4 --target_update_tau=0.005 --use_huber_loss=True --q_value_clamp=100 --train_frequency=8 --batched_sampling=True

EOF
        done
    done
done
