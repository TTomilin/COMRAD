# Sample Factory baselines

The repository ships a modified [Sample Factory](https://github.com/alex-petrenko/sample-factory) with seven MARL algorithms. It is installed together with `comrad-bench`.

```bash
python -m comrad.train --env=stag_hunt_arena --algo=MAPPO --train_for_env_steps=10000000
```

Logs, configs and checkpoints are written to `train_dir/<experiment>/`. The main log is `sf_log.txt`.

| Family | `--algo` | Reference |
|---|---|---|
| Independent actor-critic | `IPPO` | Sample Factory |
| Centralised critic | `MAPPO` | Sample Factory |
| Heterogeneous-agent trust region | `HAPPO` | [HARL](https://github.com/PKU-MARL/HARL) |
| Independent Q-learning | `IDQN` | [PyMARL](https://github.com/oxwhirl/pymarl) |
| Value decomposition | `VDN`, `QMIX` | [PyMARL](https://github.com/oxwhirl/pymarl) |
| Duplex dueling decomposition | `QPLEX` | [QPLEX](https://github.com/wjh720/QPLEX) |

## Per-algorithm commands

=== "IPPO / MAPPO"

    ```bash
    python -m comrad.train --env=stag_hunt_arena --algo=MAPPO --train_for_env_steps=5000
    ```

=== "HAPPO"

    ```bash
    python -m comrad.train --env=stag_hunt_arena --algo=HAPPO --num_agents=2 \
        --max_policy_lag=3000 --lr_schedule=linear_decay \
        --use_rnn=True --happo_critic_rnn=True --train_for_env_steps=5000
    ```

    - HAPPO updates agents one after another, so scale `--max_policy_lag` by `num_agents + 1` (default lag 1000 → 3000 for 2 agents).
    - Use `--lr_schedule=constant` or `linear_decay`. A KL-adaptive schedule does not fit sequential updates.

=== "IDQN"

    ```bash
    python -m comrad.train --env=stag_hunt_arena --algo=IDQN --use_rnn=False \
        --target_update_tau=1.0 --train_for_env_steps=5000
    ```

=== "VDN / QMIX"

    ```bash
    python -m comrad.train --env=stag_hunt_arena --algo=QMIX --use_rnn=False --train_for_env_steps=5000
    # recurrent variant
    python -m comrad.train --env=stag_hunt_arena --algo=QMIX --use_rnn=True --rnn_type=gru --rnn_size=64 \
        --rollout=16 --actor_critic_share_weights=True --per=False --train_for_env_steps=5000
    ```

    `--algo=VDN` sets `--mixer=vdn`. With RNNs, prioritised replay is disabled automatically.

=== "QPLEX"

    ```bash
    python -m comrad.train --env=stag_hunt_arena --algo=QPLEX --learning_rate=0.0001 \
        --qmix_sequence_batch_size=16 --train_for_env_steps=5000
    ```

## Useful flags

| Flag | Meaning |
|---|---|
| `--num_agents=N` | Team size, defaults to the scenario's value. |
| `--device=cpu` | Train without CUDA. |
| `--num_workers`, `--num_envs_per_worker` | Rollout parallelism. `4` and `4` with `--policy_workers_per_policy=2` is a good starting point on 16 CPUs. |
| `--shared_reward_alpha=α` | Override the scenario's shared-reward blend (actor-critic only). |
| `--use_additional_input=True` | Feed the measurement vector to the policy. |
| `--with_wandb=True`, `--wandb_record_every=N` | Log to Weights & Biases and record a video every N episodes. |
| DQN: `--learning_starts`, `--replay_buffer_size`, `--epsilon_decay_steps`, `--target_update_interval` | Replay and exploration schedule. Use `--learning_starts=0` for quick local tests. |

## Batch-size rules

- **MAPPO**: `batch_size / rollout` must be divisible by `num_agents`. For 3 agents with `rollout=32`, use `batch_size=3072`.
- **HAPPO** and **QMIX/QPLEX**: `batch_size / rollout` must be divisible by `num_agents · num_envs_per_worker / worker_num_splits`. For 3 agents with 4 envs and 2 splits, use `batch_size=2304` (HAPPO) or `1920` (QMIX).
- **QMIX with 3 or more agents**: use `--qmix_sequence_batch_size=32` to keep the learner from becoming the bottleneck.

## Shared reward

`shared_reward_alpha` only affects actor-critic algorithms. Value-decomposition learners already sum per-agent rewards.

| Scenario | Use `α = 1.0` for MAPPO/HAPPO? | Why |
|---|:---:|---|
| `ammo_carrier`, `armory_siege` | yes | One shared defence objective; local shaping is role-specific. |
| `lava_maze`, `lavapit`, `platform_chain`, `coop_puzzle` | yes | Fully joint traversal or puzzle progress. |
| `rhythm_sync`, `stealth_labyrinth` | yes | Success is defined by the pair. |
| `stag_hunt_arena`, `foraging_commons` | no | Local incentives are the point of the dilemma. |
| `coop_health_gathering`, `smart_enemies`, `dumb_enemies` | no | Keep the scenario default. |

## Recording a trained policy

```bash
python -m comrad.record_video --env=stag_hunt_arena --experiment=<experiment> \
    --load_checkpoint_kind=best --resolution=1920x1080 --max_num_episodes=5
```

This needs the full experiment directory (`config.json` and `checkpoint_p0/`), not only a `.pth` file.
