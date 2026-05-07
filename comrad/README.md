# `comrad/`

This package contains the benchmark-specific code that sits on top of ViZDoom and the modified Sample Factory runner.

## Main entrypoints

- `train.py`: single-run training entrypoint used by `python -m comrad.train`
- `train_all.py`: launcher-owned benchmark, scaling, and curriculum profiles
- `record_video.py`: high-resolution RGB checkpoint recording
- `record_topdown_heatmap.py`: automap-aligned trajectory and density rendering
- `send.sh`: HPC sync and SLURM submission helper
- `enjoy.py`: interactive or evaluation-time rollout entrypoint

## Important subdirectories

- [`envs/`](envs/README.md): environment construction, multiplayer wrappers, batch WAD support
- [`scenarios/`](scenarios/README.md): shipped `.cfg` and `.wad` benchmark assets
- [`wrappers/`](wrappers/README.md): reward shaping, shared reward, logging, and observation wrappers
- [`models/`](models/README.md): policy backbones and mixers for the supported MARL algorithms
- [`curriculum/`](curriculum/README.md): task-pool curriculum strategies and persistence helpers
- [`templates/`](templates/README.md): SLURM templates rendered by `send.sh`
- [`tests/`](tests/README.md): unit and integration tests for COMRAD-specific behavior
- [`examples/`](examples/README.md): exploratory scripts, not the main reproduction surface

## Other contents

- `play/`: local standalone VizDoom binaries and launch helpers
- `misc/`: smaller legacy scenarios and auxiliary assets that are not part of the main benchmark suite
- `utils/`: environment registration, rendering helpers, video upload, WAD utilities

Use this package as the authoritative code surface for benchmark runtime behavior. The paper defines the benchmark claims; `comrad/` defines how those claims are executed in code.

## Algorithms

#### IPPO (Independent PPO)

```bash
python -m comrad.train \
    --env=stag_hunt_arena \
    --algo=IPPO \
    --train_for_env_steps=5000
```

---

#### MAPPO

```bash
python -m comrad.train \
    --env=stag_hunt_arena \
    --algo=MAPPO \
    --train_for_env_steps=5000
```

---

#### HAPPO

```bash
python -m comrad.train \
    --env=stag_hunt_arena \
    --algo=HAPPO \
    --num_agents=N \
    --max_policy_lag=1000*(N+1) \
    --lr_schedule=linear_decay \
    --use_rnn=True --happo_critic_rnn=True \
    --train_for_env_steps=5000
```

+ HAPPO `train_step` increases N+1 times faster than base MAPPO/IPPO, so you should increase `--max_policy_lag` to `default_max_policy_lag * (num_agents + 1)`
+ For `--lr_schedule`, only `constant` or `linear_decay` allowed as policy is updated sequentially. Using a KL-adaptive learning rate doesnt make sense

---

#### IDQN (Independent DQN)

```bash
python -m comrad.train \
    --env=stag_hunt_arena \
    --algo=IDQN \
    --use_rnn=False \
    --target_update_tau=1.0 \
    --train_for_env_steps=5000
```

For training on HPC, I tuned with these parameters (this config technically edges 32GB RAM):
```bash
# Single agent
python -m comrad.train --env=stag_hunt_arena --algo=IDQN --train_for_seconds=21600 --num_workers=16 --num_envs_per_worker=16 --batch_size=2048 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_project=marl_vizdoom --use_rnn=False --learning_starts=50000 --dqn_batch_size=256 --replay_buffer_size=200000 --target_update_interval=2500 --epsilon_decay_steps=4000000 --epsilon_end=0.005 --per_beta_frames=2000000 --num_agents=1 --dqn_max_updates_per_batch=4 --target_update_tau=0.005 --dqn_reward_clip=1.0 --train_frequency=8

# Multi agent
# Same thing but num_agents=2
```

Some important DQN flags (also in `cfg.py`):
- `--replay_buffer_size`: Size of the replay buffer in transitions (default: `1000000`)
- `--learning_starts`: Start training after this many transitions are collected (default: `10000`)
- `--epsilon_start` / `--epsilon_end`: Exploration rate range (default: `1.0` -> `0.01`)
- `--epsilon_decay_steps`: Steps to anneal epsilon (default: `100000`)
- `--target_update_interval`: How many each learning steps to update the target network (default: `1000` steps)
- `--double_dqn`: Use Double DQN (default: `True`)

Note:
- You should use `--learning_starts=0` for small local tests so learner work immediately
- Avoid setting `--replay_buffer_size` too large for less memory allocation overhead
- `--train_frequency` is how many env steps per update, so increasing it decreases learner work but dont increase too much

---

#### VDN
```bash
python -m comrad.train \
    --env=stag_hunt_arena \
    --algo=VDN \
    --use_rnn=False \
    --train_for_env_steps=5000
```

This automatically set `--mixer="vdn"`. One can also use `--algo=QMIX --mixer=vdn`. This gives the same result.

---

#### QMIX
```bash
python -m comrad.train \
    --env=stag_hunt_arena \
    --algo=QMIX \
    --use_rnn=False \
    --train_for_env_steps=5000
```

This automatically set `--mixer="qmix"` as that's the default value.

---

#### VDN/QMIX + RNN (GRU)
```bash
python -m comrad.train \
    --env=stag_hunt_arena \
    --algo=QMIX \
    --use_rnn=True --rnn_type=gru --rnn_size=64 --rollout=16 \
    --actor_critic_share_weights=True --per=False \
    --train_for_env_steps=5000
```

+ actor_critic_share_weights must be set to True
+ PER (prioritized replay) is automatically disabled in RNN mode; only uniform sequence sampling is supported

---

### Some other important flags
- `--num_agents=N`: Override number of agents
- `--device=cpu`: Train on CPU, use if no CUDA
- `--with_wandb=True`: Enable WB logging
- `--wandb_record_every=N`: Record video every N episodes
- `--use_additional_input=True`: Use optional additional measurement vector to accelerate training

## Record Video

Record high-res video from a trained checkpoint. Requires a full experiment directory in `train_dir` (not just a `.pth` file) since it loads `config.json` to reconstruct the training config.

```
train_dir/<experiment>/
    config.json
    checkpoint_p0/
        checkpoint_*.pth # model weights (or best_*.pth)
```

```bash
# 1 episode at 720p
python -m comrad.record_video --env=stag_hunt_arena --experiment=my_experiment

# 1080p with deterministic actions
python -m comrad.record_video --env=stag_hunt_arena --experiment=my_experiment \
    --resolution=1920x1080 --eval_deterministic=True

# Best checkpoint, 5 episodes, custom output
python -m comrad.record_video --env=stag_hunt_arena --experiment=my_experiment \
    --load_checkpoint_kind=best --max_num_episodes=5 --output_dir=./videos

# Output: `{output_dir}/{video_prefix}_{ENV_INITIALS}_{ALGO}_{resolution}.mp4`
```

## Run in parallel with launcher

Run multiple experiments in parallel

```bash
python -m sample_factory.launcher.run --run=comrad.train_all --backend=processes --max_parallel=4 --pause_between=1
```

## Shared reward

- If the per-agent reward shaping is already the same team scalar, `shared_reward_alpha=1.0` is unnecessary. Keep `shared_reward_alpha=0.0` for benchmarks where preserving local incentives (e.g. preserving social dilemma aspects) is part of the design, especially `stag_hunt_arena` and `foraging_commons`. Leave other scenarios at their scenario default unless you intentionally want to change the benchmark.
- If the task is cooperative but the wrapper emits local shaping, `shared_reward_alpha=1.0` is often the better choice for MAPPO/HAPPO.

`shared_reward_alpha` only matters for actor-critic multi-agent algos such as `MAPPO` and `HAPPO`. Use `shared_reward_alpha=1.0` only when the benchmark is intended to be fully joint and local incentives are not part of the task definition.

| Scenario | Use `shared_reward_alpha=1.0`? | Why |
|---|---:|---|
| `ammo_carrier` | Yes | The intended objective is team defense and resupply; local role-specific shaping should be team-shared for actor-critic runs. |
| `armory_siege` | Yes | The benchmark objective is protecting one shared defense core. |
| `lava_maze` | Yes | Navigator and spectator solve one joint traversal task. |
| `lavapit` | Yes | The bridge-holding / traversal objective is fully cooperative. |
| `platform_chain` | Yes | Progress is joint and tether-constrained by design. |
| `rhythm_sync` | Yes | Success/failure is defined by synchronized team timing. |
| `rhythm_sync_dense` | Yes | Dense shaping still targets the same joint synchronization objective. |
| `stealth_labyrinth` | Yes | Torch and Gunner are asymmetrically coupled around one synchronized relay objective followed by extraction. |
| `foraging_commons` | No | Preserve the commons incentives; the benchmark is about balancing individual harvesting against shared resource collapse. |
| `coop_health_gathering` | No | Scenario is about resource sharing, so no. |
| `stag_hunt_arena` | No | Keep the rabbit option local so the stag-vs-rabbit coordination dilemma remains intact. |
| `smart_enemies` | No | The wrapper already shapes local combat performance; blanket team-sharing is not the default benchmark contract. |
| `dumb_enemies` | No | Same local-combat logic as `smart_enemies`; keep the scenario default unless you intentionally want a joint reward. |
| `coop_puzzle` | Yes | Progress depends on cross-lane plate cooperation and a joint final exit, so actor-critic runs should treat it as a fully shared objective. |

Everything not explicitly listed as `Yes` should stay at the scenario default unless you intentionally want to change the benchmark.
