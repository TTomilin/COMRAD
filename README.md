## 1. Installation

Note on python version: You should use python 3.11 and wandb 0.22.x, as there are some issues with wandb 0.24.0 (which is not compatible with python 3.11) in syncing between tensorboard and weave dashboard

### Install dependencies

#### conda/venv/pip

```bash
pip install vizdoom --pre

pip install pyglet "tensorboard>=1.15.0" "tensorboardx>=2.0" "psutil>=5.7.0" "threadpoolctl>=2.0.0" colorlog "signal-slot-mp>=1.0.3,<2.0" filelock "huggingface-hub>=0.10.0,<1.0" pandas opencv-python "pettingzoo" onnx onnxruntime "gymnasium[classic_control]>=0.27,<1.0"
```

**Note:** If runs into error, try installing failed dependencies separately.

#### UV

```bash
uv sync
```

### Setup pre-commit

```bash
# If dont have pre-commit, install first: `pip install pre-commit`

pre-commit install
```

## 2. Training Commands

The primary training script is `sf.train`.

### Algorithms

#### IPPO (Independent PPO)

```bash
python -m sf.train \
    --env=doom_pitfall \
    --algo=APPO \
    --train_for_env_steps=5000
```

---

#### MAPPO

```bash
python -m sf.train \
    --env=doom_pitfall \
    --algo=MAPPO \
    --train_for_env_steps=5000
```

---

#### IDQN (Independent DQN)

```bash
python -m sf.train \
    --env=doom_pitfall \
    --algo=DQN \
    --use_rnn=False \
    --target_update_tau=1.0 \
    --train_for_env_steps=5000
```

For training on HPC, I tuned with these parameters (this config technically edges 32GB RAM):
```bash
# Single agent
python -m sf.train --env=doom_pitfall --algo=DQN --train_for_seconds=21600 --num_workers=16 --num_envs_per_worker=16 --batch_size=2048 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_project=marl_vizdoom --use_rnn=False --learning_starts=50000 --dqn_batch_size=256 --replay_buffer_size=200000 --target_update_interval=2500 --epsilon_decay_steps=4000000 --epsilon_end=0.005 --per_beta_frames=2000000 --num_agents=1 --dqn_max_updates_per_batch=4 --target_update_tau=0.005 --dqn_reward_clip=1.0 --train_frequency=8

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
python -m sf.train \
    --env=doom_pitfall \
    --algo=QMIX \
    --mixer=vdn \
    --use_rnn=False \
    --train_for_env_steps=5000
```

---

#### QMIX
```bash
python -m sf.train \
    --env=doom_pitfall \
    --algo=QMIX \
    --mixer=qmix \
    --use_rnn=False \
    --train_for_env_steps=5000
```

---

#### VDN/QMIX + RNN (GRU)
```bash
python -m sf.train \
    --env=doom_pitfall \
    --algo=QMIX \
    --mixer=vdn/qmix \
    --use_rnn=True --rnn_type=gru --rnn_size=64 --rollout=16 \
    --actor_critic_share_weights=True --per=False \
    --train_for_env_steps=5000
```

Notes: PER (prioritized replay) is automatically disabled in RNN mode; only uniform sequence sampling is supported

---

### Some other important flags
- `--num_agents=N`: Override number of agents
- `--device=cpu`: Train on CPU, use if no CUDA
- `--with_wandb=True`: Enable WB logging
- `--wandb_record_every=N`: Record video every N episodes

## 3. HPC Configuration

**Recommended Specs:** 64GB RAM (32GB may run out quite quick for long training).

### MAPPO
```bash
python -m sf.train \
  --env=doom_pitfall \
  --algo=APPO \
  --train_for_seconds=18000 \
  --env_frameskip=4 \
  --use_rnn=True \
  --num_workers=16 \
  --num_envs_per_worker=8 \
  --num_policies=1 \
  --batch_size=1024 \
  --wide_aspect_ratio=False \
  --use_mappo \
  --with_wandb=True \
  --wandb_dir=. \
  --wandb_record_every=10 \
  --batched_sampling=True \
  --experiment=pitfall_0 <-- You can change or remove this flag
```

### QMIX
```bash
python -m sf.train --env=doom_pitfall --algo=QMIX --mixer=qmix --train_for_seconds=3600 --num_workers=8 --num_envs_per_worker=8 --policy_workers_per_policy=2 --batch_size=2048 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_project=marl_vizdoom --use_rnn=False --gamma=0.99 --learning_starts=50000 --qmix_buffer_batch_size=256 --replay_buffer_size=200000 --epsilon_decay_steps=4000000 --epsilon_end=0.005 --learning_rate=0.0001 --num_agents=2 --dqn_max_updates_per_batch=4 --target_update_tau=0.005 --use_huber_loss=True --q_value_clamp=100 --train_frequency=8 --batched_sampling=True
```

### QMIX + RNN (GRU)
rnn_size is only 256 as mixer also adds params
```bash
python -m sf.train --env=doom_pitfall --algo=QMIX --mixer=qmix --train_for_seconds=43200 --num_workers=8 --num_envs_per_worker=8 --policy_workers_per_policy=2 --batch_size=2048 --env_frameskip=4 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_project=marl_vizdoom --use_rnn=True --rnn_type=gru --rnn_size=256 --rollout=32 --gamma=0.99 --learning_starts=50000 --qmix_sequence_batch_size=64 --replay_buffer_size=500000 --epsilon_decay_steps=20000000 --epsilon_end=0.005 --learning_rate=0.0001 --num_agents=2 --dqn_max_updates_per_batch=4 --target_update_tau=0.005 --use_huber_loss=True --q_value_clamp=100 --train_frequency=8 --batched_sampling=True --per=False --actor_critic_share_weights=True
```

## 4. Run in parallel with launcher

Run multiple experiments in parallel

```bash
python -m sample_factory.launcher.run --run=sf.train_all --backend=processes --max_parallel=4 --pause_between=1
```

## 5. Main files
- **Scenarios:** `sf/doom/doom_utils.py` (Defines `DOOM_ENVS`)
- **Reward Shaping:** `sf/doom/wrappers/scenario_wrappers/`
- **Player init, ZDoom flags:** `sf/doom/multiplayer/doom_multiagent.py`
- **Wrapper for game instances:** `sf/doom/multiplayer/doom_multiagent_wrapper.py`
- **Config:** `sample_factory/cfg/cfg.py` + `sf/doom/doom_params.py`

## 6. Side Notes
- Default obs shape is `(3, 72, 128)`.
- To avoid nested folders, set `--wandb_dir` to root (`--wandb_dir=.`)
- Use `--record_to` to output frames
