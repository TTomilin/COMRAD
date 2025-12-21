## Install and run
<!-- Install sample factory 2.1.3 with (last pypi pkg released was 2 years ago, which was 2.1.1):
```
pip install git+https://github.com/alex-petrenko/sample-factory.git

# or at specific commit
pip install git+https://github.com/alex-petrenko/sample-factory.git@8008921cd8823f4c53f68afea5b9e9da040d0e4a
``` -->

We need to modify SF's codebase to adapt MAPPO, which is rather inconvenient with monkey patch. Thus, sample_factory is cloned directly (similar to HASARD's approach). Install all dependencies required by SF with (These will be in pyproject.toml and setup.py when we get our own independent repo):
```
pip install pyglet "tensorboard>=1.15.0" "tensorboardx>=2.0" "psutil>=5.7.0" "threadpoolctl>=2.0.0" colorlog "signal-slot-mp>=1.0.3,<2.0" filelock "huggingface-hub>=0.10.0,<1.0" pandas opencv-python "pettingzoo[classic]" onnx onnxruntime "gymnasium[classic_control]>=0.27,<1.0"
```

## Training Commands

### Run locally with IPPO
```bash
python -m sf.train \
  --env=doom_pitfall \
  --algo=APPO \
  --train_for_env_steps=10000 \
  --env_frameskip=4 \
  --use_rnn=True \
  --num_workers=4 \
  --num_envs_per_worker=4 \
  --num_policies=1 \
  --batch_size=1024 \
  --wide_aspect_ratio=False \
  --experiment=pitfall_0 \
  --with_wandb=True \
  --wandb_dir=. \
  --wandb_record_every=1
```

### Run locally with MAPPO
Add `--use_mappo` (and you can add `--num_agents=...` to config more agents):
```bash
python -m sf.train \
  --env=doom_pitfall \
  --algo=APPO \
  --train_for_env_steps=5000 \
  --env_frameskip=4 \
  --use_rnn=True \
  --num_workers=4 \
  --num_envs_per_worker=2 \
  --num_policies=1 \
  --wide_aspect_ratio=False \
  --use_mappo \
  --with_wandb=True \
  --wandb_dir=. \
  --wandb_record_every=1
```

### Run on HPC
Use `--train_for_seconds` instead of `--train_for_env_steps` and increase workers:

```bash
python -m sf.train \
  --env=doom_pitfall \
  --algo=APPO \
  --train_for_seconds=1800 \
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
  --experiment=pitfall_0 <-- You can change or remove this flag
```

System recommendation for hpc: 64gb RAM (32gb runs out in 4-5h)

> **Note:** If you don't have CUDA, add `--device=cpu` flag.

## Run with launcher

```bash
python -m sample_factory.launcher.run --run=sf.train_all --backend=processes --max_parallel=4 --pause_between=1
```

This will run multiple experiments with defined seeds in parallel. Might cause issues if run locally without enough resources, but it parallelizes multiple experiments.

## Runs
+ 7 cores 16gb 2gpu
+ Four runs with 4 configurations

### 1. Force respawn on (Agents respawn immediately after they die), reset episode on time limit/batch
+ Finished map but agent runs back after reaching the end of the tunnel as no termination condition at the end
+ To recreate this assignment, add the flag `forcerespawn` to coop agents in `doom_multiagent.py`, and remove the use of `wipe_when_one_die` in `doom_multiagent_wrapper.py`
+ Link: https://wandb.ai/khoi-eindhoven-university-of-technology/marl_vizdoom/runs/pitfall_hpc_0_20251105_191320_788324

### 2. Force respawn off, terminates when one agent dies (to make it more 'cooperative')
+ Train much slower than 1, training stopped because there was 1h time limit
+ Didn't finish map
+ I used this setting as a more realistic scenario to see how it performs, but no credit assignment so......
+ Link: https://wandb.ai/khoi-eindhoven-university-of-technology/marl_vizdoom/runs/pitfall_hpc_10_20251106_190442_448800

### 3. Run (2) but reward when all alive, not punish all agents when one die
+ Finishes map in like 13 mins
+ Agent might learn to sacrifice for exploration I think
+ If we punish all agents when one die, it runs much much slower ([run with this config](https://wandb.ai/khoi-eindhoven-university-of-technology/marl_vizdoom/runs/pitfall_hpc_100_20251107_162500_027035)). To enable this, edit `pitfall_reward_shaping.py`
+ Finished in 13m: https://wandb.ai/khoi-eindhoven-university-of-technology/temp/runs/pitfall_hpc_10_20251106_190442_448800
+ Secondary link: https://wandb.ai/khoi-eindhoven-university-of-technology/marl_vizdoom/runs/pitfall_hpc_1000_20251107_162846_064626

## Maybe useful notes
+ obs is (3, 72, 128)
+ Add more scenarios in `doom_utils.py`, this is where most important stuffs gets called
+ `wrappers` folder is for reward shaping
+ `reward_shaping.py` is quite for competitive tasks
+ `doom_multiagent.py` inits the player, so there you changes zdoom flags/configs
+ `doom_multiagent_wrapper.py` wraps the agent in a game instance
+ We can use `--record_to` flag, built-in from sample factory, but it outputs frames (png). Read `train.py` comments for guide to use it
+ To prevent nested wandb folders, set `--wandb_dir` to root dir or change `sample_factory/cfg/cfg.py` to default `os.getcwd()`
