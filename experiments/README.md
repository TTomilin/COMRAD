# Experiments

Studies built on top of the COMRAD benchmark that are not part of the `comrad` package. Each subfolder holds the
training profiles, evaluation scripts and SLURM submitters of one study; results are written to `results/`.

| Folder | Study | Results |
|---|---|---|
| [`very_difficult/`](very_difficult/README.md) | Learnability references for the Very Difficult tier (scripted partners, oracles) | `results/heldout_very_difficult/`, `results/very_difficult_*` |
| [`iclr2027_rebuttal/`](iclr2027_rebuttal/README.md) | ICLR 2027 rebuttal studies (reviewers kd6J and C2K6) | `results/rebuttal_kd6j/`, `results/rebuttal_c2k6/`, `results/difficulty_grid/` |

Shared code lives in `common.py` (algorithm lookup, budget override, canonical QMIX replay buffer) and the
Sample Factory launcher entry point in `runs.py`. The paper's main benchmark profiles stay in `comrad/train_all.py`.

## Training

Pick a profile from any study with `COMRAD_EXPERIMENT_PROFILE`:

```bash
COMRAD_EXPERIMENT_PROFILE=very_difficult_oracle uv run python -m sample_factory.launcher.run \
    --run=experiments.runs --backend=processes --max_parallel=1 --pause_between=1
```

On the HPC, `COMRAD_EXPERIMENT_PROFILE=<profile> ./comrad/send.sh 3` submits one SLURM job per run. Every `COMRAD_*`
variable of the local shell (seed or env filters used by some profiles) is forwarded to the launcher.

## Evaluation

All held-out evaluations reuse `scripts/heldout_wad_evaluation.py` (5 held-out layouts, seeds
1009, 2027, 3037, 4051, 5099). Run every script from the repository root.

## Tests

```bash
uv run python -m pytest experiments/tests
```
