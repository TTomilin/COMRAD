# Very Difficult learnability references

Shows that the Very Difficult scenarios (Stealth Labyrinth, Lava Maze) are learnable once the coordination
bottleneck is removed (paper appendix `very_difficult`, table `tab:very_difficult_reference`). Algorithms,
hyper-parameters and seeds (42, 68, 81, 97, 154) follow the benchmark protocol.

## Profiles (`profiles.py`)

| Profile | Scenarios | Algorithms | Budget |
|---|---|---|---|
| `very_difficult_scripted` | Scripted scout / scripted observer, learned follower | IPPO, IDQN | 50M (single learner) |
| `very_difficult_converse` | Scripted gunner / scripted navigator, learned leader | IPPO, IDQN | 50M (single learner) |
| `very_difficult_oracle` | Oracle message, oracle hint, always lit, shared vision | MAPPO, QMIX | 100M |

A single learner behind a scripted partner gets the same per-agent experience at half the counted env steps, so its
budget and IDQN epsilon horizon are halved.

## Pipeline

1. Train: `COMRAD_EXPERIMENT_PROFILE=<profile> ./comrad/send.sh 3`.
2. Evaluate on the held-out layouts: `submit_very_difficult_eval.sh` (one SLURM job per env and algorithm, runs
   `very_difficult_heldout_eval.sh`).
3. Fully scripted upper references: `fully_scripted_reference_eval.py`.
4. Train-layout diagnostic (generalisation gap): `make_train_wad_diag_batches.py`, then evaluate those batches.
5. Aggregate into the paper table: `very_difficult_reference_table.py` writes
   `results/very_difficult_reference_summary.json` and `results/very_difficult_reference_rows.tex`
   (`--row-set scripted_partner` builds the scripted-partner rows instead). It stops if a row of the chosen row set
   lacks any of the five training seeds; `--allow-incomplete` writes the available rows.
6. Optional videos: `record_scripted_partner_videos.sh` (mp4 files are not tracked).

`very_difficult_cmd.py` prints the exact benchmark command of any scenario and algorithm, with optional resource
overrides, for local smoke runs.
