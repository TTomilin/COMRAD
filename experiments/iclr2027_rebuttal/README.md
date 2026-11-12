# ICLR 2027 rebuttal studies

Experiments run for the ICLR 2027 rebuttal. All evaluations use the 5 held-out layouts of the benchmark.

## Reviewer kd6J: observability and partner reliance

| Profile (`profiles.py`) | Question |
|---|---|
| `obs_modality_2x2` | {Pixel, State} x {Partial, Full} observations on four representative scenarios |
| `full_observability`, `full_obs_clean` | Shared-vision full observability without implicit extra inputs |
| `full_obs_global`, `full_obs_global_v2` | Global-state full observability (v2: wider task clip, held-weapon ammo, layout pools) |
| `state_entity` | Entity-list state encoding |
| `blind_ablation` | SMACv2-style blind policy |
| `visibility_telemetry` | Partner and target visibility statistics |
| `mask_partner_training` | MAPPO trained from scratch with the partner masked |

Most profiles use 2 seeds (42, 68) and 50M steps (`COMRAD_KD6J_SEEDS`, `COMRAD_KD6J_TRAIN_STEPS`). `full_obs_clean`,
`state_entity` and `full_obs_global_v2` have their own seed lists and filters (`COMRAD_KD6J_CLEAN_SEEDS`,
`COMRAD_STATE_ENTITY_*`, `COMRAD_FULL_OBS_V2_*`); `full_obs_global_v2` follows the Table 3 protocol (5 seeds, 100M).

- Train: `submit_rebuttal_kd6j_train.sh`, `submit_kd6j_mask_partner_train.sh`
  (`run_kd6j_mig_pack.sh` packs several runs on one `gpu_mig` slice).
- Evaluate: `kd6j_heldout_eval_array.sh` runs `eval_kd6j_checkpoint_pack.py` (held-out, partner occlusion, zero-shot
  blind); `submit_rebuttal_kd6j_eval.sh` runs `rebuttal_kd6j_eval.py` (random lower bound, visibility telemetry).
- Report: `aggregate_kd6j_results.py`, then `generate_kd6j_tables.py` into `results/rebuttal_kd6j/`.

`profiles.layout_batch_dir` points global-state v2 runs at the layout-only training pools in `layouts/`, generated
with `generate_layout_training_batch.py` (training seeds disjoint from the held-out seeds).

## Reviewer C2K6: frozen partners and difficulty

- Frozen-partner baseline: profile `frozen_partner_winners` (`submit_c2k6_frozen_partner_train.sh`) trains the
  Table 3 winners missing a Pixel-Partial checkpoint; `c2k6_frozen_partner_array.sh` runs `eval_frozen_partner.py`
  (one seat replaced by a no-op or random partner), and `generate_c2k6_frozen_partner_table.py` writes
  `results/rebuttal_c2k6/`.
- Difficulty grid: `difficulty_grid_eval.py` evaluates one checkpoint zero-shot on a 3x3 spatial x mechanical grid
  (`results/difficulty_grid/`).
