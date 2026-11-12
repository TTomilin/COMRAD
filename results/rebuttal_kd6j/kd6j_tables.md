### Placeholder 1: visibility telemetry (MAPPO Pixel-Partial, held-out maps; mean of both agents' viewpoints)

| Scenario | Teammate in FOV cone (%) | Teammate visible (%) | In-cone but occluded (%) | Teammate out of view (%) | Mutual visibility (%) | Teammate visible, random policy (%) |
|---|---|---|---|---|---|---|
| Stag Hunt Arena | 56 | 53 | 5 | 47 | 26 | 46 |
| Rhythm Sync | 20 | 19 | 5 | 81 | 0 | 2 |
| Foraging Commons | 22 | 17 | 19 | 83 | 3 | 27 |
| Co-op Puzzle | 24 | 8 | 66 | 92 | 1 | 11 |
| Platform Chain | 47 | 27 | 42 | 73 | 0 | 31 |
| Armory Siege | 38 | 27 | 28 | 73 | 9 | 21 |
| Co-op Health Gathering | 17 | 15 | 12 | 85 | 2 | 20 |
| Lava Pit | 29 | 29 | 2 | 71 | 0 | 21 |
| Smart Enemies | 25 | 11 | 54 | 89 | 2 | 19 |
| Dumb Enemies | 26 | 13 | 47 | 87 | 3 | 19 |
| Ammo Carrier | 34 | 17 | 48 | 83 | 4 | 11 |
| Stealth Labyrinth | 33 | 31 | 7 | 69 | 1 | 25 |
| Lava Maze | 63 | 50 | 21 | 50 | 0 | 49 |

### Placeholder 1b: per-agent visibility for asymmetric roles

| Scenario | Viewpoint | Teammate in FOV cone (%) | Teammate visible (%) | In-cone but occluded (%) | Teammate in FOV cone, random policy (%) | Teammate visible, random policy (%) |
|---|---|---|---|---|---|---|
| Lava Maze | Navigator (agent 0) | 26 | 0 | 100 | 26.0 | 0.0 |
| Lava Maze | Observer (agent 1) | 100 | 100 | 0 | 100.0 | 97.4 |

### Q1: Partial vs. Shared vision vs. Full (global state) observability (MAPPO, % of ceiling, mean ± 95% CI)

| Scenario | Seeds (P/SV/F) | Partial | Shared vision | Full | Δ Shared vision | Δ Full | Per-seed (P; SV; F) |
|---|---|---|---|---|---|---|---|
| Rhythm Sync | 3/3/3 | 74.4 ± 8.1 | 24.9 ± 86.1 | 57.4 ± 8.9 | -49.5 | -17.0 | 75.4, 77.0, 70.8; 0.1, 10.2, 64.5; 61.5, 55.2, 55.5 |
| Armory Siege | 3/3/3 | 6.0 ± 2.4 | 27.5 ± 33.0 | 13.1 ± 8.4 | +21.6 | +7.1 | 5.7, 5.2, 7.0; 18.3, 21.5, 42.8; 12.7, 16.6, 9.9 |
| Co-op Puzzle | 3/3/3 | 54.9 ± 6.1 | 88.1 ± 38.5 | 48.6 ± 47.2 | +33.1 | -6.3 | 54.4, 57.6, 52.8; 96.0, 70.2, 98.0; 70.4, 35.6, 39.8 |

### Full (own RGB + global state v2, layout pool, 100M) vs. MAPPO Partial from Table 3 (% of ceiling, mean ± 95% CI)

| Scenario | Seeds | Partial (Table 3) | Full | Δ Full − Partial | Welch p | Per-seed Full |
|---|---|---|---|---|---|---|
| Rhythm Sync | 5 | 67.0 ± 4.0 | 65.8 ± 15.0 | -1.2 | 0.845 | 71.3, 83.7, 51.9, 62.0, 60.4 |
| Armory Siege | 5 | 19.2 ± 2.5 | 29.2 ± 12.4 | +10.0 | 0.087 | 39.4, 15.9, 27.4, 38.7, 24.7 |
| Co-op Puzzle | 5 | 76.4 ± 2.2 | 98.3 ± 1.0 | +21.9 | <0.001 | 98.6, 98.2, 97.2, 99.4, 98.2 |

### Placeholder 2: partner occlusion and blind policies (% of ceiling)

| Scenario | Seeds | Normal RGB | Partner sprite masked | Zero-shot blind | Trained blind (t/H only) | Random |
|---|---|---|---|---|---|---|
| Stag Hunt Arena | 1 | 74.0 | 74.0 | 0.0 | — | 0.6 ± 0.2 |
| Rhythm Sync | 2 | 76.2 ± 9.8 | 72.4 ± 13.7 | 0.1 ± 1.0 | 1.7 ± 1.9 | 0.0 ± 0.0 |
| Rhythm Sync (QMIX) | 2 | 0.0 ± 0.3 | 0.0 ± 0.3 | 0.0 | — | 0.0 ± 0.0 |
| Foraging Commons | 1 | 12.0 | 11.9 | 9.6 | — | 9.2 ± 0.1 |
| Co-op Puzzle | 2 | 56.0 ± 20.3 | 56.3 ± 11.4 | 6.2 ± 22.9 | — | 4.1 ± 0.5 |
| Platform Chain | 1 | 3.9 | 3.8 | 1.4 | — | 1.4 ± 0.1 |
| Armory Siege | 2 | 5.4 ± 3.0 | 5.8 ± 21.3 | 0.0 ± 22.6 | 0.0 ± 5.0 | 0.0 ± 0.5 |
| Armory Siege (QMIX) | 2 | 14.3 ± 21.7 | 14.3 ± 46.8 | 0.0 ± 0.4 | — | 0.0 ± 0.5 |
| Co-op Health Gathering | 1 | 14.9 | 12.6 | 6.4 | — | 6.2 ± 0.0 |
| Lava Pit | 1 | 8.5 | 8.8 | 0.0 | — | 0.4 ± 0.4 |
| Smart Enemies | 1 | 10.7 | 10.5 | 0.0 | — | 0.1 ± 0.0 |
| Dumb Enemies | 1 | 3.0 | 2.9 | 0.1 | — | 0.1 ± 0.0 |
| Ammo Carrier | 2 | 15.5 ± 5.1 | 15.4 ± 9.3 | 13.6 ± 8.9 | 14.6 ± 2.7 | 13.2 ± 1.2 |
| Ammo Carrier (QMIX) | 2 | 18.3 ± 8.0 | 17.3 ± 0.6 | 12.6 ± 21.5 | — | 13.2 ± 1.2 |
| Stealth Labyrinth | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Stealth Labyrinth (QMIX) | 2 | 0.0 | 0.0 | 0.0 | — | 0.0 |
| Lava Maze | 1 | 0.0 | 0.0 | 0.0 | — | 0.0 |

### Placeholder 3: State vs. Pixel x Partial vs. Full (raw score, mean ± 95% CI; * = Full also adds task scalars)

| Scenario | Algorithm | Pixel / Partial | Pixel / Full* | State / Partial | State / Full |
|---|---|---|---|---|---|
| Rhythm Sync | MAPPO | 0.76 ± 0.10 | 0.93 ± 0.27 | 0.76 ± 0.55 | 0.92 ± 0.26 |
| Rhythm Sync | QMIX | 0.00 ± 0.00 | 0.00 ± 0.02 | 0.00 | 0.00 |
| Armory Siege | MAPPO | 5.43 ± 3.03 | 13.59 ± 33.46 | -9.17 ± 19.69 | -8.20 ± 0.71 |
| Armory Siege | QMIX | 14.27 ± 21.66 | -2.44 ± 54.70 | -9.89 ± 1.25 | -8.80 ± 7.27 |
| Ammo Carrier | MAPPO | 1306 ± 426 | 1592 ± 956 | 1157 ± 1948 | 1133 ± 399 |
| Ammo Carrier | QMIX | 1533 ± 670 | 1334 ± 2429 | 1384 ± 343 | 1232 ± 1780 |
| Stealth Labyrinth | MAPPO | 0.00 | 0.00 | 0.00 | 0.00 |
| Stealth Labyrinth | QMIX | 0.00 | 0.00 | 0.00 | 0.02 ± 0.29 |

### Placeholder 4: ceiling-normalized Table 3 (% of ceiling)

| Scenario | Random | IPPO | MAPPO | HAPPO | IDQN | VDN | QMIX | QPLEX-D | QPLEX-Q |
|---|---|---|---|---|---|---|---|---|---|
| Stag Hunt Arena | 0.6 | 48.4 | **70.8** | 7.3 | 39.4 | 50.3 | 0.7 | 4.2 | 9.3 |
| Rhythm Sync | 0.0 | 57.0 | **67.0** | 21.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Foraging Commons | 9.2 | 11.1 | 11.9 | 10.5 | 10.4 | 11.1 | **12.1** | 9.1 | 11.2 |
| Co-op Puzzle | 4.1 | 59.6 | **76.4** | 43.0 | 0.4 | 1.2 | 0.0 | 0.0 | 1.0 |
| Platform Chain | 1.4 | 4.2 | **10.3** | 9.8 | 1.5 | 2.3 | 2.0 | 7.1 | 2.1 |
| Armory Siege | 0.0 | 16.0 | 19.2 | 0.0 | 10.5 | 17.8 | **20.4** | 1.1 | 13.6 |
| Co-op Health Gathering | 6.2 | **52.0** | 26.7 | 9.6 | 7.7 | 27.0 | 8.0 | 9.1 | 13.0 |
| Lava Pit | 0.4 | 8.2 | **9.1** | **9.1** | 8.5 | 7.1 | 8.7 | 0.0 | **9.1** |
| Smart Enemies | 0.1 | 13.6 | **17.2** | 8.0 | 4.0 | 5.7 | 0.7 | 0.0 | 0.8 |
| Dumb Enemies | 0.1 | **13.6** | 2.6 | 1.1 | 1.2 | 3.0 | 1.3 | 0.0 | 2.1 |
| Ammo Carrier | 13.2 | **83.7** | 16.1 | 15.6 | 45.2 | 29.9 | 27.9 | 18.0 | 24.3 |
| Stealth Labyrinth | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **21.0** | 0.0 | 0.0 |
| Lava Maze | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### State (entity encoding) vs. Pixel, Partial observability (% of ceiling, mean ± 95% CI)

| Scenario | Algorithm | Seeds (S/P) | State | Pixel | Δ State − Pixel | Per-seed State | Per-seed Pixel |
|---|---|---|---|---|---|---|---|
| Rhythm Sync | MAPPO | 3/3 | 61.6 ± 80.1 | 74.3 ± 8.6 | -12.7 | 24.4, 81.6, 78.8 | 75.4, 77.0, 70.4 |
| Armory Siege | MAPPO | 3/3 | 8.3 ± 3.9 | 4.2 ± 5.3 | +4.1 | 8.4, 9.8, 6.7 | 5.7, 5.2, 1.7 |
| Armory Siege | QMIX | 3/3 | 2.9 ± 7.5 | 12.4 ± 9.2 | -9.5 | 6.1, 0.1, 2.4 | 12.6, 16.0, 8.6 |
| Ammo Carrier | MAPPO | 3/3 | 18.2 ± 3.1 | 15.5 ± 1.1 | +2.7 | 17.0, 19.5, 18.1 | 15.9, 15.1, 15.3 |
| Ammo Carrier | QMIX | 3/3 | 19.7 ± 2.7 | 18.3 ± 1.6 | +1.4 | 20.3, 20.3, 18.4 | 17.6, 18.9, 18.3 |
