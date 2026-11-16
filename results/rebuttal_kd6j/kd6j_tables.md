### Placeholder 1: visibility telemetry (MAPPO Pixel-Partial, held-out maps; mean of both agents' viewpoints)

| Scenario | Teammate in FOV cone (%) | Teammate visible (%) | In-cone but occluded (%) | Teammate out of view (%) | Mutual visibility (%) | Teammate visible, random policy (%) |
|---|---|---|---|---|---|---|
| Stag Hunt Arena | 57 | 54 | 5 | 46 | 27 | 46 |
| Rhythm Sync | 20 | 18 | 5 | 82 | 0 | 2 |
| Foraging Commons | 22 | 18 | 18 | 82 | 3 | 27 |
| Co-op Puzzle | 24 | 9 | 63 | 91 | 1 | 11 |
| Platform Chain | 47 | 27 | 44 | 73 | 0 | 31 |
| Armory Siege | 38 | 28 | 27 | 72 | 10 | 21 |
| Co-op Health Gathering | 18 | 16 | 11 | 84 | 2 | 20 |
| Lava Pit | 30 | 29 | 1 | 71 | 0 | 21 |
| Smart Enemies | 24 | 10 | 57 | 90 | 2 | 19 |
| Dumb Enemies | 26 | 15 | 42 | 85 | 3 | 19 |
| Ammo Carrier | 35 | 18 | 46 | 82 | 5 | 11 |
| Stealth Labyrinth | 33 | 30 | 8 | 70 | 2 | 25 |
| Lava Maze | 59 | 50 | 15 | 50 | 0 | 49 |

### Placeholder 1b: per-agent visibility for asymmetric roles

| Scenario | Viewpoint | Teammate in FOV cone (%) | Teammate visible (%) | In-cone but occluded (%) | Teammate in FOV cone, random policy (%) | Teammate visible, random policy (%) |
|---|---|---|---|---|---|---|
| Lava Maze | Navigator (agent 0) | 17 | 0 | 100 | 26.0 | 0.0 |
| Lava Maze | Observer (agent 1) | 100 | 100 | 0 | 100.0 | 97.4 |

### Q1: Partial vs. Shared vision vs. Full (global state) observability (MAPPO, % of ceiling, mean ± 95% CI)

| Scenario | Seeds (P/SV/F) | Partial | Shared vision | Full | Δ Shared vision | Δ Full | Per-seed (P; SV; F) |
|---|---|---|---|---|---|---|---|
| Rhythm Sync | 3/3/3 | 75.1 ± 3.1 | 22.2 ± 78.0 | 53.5 ± 12.9 | -52.9 | -21.5 | 73.9, 74.9, 76.4; 0.1, 8.4, 58.1; 58.7, 48.4, 53.4 |
| Armory Siege | 3/3/3 | 7.9 ± 5.1 | 23.1 ± 29.0 | 12.6 ± 6.4 | +15.2 | +4.7 | 9.2, 5.5, 8.9; 15.1, 17.7, 36.5; 12.1, 15.3, 10.3 |
| Co-op Puzzle | 3/3/3 | 56.2 ± 8.7 | 87.2 ± 43.6 | 26.5 ± 32.6 | +31.0 | -29.7 | 54.8, 60.2, 53.6; 96.0, 67.0, 98.6; 38.4, 28.6, 12.4 |

### Full (own RGB + global state v2, layout pool, 100M) vs. MAPPO Partial from Table 3 (% of ceiling, mean ± 95% CI)

| Scenario | Seeds | Partial (Table 3) | Full | Δ Full − Partial | Welch p | Per-seed Full |
|---|---|---|---|---|---|---|
| Rhythm Sync | 5 | 67.0 ± 4.0 | 68.9 ± 15.9 | +1.9 | 0.759 | 72.4, 86.6, 53.8, 59.4, 72.5 |
| Armory Siege | 5 | 19.2 ± 2.5 | 29.2 ± 12.1 | +10.0 | 0.084 | 40.4, 16.2, 26.1, 37.4, 25.9 |
| Co-op Puzzle | 5 | 76.4 ± 2.2 | 98.1 ± 1.0 | +21.7 | <0.001 | 98.0, 98.4, 97.6, 99.2, 97.2 |

### Placeholder 2: partner occlusion and blind policies (% of ceiling)

| Scenario | Seeds | Normal RGB | Partner sprite masked | Zero-shot blind | Trained blind (t/H only) | Random |
|---|---|---|---|---|---|---|
| Stag Hunt Arena | 1 | 74.2 | 74.1 | 0.0 | — | 0.6 ± 0.2 |
| Rhythm Sync | 2 | 74.4 ± 6.4 | 73.0 ± 29.2 | 0.0 ± 0.3 | 1.8 | 0.0 ± 0.0 |
| Rhythm Sync (QMIX) | 2 | 0.1 ± 1.0 | 0.0 ± 0.3 | 0.0 | — | 0.0 ± 0.0 |
| Foraging Commons | 1 | 12.0 | 11.8 | 9.6 | — | 9.2 ± 0.1 |
| Co-op Puzzle | 2 | 57.5 ± 34.3 | 54.5 ± 29.2 | 6.2 ± 2.5 | — | 4.1 ± 0.5 |
| Platform Chain | 1 | 3.8 | 3.9 | 1.4 | — | 1.4 ± 0.1 |
| Armory Siege | 2 | 7.4 ± 23.4 | 5.4 ± 6.2 | 0.0 ± 21.5 | 0.0 ± 6.2 | 0.0 ± 0.5 |
| Armory Siege (QMIX) | 2 | 14.9 ± 27.9 | 13.0 ± 3.4 | 0.0 ± 0.0 | — | 0.0 ± 0.5 |
| Co-op Health Gathering | 1 | 14.7 | 11.3 | 6.3 | — | 6.2 ± 0.0 |
| Lava Pit | 1 | 8.8 | 8.5 | 0.0 | — | 0.4 ± 0.4 |
| Smart Enemies | 1 | 12.0 | 10.6 | 0.1 | — | 0.1 ± 0.0 |
| Dumb Enemies | 1 | 2.7 | 2.3 | 0.1 | — | 0.1 ± 0.0 |
| Ammo Carrier | 2 | 15.9 ± 11.8 | 15.2 ± 2.0 | 14.0 ± 7.7 | 15.2 ± 5.6 | 13.2 ± 1.2 |
| Ammo Carrier (QMIX) | 2 | 18.0 ± 9.5 | 17.2 ± 11.8 | 13.6 ± 12.4 | — | 13.2 ± 1.2 |
| Stealth Labyrinth | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Stealth Labyrinth (QMIX) | 2 | 0.0 | 0.0 | 0.0 | — | 0.0 |
| Lava Maze | 1 | 0.0 | 0.0 | 0.0 | — | 0.0 |

### Placeholder 3: State vs. Pixel x Partial vs. Full (raw score, mean ± 95% CI; * = Full also adds task scalars)

| Scenario | Algorithm | Pixel / Partial | Pixel / Full* | State / Partial | State / Full |
|---|---|---|---|---|---|
| Rhythm Sync | MAPPO | 0.74 ± 0.06 | 0.91 ± 0.14 | 0.72 ± 0.21 | 0.93 ± 0.27 |
| Rhythm Sync | QMIX | 0.00 ± 0.01 | 0.00 ± 0.04 | 0.00 | 0.00 |
| Armory Siege | MAPPO | 7.36 ± 23.40 | 13.15 ± 14.82 | -9.35 ± 20.91 | -9.04 ± 4.16 |
| Armory Siege | QMIX | 14.93 ± 27.89 | -3.88 ± 43.96 | -9.87 ± 0.63 | -9.10 ± 2.90 |
| Ammo Carrier | MAPPO | 1332 ± 988 | 1767 ± 838 | 1192 ± 1813 | 1130 ± 409 |
| Ammo Carrier | QMIX | 1508 ± 802 | 1327 ± 1501 | 1356 ± 374 | 1236 ± 1422 |
| Stealth Labyrinth | MAPPO | 0.00 | 0.00 | 0.00 | 0.00 |
| Stealth Labyrinth | QMIX | 0.00 | 0.00 | 0.00 | 0.02 ± 0.22 |

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
| Rhythm Sync | MAPPO | 3/3 | 60.2 ± 76.5 | 74.0 ± 2.2 | -13.8 | 25.0, 82.6, 72.9 | 73.9, 74.9, 73.1 |
| Armory Siege | MAPPO | 3/3 | 8.5 ± 1.3 | 5.5 ± 9.2 | +3.0 | 8.7, 8.9, 7.9 | 9.2, 5.5, 1.8 |
| Armory Siege | QMIX | 3/3 | 1.5 ± 6.3 | 13.2 ± 9.3 | -11.2 | 3.5, 0.0, 2.4 | 12.7, 17.1, 9.6 |
| Ammo Carrier | MAPPO | 3/3 | 18.3 ± 2.6 | 16.3 ± 3.0 | +2.0 | 17.3, 19.3, 18.2 | 16.8, 14.9, 17.2 |
| Ammo Carrier | QMIX | 3/3 | 20.6 ± 4.1 | 17.9 ± 1.9 | +2.6 | 21.0, 22.0, 18.8 | 18.7, 17.2, 17.9 |
