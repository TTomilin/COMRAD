| Scenario | Best method | Both learned (%) | Partner no-op (%) | Partner random (%) | Random policy (%) |
|---|---|---|---|---|---|
| Stag Hunt Arena | MAPPO | 74.1 | 17.1 | 20.4 | 0.6 |
| Rhythm Sync | MAPPO | 74.7 | 0.0 | 0.4 | 0.0 |
| Foraging Commons | MAPPO | 11.9 | 8.7 | 9.2 | 9.2 |
| Co-op Puzzle | MAPPO | 57.0 | 0.0 | 17.2 | 4.1 |
| Platform Chain | MAPPO | 3.7 | 1.1 | 1.5 | 1.4 |
| Armory Siege | QMIX | 13.1 | 9.2 | 11.7 | 0.0 |
| Co-op Health Gathering | IPPO | 29.7 | 6.1 | 6.4 | 6.2 |
| Lava Pit | MAPPO | 9.0 | 0.0 | 1.5 | 0.4 |
| Smart Enemies | MAPPO | 11.7 | 5.8 | 4.8 | 0.1 |
| Dumb Enemies | IPPO | 7.7 | 1.4 | 2.3 | 0.1 |
| Ammo Carrier | IPPO | 20.5 | 12.4 | 15.7 | 13.2 |
| Stealth Labyrinth | QMIX | 0.0 | 0.0 | 0.0 | 0.0 |
| Lava Maze | MAPPO | 0.0 | 0.0 | 0.0 | 0.0 |

Diagnostics (% of ceiling; CIs are Student-t over 5 held-out layouts):

| Scenario | Seed (of candidates) | Steps | Random policy (%) | Both learned (%, t-CI) | No-op seat 0 / 1 (%) | Random seat 0 / 1 (%) | Drop no-op / random (pp, paired t-CI) | Retention no-op / random | Category | Discarded crash eps |
|---|---|---|---|---|---|---|---|---|---|---|
| Stag Hunt Arena | 42 (of 1) | 50020352 | 0.6 | 74.1 ± 0.7 | 15.8 / 18.4 | 20.7 / 20.1 | +57.0 ± 2.7 / +53.7 ± 2.4 | 0.22 / 0.27 | partner-dependent | noop_seat0 2, noop_seat1 2, random_seat0 1, random_seat1 1 |
| Rhythm Sync | 68 (of 1) | 50020352 | 0.0 | 74.7 ± 6.1 | 0.0 / 0.0 | 0.6 / 0.1 | +74.6 ± 6.1 / +74.3 ± 6.2 | -0.00 / 0.00 | drops to the random level | 0 |
| Foraging Commons | 42 (of 1) | 50020352 | 9.2 | 11.9 ± 0.3 | 8.7 / 8.7 | 9.1 / 9.2 | +3.2 ± 0.3 / +2.7 ± 0.3 | -0.19 / -0.01 | drops to the random level | 0 |
| Co-op Puzzle | 68 (of 1) | 50020352 | 4.1 | 57.0 ± 2.8 | 0.0 / 0.0 | 18.4 / 16.0 | +57.0 ± 2.8 / +39.8 ± 6.9 | -0.08 / 0.25 | partner-dependent | 0 |
| Platform Chain | 42 (of 1) | 50020352 | 1.4 | 3.7 ± 0.5 | 0.1 / 2.0 | 1.1 / 2.0 | +2.7 ± 0.4 / +2.2 ± 0.5 | -0.16 / 0.04 | drops to the random level | 0 |
| Armory Siege | 68 (of 1) | 50020352 | 0.0 | 13.1 ± 8.0 | 9.3 / 9.1 | 14.2 / 9.2 | +3.9 ± 6.6 / +1.4 ± 10.5 | 0.70 / 0.89 | partner-dependent | 0 |
| Co-op Health Gathering | 68 (of 2) | 100024320 | 6.2 | 29.7 ± 22.7 | 6.1 / 6.1 | 6.4 / 6.3 | +23.5 ± 19.0 / +23.3 ± 18.9 | -0.00 / 0.01 | drops to the random level | 0 |
| Lava Pit | 42 (of 1) | 50020352 | 0.4 | 9.0 ± 0.3 | 0.0 / 0.0 | 1.3 / 1.8 | +9.0 ± 0.3 / +7.5 ± 0.6 | -0.04 / 0.14 | drops to the random level | 0 |
| Smart Enemies | 42 (of 1) | 50020352 | 0.1 | 11.7 ± 2.4 | 5.5 / 6.1 | 4.4 / 5.2 | +5.9 ± 1.5 / +6.8 ± 1.1 | 0.49 / 0.41 | partner-dependent | 0 |
| Dumb Enemies | 42 (of 2) | 100024320 | 0.1 | 7.7 ± 4.7 | 1.4 / 1.4 | 2.2 / 2.4 | +6.3 ± 4.6 / +5.4 ± 4.4 | 0.18 / 0.29 | partner-dependent | 0 |
| Ammo Carrier | 68 (of 2) | 100024320 | 13.2 | 20.5 ± 7.9 | 13.8 / 11.1 | 17.4 / 14.0 | +8.1 ± 7.6 / +4.8 ± 5.5 | -0.10 / 0.34 | partner-dependent | both_learned 1, random_seat0 1 |
| Stealth Labyrinth | 42 (of 3) | 100024320 | 0.0 | 0.0 ± 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | +0.0 ± 0.0 / +0.0 ± 0.0 | — | near the random floor already (inconclusive) | 0 |
| Lava Maze | 42 (of 1) | 50020352 | 0.0 | 0.0 ± 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | +0.0 ± 0.0 / +0.0 ± 0.0 | — | near the random floor already (inconclusive) | 0 |

- near the random floor already (inconclusive): Stealth Labyrinth, Lava Maze
- drops to the random level: Rhythm Sync, Foraging Commons, Platform Chain, Co-op Health Gathering, Lava Pit
- partner-dependent: Stag Hunt Arena, Co-op Puzzle, Armory Siege, Smart Enemies, Dumb Enemies, Ammo Carrier
- partner-insensitive: —
