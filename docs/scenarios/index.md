# Scenarios

```python
import comrad

for s in comrad.list_scenarios():
    print(f"{s.name:32} {s.num_agents} agents  score: {s.score}")
```

Every scenario is a first-person, partially observed, multi-agent task. Each agent sees only its own `128×72` RGB view and acts through a small tuple of discrete action heads. Scenarios are grouped by the cooperative capability they stress.

| Scenario | Players | Score (`true_objective`) | Capability tested |
|---|---|---|---|
| [Stag Hunt Arena](stag_hunt_arena.md) | 2–4 | Stags killed | Convention formation, equilibrium selection |
| [Rhythm Sync](rhythm_sync.md) | 2 | Stages completed | Temporal coordination |
| [Foraging Commons](foraging_commons.md) | 2–4 | Survival time | Commons dilemma, public goods |
| [Co-op Puzzle](coop_puzzle.md) | 2 | Completed puzzle pairs | Turn-taking, partner affordances |
| [Platform Chain](platform_chain.md) | 2 | Joint checkpoint progress | Long-horizon joint movement |
| [Armory Siege](armory_siege.md) | 2–8 | Shaped episodic return | Combat coordination, resource routing |
| [Co-op Health Gathering](coop_health_gathering.md) | 2–8 | Survival time | Resource sharing |
| [Lava Pit](lavapit.md) | 2 | Joint bridges cleared | Reciprocal assistance |
| [Smart Enemies](smart_enemies.md) | 2–4 | Enemies killed | Spacing, target prioritisation |
| [Dumb Enemies](dumb_enemies.md) | 2–4 | Enemies killed | Cooperative pursuit |
| [Ammo Carrier](ammo_carrier.md) | 2 | Survival time | Role-specialised logistics |
| [Stealth Labyrinth](stealth_labyrinth.md) | 2 | Enemies destroyed / total | Extreme role asymmetry |
| [Lava Maze](lava_maze.md) | 2 | Maze levels cleared | Emergent communication |

`comrad.list_scenarios()` also returns a few easier or alternative variants (`platform_chain_easy`, `lava_maze_simple`, `rhythm_sync_dense`, `stag_hunt_arena_continuous*`). They are useful for debugging and curricula, but they are not part of the main results table.

## Map generation

The maps were generated procedurally. The generators live in the separate `DoomGen/` folder of the repository and are not part of the `comrad-bench` package. The package ships ready-made `.wad` and `.cfg` files under `comrad/scenarios/`.
