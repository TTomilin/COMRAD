# `comrad/scenarios/`

This directory stores the runtime benchmark assets consumed by COMRAD training and evaluation. Each scenario is represented by a `.cfg` plus `.wad` pair, and some scenarios also ship pre-generated batch directories for curriculum or layout-variation experiments.

## Contents

- One `.cfg` plus one `.wad` per scenario, for example `armory_siege.cfg` and `armory_siege.wad`
- Pre-generated task pools such as `batch_armory/`, `batch_lava_maze/`, and `batch_platform_chain_curriculum/`
- `batch_registry.json` files that enumerate the WAD instances available to a pool
- Optional pool metadata such as interestingness graphs for curriculum experiments

## Scenario catalog

The descriptions below summarize the benchmark tasks using the paper's scenario definitions together with the benchmark notes in `DoomGen/examples/benchmark/README.md`.

| Scenario | Players | Benchmark score | Description |
| --- | --- | --- | --- |
| `stag_hunt_arena` | 2-4 | Stags killed | A stag-hunt coordination dilemma between reliable solo rabbit rewards and a larger stag payoff that only becomes available through simultaneous commitment. Tests convention formation, trust, and whether policies choose the payoff-dominant cooperative equilibrium. |
| `rhythm_sync` | 2 | Stages completed | Separated agents must activate paired switches inside strict shared timing windows, with unilateral activation causing immediate failure. Tests temporal coordination, implicit synchronization, and partial-observability adaptation. |
| `foraging_commons` | 2-4 | Survival time | Harvesting improves short-term health but depletes a shared resource process, while cleanup restores the commons at an opportunity cost. Tests tragedy-of-the-commons dynamics, public-goods provision, and cooperator versus free-rider behavior. |
| `coop_puzzle` | 2 | Completed puzzle pairs | Two players progress through paired lanes by alternately unlocking one another's path, with periodic stages requiring simultaneous occupancy on both sides. Tests turn-taking, partner-specific affordance use, and synchronized progression. |
| `platform_chain` | 2 | Joint checkpoint progress | Players climb a vertical route while constrained by a physical tether that punishes overstretch and poor spatial synchronization. Tests long-horizon joint movement, proximity maintenance, and coordinated traversal under hazard variation. |
| `armory_siege` | 2-8 | Shaped episodic return | A team defends a central core while rotating through gated weapon, ammo, and medical rooms. Tests combat coordination, resource routing, rotating roles, and defensive positioning under sustained pressure. |
| `coop_health_gathering` | 2-8 | Survival time | A cooperative survival task in a toxic maze where players share a tether and must manage exploration, kit collection, and chain slack while under continuous environmental damage. |
| `lavapit` | 2 | Joint bridges cleared | A reciprocal bridge-handoff traversal task where one agent must hold support for the other, then the roles invert before monotone progress can continue. Tests temporary role asymmetry, reciprocal assistance, and reliable conversion of split states into joint progress. |
| `smart_enemies` | 2-4 | Enemies killed | Players fight fleeing enemies that become harder to finish when several agents converge on the same target. Tests combat spacing, target prioritization, and whether teams learn to avoid over-concentrated pressure. |
| `dumb_enemies` | 2-4 | Enemies killed | Uses the same arena family as Smart Enemies, but reverses the density-response rule so coordinated pressure makes enemies easier to corner. Tests cooperative pursuit and synchronized attack patterns. |
| `ammo_carrier` | 2 | Survival time | A fixed-role asymmetric defense task with one immobilized defender and one or more runners who must maintain the defender's ammunition supply under enemy pressure. Tests role-specialized logistics, resupply timing, and defense-support coordination. |
| `stealth_labyrinth` | 2 | Enemies destroyed / total enemies | An asymmetric illumination-and-combat task where a non-attacking scout reveals rooms for a visually dependent gunner, and combat rooms activate sequentially. Tests tightly coupled movement, dependent combat, and extreme role asymmetry. |
| `lava_maze` | 2 | Maze levels cleared | One agent navigates a dynamically carved safe path over lava while the other observes from above and sends color-coded guidance signals. Tests emergent communication, spatial reasoning, and asymmetric information use. |

## Additional assets

- `platform_chain_easy`: easier or diagnostic variant of Platform Chain
- `lava_maze_simple`: easier or diagnostic variant of Lava Maze

These are useful for ablations, debugging, or curriculum-style warm-up settings, but they are not the main benchmark scenarios described in the paper's central results table.

## Relationship to DoomGen

`comrad/scenarios/` is the runtime asset store. Generation workflows assume a co-located `DoomGen/` checkout, which is responsible for producing or mutating the underlying WAD geometry. When a generator changes:

1. Regenerate the WAD in `DoomGen/`.
2. Copy the resulting `.wad` into this directory or the relevant batch subdirectory.
3. Keep the paired `.cfg` and any `batch_registry.json` metadata aligned with the new artifact set.

Training and evaluation always consume the files from `comrad/scenarios/`, not the generator sources directly.

## Batch directories

The batch directories are used when COMRAD trains over pools of pre-generated WAD instances instead of a single static map.

- `batch_armory/`: Armory Siege layout variants plus pool metadata
- `batch_lava_maze/`: Lava Maze task pool
- `batch_platform_chain_curriculum/`: the reference Platform Chain curriculum pool used by `comrad.train_all.py`

These directories are consumed through `--wad_batch=...` and loaded via `comrad.envs.wad_catalog`.
