from __future__ import annotations

import json

from doomgen.batch import BatchGenerator

from scenario_generation.paths import RUNTIME_SCENARIOS
from scenario_generation.scenarios.lava_maze import LavaMazeScenario


OUTPUT_DIR = RUNTIME_SCENARIOS / "batch_lava_maze"


def main() -> None:
    param_grids = [
        {
            "physical_size": [4],
            "initial_maze_size": [3, 5],
            "max_maze_size": [4],
            "lava_depth": [24, 64],
            "lava_damage": [10, 20],
        },
        {
            "physical_size": [6],
            "initial_maze_size": [6, 8],
            "max_maze_size": [12],
            "lava_depth": [24],
            "lava_damage": [20, 35],
        },
        {
            "physical_size": [6, 8],
            "initial_maze_size": [6, 8, 10],
            "max_maze_size": [12, 20],
            "lava_depth": [64],
            "lava_damage": [35],
        },
    ]

    for grid in param_grids:
        grid["cell_size"] = [128]
        grid["seed"] = list(range(3))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    batch_metadata = []
    total_generated = 0

    for param_grid in param_grids:
        generator = BatchGenerator(LavaMazeScenario, param_grid=param_grid)

        for scenario in generator.generate_scenarios(base_name="lava_maze"):
            unique_name = f"lava_maze_{total_generated}"
            scenario.name = unique_name
            total_generated += 1

            filename = f"{unique_name}.wad"
            path = OUTPUT_DIR / filename
            try:
                scenario.generate(str(path))
                batch_metadata.append(
                    {
                        "id": unique_name,
                        "filename": filename,
                        "config": scenario.config,
                    }
                )
            except Exception as exc:
                print(f"Failed to generate {filename}: {exc}")

    registry_path = OUTPUT_DIR / "batch_registry.json"
    with registry_path.open("w", encoding="utf-8") as handle:
        json.dump(batch_metadata, handle, indent=4)

    print(f"Done. Total scenarios: {total_generated}. Registry: {registry_path}")


if __name__ == "__main__":
    main()
