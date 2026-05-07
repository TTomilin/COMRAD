import os
import sys
import json

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.doomgen.batch import BatchGenerator
from examples.benchmark.lava_maze import LavaMazeScenario

def main():
    param_grids = [
        {
            "physical_size":     [4],
            "initial_maze_size": [3, 5],
            "max_maze_size":     [4],
            "lava_depth":        [24, 64],
            "lava_damage":       [10, 20],
        },
        {
            "physical_size":     [6],
            "initial_maze_size": [6, 8],
            "max_maze_size":     [12],
            "lava_depth":        [24],
            "lava_damage":       [20, 35],
        },

        {
            "physical_size":     [6, 8],
            "initial_maze_size": [6, 8, 10],
            "max_maze_size":     [12, 20],
            "lava_depth":        [64],
            "lava_damage":       [35],
        }
    ]

    for grid in param_grids:
        grid["cell_size"] = [128]
        grid["seed"] = list(range(3))

    output_dir = "examples/benchmark/output/batch_lava_maze"
    os.makedirs(output_dir, exist_ok=True)
    batch_metadata = []
    total_generated = 0

    for grid_idx, param_grid in enumerate(param_grids):
        generator = BatchGenerator(LavaMazeScenario, param_grid=param_grid)

        for scenario in generator.generate_scenarios(base_name="lava_maze"):
            unique_name = f"lava_maze_{total_generated}"
            scenario.name = unique_name

            total_generated += 1

            filename = f"{unique_name}.wad"
            path = os.path.join(output_dir, filename)
            try:
                scenario.generate(path)
                meta = {"id": unique_name, "filename": filename, "config": scenario.config}
                batch_metadata.append(meta)
            except Exception as e:
                print(f"Failed to generate {filename}: {e}")

    registry_path = os.path.join(output_dir, "batch_registry.json")
    with open(registry_path, "w") as f:
        json.dump(batch_metadata, f, indent=4)

    print(f"Done. Total scenarios: {total_generated}. Registry: {registry_path}")

if __name__ == "__main__":
    main()
