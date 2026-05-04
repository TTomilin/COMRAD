import os
import sys
import json

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from doomgen.batch import BatchGenerator
from examples.benchmark.lava_maze import LavaMazeScenario


def main():
    param_grid = {
        "physical_size":     [8],
        "cell_size":         [128],
        "initial_maze_size": [6, 8, 10],
        "max_maze_size":     [20, 25],
        "lava_damage":       [35],
        "seed":              list(range(4)),
    }

    generator = BatchGenerator(LavaMazeScenario, param_grid=param_grid)
    print(f"Total scenarios: {generator.count_combinations()}")

    output_dir = "examples/benchmark/output/batch_lava_maze"
    os.makedirs(output_dir, exist_ok=True)

    batch_metadata = []

    for i, scenario in enumerate(generator.generate_scenarios(base_name="lava_maze")):
        filename = f"{scenario.name}.wad"
        path = os.path.join(output_dir, filename)

        print(f"[{i+1}/{generator.count_combinations()}] {filename}")
        try:
            scenario.generate(path)

            meta = {"id": scenario.name, "filename": filename, "config": scenario.config}
            batch_metadata.append(meta)

        except Exception as e:
            print(f"Failed to generate {filename}: {e}")

    registry_path = os.path.join(output_dir, "batch_registry.json")
    with open(registry_path, "w") as f:
        json.dump(batch_metadata, f, indent=4)

    print(f"Done. Registry: {registry_path}")


if __name__ == "__main__":
    main()
