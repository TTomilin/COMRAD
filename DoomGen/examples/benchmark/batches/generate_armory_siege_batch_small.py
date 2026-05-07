import os
import sys
import json
import logging

# Ensure project root and src are in path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from doomgen.batch import BatchGenerator
from examples.benchmark.armory_siege import ArmorySiegeScenario

def main():
    print("Initializing Batch Generation for Armory Siege...")

    param_grid = {
        'distance': [600, 800, 1100],
        'corridor_width': [4, 2],
        'door_timer': [600, 250, 100],
        'core_health': [1500, 800, 300],
        'enemy_difficulty': [0.25, 0.5, 1],
    }

    demo_grid = {
        'distance': [600, 800, 1100],
        'corridor_width': [3],
        'door_timer': [600, 250],
        'core_health': [1500, 500],
        'enemy_difficulty': [0.5, 1.25]
    }

    generator = BatchGenerator(ArmorySiegeScenario, param_grid=param_grid)

    print(f"Total scenarios to generate: {generator.count_combinations()}")

    output_dir = "examples/benchmark/output/batch_armory"
    os.makedirs(output_dir, exist_ok=True)

    batch_metadata = []

    for i, scenario in enumerate(generator.generate_scenarios(base_name="armory")):
        filename = f"{scenario.name}.wad"
        path = os.path.join(output_dir, filename)

        print(f"[{i+1}/{generator.count_combinations()}] Generating {filename}...")
        try:
            scenario.generate(path)

            # Record metadata
            meta = {
                'id': scenario.name,
                'filename': filename,
                'config': scenario.config
            }
            batch_metadata.append(meta)

        except Exception as e:
            print(f"Failed to generate {filename}: {e}")
            # traceback.print_exc()

    #Central JSON
    central_json_path = os.path.join(output_dir, "batch_registry.json")
    with open(central_json_path, 'w') as f:
        json.dump(batch_metadata, f, indent=4)

    print(f"Batch generation complete. Registry saved to {central_json_path}")

if __name__ == "__main__":
    main()
