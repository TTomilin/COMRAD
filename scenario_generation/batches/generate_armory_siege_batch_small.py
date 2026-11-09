from __future__ import annotations

import json

from doomgen.batch import BatchGenerator

from scenario_generation.paths import RUNTIME_SCENARIOS
from scenario_generation.scenarios.armory_siege import ArmorySiegeScenario


OUTPUT_DIR = RUNTIME_SCENARIOS / "batch_armory"


def main() -> None:
    print("Initializing Batch Generation for Armory Siege...")

    param_grid = {
        "distance": [600, 800, 1100],
        "corridor_width": [4, 2],
        "door_timer": [600, 250, 100],
        "core_health": [1500, 800, 300],
        "enemy_difficulty": [0.25, 0.5, 1],
    }

    generator = BatchGenerator(ArmorySiegeScenario, param_grid=param_grid)
    print(f"Total scenarios to generate: {generator.count_combinations()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    batch_metadata = []

    for i, scenario in enumerate(generator.generate_scenarios(base_name="armory")):
        filename = f"{scenario.name}.wad"
        path = OUTPUT_DIR / filename

        print(f"[{i + 1}/{generator.count_combinations()}] Generating {filename}...")
        try:
            scenario.generate(str(path))
            batch_metadata.append(
                {
                    "id": scenario.name,
                    "filename": filename,
                    "config": scenario.config,
                }
            )
        except Exception as exc:
            print(f"Failed to generate {filename}: {exc}")

    registry_path = OUTPUT_DIR / "batch_registry.json"
    with registry_path.open("w", encoding="utf-8") as handle:
        json.dump(batch_metadata, handle, indent=4)

    print(f"Batch generation complete. Registry saved to {registry_path}")


if __name__ == "__main__":
    main()
