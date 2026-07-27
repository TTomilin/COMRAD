"""
cd DoomGen && uv run python examples/benchmark/batches/generate_armory_siege_w2_eval.py
"""

import json
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from examples.benchmark.armory_siege import ArmorySiegeScenario

OUTPUT_DIR = "../comrad/scenarios/batch_armory_siege_w2_eval"
HELD_OUT_SEED = 999  # Training uses seed=42

# 3x3 grid: Spatial difficulty (rows) x Param difficulty (columns)
# Each cell combines the two variables for that axis into a single config
GRID = {
    # (spatial_level, param_level) -> config overrides
    ("easy", "easy"):   {"distance": 600, "corridor_width": 4, "core_health": 1400, "enemy_difficulty": 0.5},
    ("easy", "medium"): {"distance": 600, "corridor_width": 4, "core_health": 1000, "enemy_difficulty": 1.0},
    ("easy", "hard"):   {"distance": 600, "corridor_width": 4, "core_health": 600,  "enemy_difficulty": 1.5},
    ("medium", "easy"):   {"distance": 700, "corridor_width": 3, "core_health": 1400, "enemy_difficulty": 0.5},
    ("medium", "medium"): {"distance": 700, "corridor_width": 3, "core_health": 1000, "enemy_difficulty": 1.0},
    ("medium", "hard"):   {"distance": 700, "corridor_width": 3, "core_health": 600,  "enemy_difficulty": 1.5},
    ("hard", "easy"):   {"distance": 900, "corridor_width": 2, "core_health": 1400, "enemy_difficulty": 0.5},
    ("hard", "medium"): {"distance": 900, "corridor_width": 2, "core_health": 1000, "enemy_difficulty": 1.0},
    ("hard", "hard"):   {"distance": 900, "corridor_width": 2, "core_health": 600,  "enemy_difficulty": 1.5},
}

# Difficulty labeling for each axis
SPATIAL_LABELS = {"easy": "close+wide", "medium": "mid+standard", "hard": "far+tight"}
PARAM_LABELS = {"easy": "durable+light", "medium": "baseline+standard", "hard": "fragile+heavy"}


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    registry = []
    for idx, ((spatial_level, param_level), overrides) in enumerate(sorted(GRID.items()), start=1):
        scenario_name = f"armory_siege_w2_eval_{idx:02d}"
        config = {
            "seed": HELD_OUT_SEED,
            "door_timer": 300,  # keep same as training
            **overrides,
        }
        scenario = ArmorySiegeScenario(config, name=scenario_name)
        wad_path = os.path.join(OUTPUT_DIR, f"{scenario_name}.wad")

        print(f"[{idx}/9] {scenario_name}: "
              f"spatial={spatial_level} ({SPATIAL_LABELS[spatial_level]}), "
              f"param={param_level} ({PARAM_LABELS[param_level]})")
        scenario.generate(wad_path)

        registry.append({
            "id": scenario_name,
            "filename": f"{scenario_name}.wad",
            "config": scenario.config,
            "axes": {
                "spatial_level": spatial_level,
                "spatial_components": SPATIAL_LABELS[spatial_level],
                "param_level": param_level,
                "param_components": PARAM_LABELS[param_level],
            },
        })

    registry_path = os.path.join(OUTPUT_DIR, "batch_registry.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print()
    print("3x3 Grid Summary")
    print("{:<22} {:<16} {:<16} {:<16}".format("Spatial \\ Param", "Easy", "Medium", "Hard"))
    print("-" * 72)
    for s_level in ["easy", "medium", "hard"]:
        label = SPATIAL_LABELS[s_level]
        cell = f"{s_level} ({label})"
        row = [cell.ljust(21)]
        for p_level in ["easy", "medium", "hard"]:
            key = (s_level, p_level)
            info = registry[list(GRID.keys()).index(key)]
            row.append(info["id"].ljust(16))
        print(" | ".join(row))
    print(f"\nGenerated {len(registry)} WADs in {OUTPUT_DIR} with seed={HELD_OUT_SEED}")
    print(f"Registry saved to {registry_path}")


if __name__ == "__main__":
    main()
