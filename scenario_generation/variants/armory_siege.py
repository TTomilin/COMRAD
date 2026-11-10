from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenario_generation.scenarios.armory_siege import ArmorySiegeScenario


REPO_ROOT = Path(__file__).resolve().parents[2]

VARIANT_SETS = {
    "2x2": [
        {"seed": 11, "distance": 600, "corridor_width": 4, "door_timer": 600, "core_health": 1500, "enemy_difficulty": 0.5},
        {"seed": 42, "distance": 800, "corridor_width": 2, "door_timer": 300, "core_health": 1000, "enemy_difficulty": 0.75},
        {"seed": 68, "distance": 1100, "corridor_width": 4, "door_timer": 200, "core_health": 700, "enemy_difficulty": 1.0},
        {"seed": 151, "distance": 1500, "corridor_width": 2, "door_timer": 120, "core_health": 500, "enemy_difficulty": 1.25},
    ],
    "2x3": [
        {"seed": 11, "distance": 600, "corridor_width": 4, "door_timer": 600, "core_health": 1500, "enemy_difficulty": 0.5},
        {"seed": 29, "distance": 800, "corridor_width": 2, "door_timer": 300, "core_health": 1000, "enemy_difficulty": 0.75},
        {"seed": 61, "distance": 1100, "corridor_width": 4, "door_timer": 200, "core_health": 700, "enemy_difficulty": 1.0},
        {"seed": 97, "distance": 1100, "corridor_width": 2, "door_timer": 120, "core_health": 500, "enemy_difficulty": 1.25},
        {"seed": 133, "distance": 900, "corridor_width": 5, "door_timer": 120, "core_health": 800, "enemy_difficulty": 1.5},
        {"seed": 201, "distance": 700, "corridor_width": 1, "door_timer": 450, "core_health": 2000, "enemy_difficulty": 0.3},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Armory Siege variants for figures and visual comparisons."
    )
    parser.add_argument(
        "--variant-set",
        choices=sorted(VARIANT_SETS),
        default="2x2",
        help="Variant set used for the figure layout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "variants",
        help="Directory where the WAD variants will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = VARIANT_SETS[args.variant_set]
    registry: list[dict[str, object]] = []

    for stale_path in output_dir.glob("armory_siege_variant_*.wad"):
        stale_path.unlink()

    for index, config in enumerate(configs, start=1):
        filename = f"armory_siege_variant_{index}.wad"
        wad_path = output_dir / filename

        print(f"Generating {filename} with config {config}...", flush=True)
        scenario = ArmorySiegeScenario(config=config, name=f"armory_siege_variant_{index}")
        scenario.generate(str(wad_path))
        registry.append({"filename": filename, "config": config})

    registry_path = output_dir / "variants_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {registry_path}")

    print(f"Wrote {len(configs)} variants to {output_dir}")


if __name__ == "__main__":
    main()
