from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOOMGEN_ROOT = (REPO_ROOT / "DoomGen").resolve()
DOOMGEN_SRC = DOOMGEN_ROOT / "src"
DOOMGEN_BENCHMARK = DOOMGEN_ROOT / "examples" / "benchmark"

for path in (DOOMGEN_SRC, DOOMGEN_BENCHMARK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from armory_siege import ArmorySiegeScenario

DIVERSE_CONFIGS = [
    {"seed": 11, "distance": 600, "corridor_width": 4, "door_timer": 600, "core_health": 1500, "enemy_difficulty": 0.5},
    {"seed": 29, "distance": 800, "corridor_width": 2, "door_timer": 300, "core_health": 1000, "enemy_difficulty": 0.75},
    {"seed": 61, "distance": 1100, "corridor_width": 4, "door_timer": 200, "core_health": 700, "enemy_difficulty": 1.0},
    {"seed": 97, "distance": 1100, "corridor_width": 2, "door_timer": 120, "core_health": 500, "enemy_difficulty": 1.25},
    {"seed": 133, "distance": 900, "corridor_width": 5, "door_timer": 120, "core_health": 800, "enemy_difficulty": 1.5},
    {"seed": 201, "distance": 700, "corridor_width": 1, "door_timer": 450, "core_health": 2000, "enemy_difficulty": 0.3},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate four diverse Armory Siege DoomGen variants as WAD files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "variants",
        help="Directory where the WAD variants will be written.",
    )
    parser.add_argument(
        "--write-registry",
        action="store_true",
        help="Also write a JSON registry with the generated filenames and configs.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_output_dir(output_dir)

    registry: list[dict[str, object]] = []

    for index, config in enumerate(DIVERSE_CONFIGS, start=1):
        filename = f"armory_siege_variant_{index}.wad"
        wad_path = output_dir / filename

        # DoomGen/omg can fail on overwrite after already writing the new file.
        # Remove the previous artifact first so repeated runs stay reliable.
        if wad_path.exists():
            wad_path.unlink()

        print(f"Generating {filename} with config {config}...", flush=True)
        scenario = ArmorySiegeScenario(config=config, name=f"armory_siege_variant_{index}")
        scenario.generate(str(wad_path))
        registry.append({"filename": filename, "config": config})

    if args.write_registry:
        registry_path = output_dir / "variants_registry.json"
        registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {registry_path}")

    print(f"Wrote {len(DIVERSE_CONFIGS)} variants to {output_dir}")


if __name__ == "__main__":
    main()
