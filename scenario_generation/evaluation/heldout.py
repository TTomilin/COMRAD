from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from scenario_generation.manifest import WadManifestEntry, sha256_file


HELDOUT_SEEDS = (1009, 2027, 3037, 4051, 5099)
DEFAULT_SCENARIOS = ("ammo_carrier", "armory_siege", "coop_puzzle")
DEFAULT_BATCH_ROOT = Path("results") / "heldout_wad_batches"

SCENARIO_CLASSES = {
    "ammo_carrier": ("scenario_generation.scenarios.ammo_carrier", "AmmoCarrierScenario"),
    "armory_siege": ("scenario_generation.scenarios.armory_siege", "ArmorySiegeScenario"),
    "coop_puzzle": ("scenario_generation.scenarios.coop_puzzle", "CoopPuzzleScenario"),
}


def parse_csv_argument(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def generate_heldout_batch(scenario: str, output_dir: Path, seeds: tuple[int, ...]) -> list[WadManifestEntry]:
    """Generate only layout changes: each WAD uses the scenario's default config plus a new seed."""
    if scenario not in SCENARIO_CLASSES:
        raise ValueError(f"Unsupported scenario {scenario!r}; choices: {sorted(SCENARIO_CLASSES)}")
    module_name, class_name = SCENARIO_CLASSES[scenario]
    module = __import__(module_name, fromlist=[class_name])
    scenario_cls = getattr(module, class_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[WadManifestEntry] = []
    for seed in seeds:
        wad_id = f"{scenario}_heldout_seed_{seed}"
        filename = f"{wad_id}.wad"
        wad_path = output_dir / filename
        generated = scenario_cls({"seed": int(seed)}, name=wad_id)
        generated.generate(str(wad_path))
        entries.append(
            WadManifestEntry(
                scenario=scenario,
                wad_id=wad_id,
                seed=int(seed),
                filename=filename,
                sha256=sha256_file(wad_path),
                config=dict(generated.config),
            )
        )

    manifest_path = output_dir / "batch_registry.json"
    manifest_payload = [asdict(entry) for entry in entries]
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entries


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate held-out WAD batches for COMRAD evaluation.")
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", default=",".join(map(str, HELDOUT_SEEDS)))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    scenarios = parse_csv_argument(args.scenarios)
    seeds = tuple(int(value) for value in parse_csv_argument(args.seeds))
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Held-out protocol requires exactly five distinct layout seeds")

    for scenario in scenarios:
        batch_dir = args.batch_root / scenario
        if batch_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"{batch_dir} already exists; pass --overwrite to regenerate")
            shutil.rmtree(batch_dir)
        generate_heldout_batch(scenario, batch_dir, seeds)
        print(f"Wrote {batch_dir}")


if __name__ == "__main__":
    main()
