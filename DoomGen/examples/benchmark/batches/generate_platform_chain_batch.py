import json
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from doomgen.batch import BatchGenerator
from examples.benchmark.platform_chain import PlatformChainScenario


def main() -> None:
    param_grid = {
        "num_platforms": [120, 144, 168],
        "tier_count": [10, 12],
        "chain_max_len": [180, 220, 260],
        "lava_damage": [8, 12, 16],
        "runtime_reseed_tics": [140, 175],
    }

    generator = BatchGenerator(PlatformChainScenario, param_grid=param_grid)
    total = generator.count_combinations()
    print(f"Generating {total} Platform Chain scenarios...")

    output_dir = "examples/benchmark/output/batch_platform_chain"
    os.makedirs(output_dir, exist_ok=True)

    registry = []
    for idx, scenario in enumerate(generator.generate_scenarios(base_name="platform_chain"), start=1):
        wad_name = f"{scenario.name}.wad"
        wad_path = os.path.join(output_dir, wad_name)
        print(f"[{idx}/{total}] {wad_name}")
        scenario.generate(wad_path)

        meta = {
            "id": scenario.name,
            "filename": wad_name,
            "config": scenario.config,
        }
        registry.append(meta)

    with open(os.path.join(output_dir, "batch_registry.json"), "w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2)

    print("Platform Chain batch generation complete.")


if __name__ == "__main__":
    main()
