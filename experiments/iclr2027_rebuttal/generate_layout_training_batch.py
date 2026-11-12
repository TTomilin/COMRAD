"""Generate a layout-only training WAD pool for a scenario (the protocol of the held-out batches).

Every WAD uses the scenario's default generator config (plus variant overrides, see heldout_wad_evaluation) and only
changes the layout seed, exactly like `generate_heldout_batch`. Training seeds must be disjoint from HELDOUT_SEEDS.

    python experiments/iclr2027_rebuttal/generate_layout_training_batch.py --scenario coop_puzzle --num-layouts 64
        -> comrad/scenarios/batch_coop_puzzle_layouts/{batch_registry.json, coop_puzzle_train_seed_*.wad}
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from heldout_wad_evaluation import (  # noqa: E402
    HELDOUT_SEEDS,
    REPO_ROOT,
    SCENARIO_CLASSES,
    WadManifestEntry,
    resolve_generator,
    sha256_file,
)

TRAIN_SEED_START = 100_000


def training_seeds(num_layouts: int, start: int = TRAIN_SEED_START) -> list[int]:
    seeds = list(range(start, start + num_layouts))
    overlap = set(seeds) & set(HELDOUT_SEEDS)
    if overlap:
        raise ValueError(f"training seeds overlap the held-out seeds: {sorted(overlap)}")
    return seeds


def default_batch_dir(scenario: str) -> Path:
    return REPO_ROOT / "comrad" / "scenarios" / f"batch_{scenario}_layouts"


def generate_training_batch(scenario: str, output_dir: Path, seeds: list[int]) -> list[WadManifestEntry]:
    generator, overrides = resolve_generator(scenario)
    module_name, class_name = SCENARIO_CLASSES[generator]
    scenario_cls = getattr(__import__(module_name, fromlist=[class_name]), class_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for seed in seeds:
        wad_id = f"{scenario}_train_seed_{seed}"
        wad_path = output_dir / f"{wad_id}.wad"
        generated = scenario_cls({**overrides, "seed": int(seed)}, name=wad_id)
        generated.generate(str(wad_path))
        entries.append(
            WadManifestEntry(
                scenario=scenario,
                wad_id=wad_id,
                seed=int(seed),
                filename=wad_path.name,
                sha256=sha256_file(wad_path),
                config=dict(generated.config),
            )
        )
    # comrad.envs.wad_catalog.WadBatch reads `id` and `filename`; the remaining keys are kept as metadata
    payload = [{"id": e.wad_id, **asdict(e)} for e in entries]
    (output_dir / "batch_registry.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True)
    p.add_argument("--num-layouts", type=int, default=64)
    p.add_argument("--seed-start", type=int, default=TRAIN_SEED_START)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    out = Path(args.output_dir) if args.output_dir else default_batch_dir(args.scenario)
    entries = generate_training_batch(args.scenario, out, training_seeds(args.num_layouts, args.seed_start))
    print(f"{len(entries)} layouts -> {out}")


if __name__ == "__main__":
    main()
