"""Generate a layout-only training WAD pool for a scenario (the protocol of the held-out batches).

Every WAD uses the scenario's default generator config (plus variant overrides, see heldout_wad_evaluation) and only
changes the layout seed, exactly like `generate_heldout_batch`. Training seeds must be disjoint from HELDOUT_SEEDS.

    python -m experiments.iclr2027_rebuttal.generate_layout_training_batch --scenario coop_puzzle --num-layouts 64
        -> experiments/iclr2027_rebuttal/layouts/batch_coop_puzzle_layouts/{batch_registry.json, *.wad}
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.iclr2027_rebuttal.profiles import layout_batch_dir
from scripts.heldout_wad_evaluation import HELDOUT_SEEDS, REPO_ROOT, WadManifestEntry, generate_layout_batch

TRAIN_SEED_START = 100_000


def training_seeds(num_layouts: int, start: int = TRAIN_SEED_START) -> list[int]:
    seeds = list(range(start, start + num_layouts))
    overlap = set(seeds) & set(HELDOUT_SEEDS)
    if overlap:
        raise ValueError(f"training seeds overlap the held-out seeds: {sorted(overlap)}")
    return seeds


def default_batch_dir(scenario: str) -> Path:
    return REPO_ROOT / layout_batch_dir(scenario)


def generate_training_batch(scenario: str, output_dir: Path, seeds: list[int]) -> list[WadManifestEntry]:
    return generate_layout_batch(scenario, output_dir, seeds, split="train")


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
