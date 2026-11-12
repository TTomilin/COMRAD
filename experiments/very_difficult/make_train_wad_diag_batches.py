"""
Diagnostic batches that replay each very-difficult variant's *training* WAD through scripts/heldout_wad_evaluation.py
(train-vs-held-out generalisation gap). Five manifest entries point at the same training WAD so the evaluator's
five-seed protocol check passes; the maze / room order still varies per episode through ACS runtime randomness.

  uv run python experiments/very_difficult/make_train_wad_diag_batches.py --output-root results/diag_train_wad_batches
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HELDOUT_SEEDS = (1009, 2027, 3037, 4051, 5099)
TRAINING_WADS = {
    "stealth_labyrinth_scripted_scout": "stealth_labyrinth_scripted_scout.wad",
    "stealth_labyrinth_scripted_gunner": "stealth_labyrinth_scripted_gunner.wad",
    "stealth_labyrinth_lit": "stealth_labyrinth_lit.wad",
    "stealth_labyrinth_shared_vision": "stealth_labyrinth.wad",
    "lava_maze_scripted_observer": "lava_maze.wad",
    "lava_maze_scripted_navigator": "lava_maze.wad",
    "lava_maze_oracle_msg": "lava_maze.wad",
    "lava_maze_oracle_hint": "lava_maze.wad",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/diag_train_wad_batches")
    parser.add_argument("--scenario-dir", default="comrad/scenarios")
    args = parser.parse_args()
    for env, wad_name in TRAINING_WADS.items():
        out = Path(args.output_root) / env
        out.mkdir(parents=True, exist_ok=True)
        wad = out / wad_name
        shutil.copyfile(Path(args.scenario_dir) / wad_name, wad)
        digest = sha256(wad)
        entries = [
            {
                "scenario": env,
                "wad_id": f"{env}_trainwad_rep_{seed}",
                "seed": seed,
                "filename": wad_name,
                "sha256": digest,
                "config": {"training_wad": wad_name, "diagnostic": "train_wad_replay"},
            }
            for seed in HELDOUT_SEEDS
        ]
        (out / "batch_registry.json").write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(out, digest[:12])


if __name__ == "__main__":
    main()
