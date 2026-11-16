"""
Run selection for `kd6j_heldout_eval_array.sh`: one named evaluation set per results directory the aggregator reads.

A set lists exactly the (scenario, algorithm, cell, seed) the aggregator's `expected_*` coverage requires. A training run
directory counts only if it has a `config.json` and a checkpoint at or above the set's minimum steps (unfinished or
cancelled launches drop out); any other duplicate must be listed in `superseded_runs.json` with a reason. A missing,
duplicated or unexpected run raises `ManifestError`, so no GPU time is spent on an incomplete table.

    python -m experiments.iclr2027_rebuttal.kd6j_manifest --eval-set kd6j_50m > manifest.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from experiments.iclr2027_rebuttal.aggregate_kd6j_results import (
    DEFAULT_CLEAN_SEEDS,
    DEFAULT_SEEDS,
    DEFAULT_V2_SEEDS,
    STATE_ENTITY_CELLS,
    STATE_ENTITY_PIXEL_REUSED_SEEDS,
    STATE_ENTITY_SEEDS,
    V2_MIN_STEPS,
    Cell,
    expected_2x2,
    expected_clean,
    expected_full_obs,
    expected_v2,
    parse_seeds,
)
from experiments.iclr2027_rebuttal.kd6j_protocol import is_current_result_file
from scripts.heldout_wad_evaluation import select_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "rebuttal_kd6j"
SUPERSEDED_RUNS = Path(__file__).resolve().parent / "superseded_runs.json"
RUN_DIR_RE = re.compile(
    r"^\d+_(?P<algorithm>MAPPO|QMIX)_"
    r"(?P<cell>pixel_partial|pixel_full_clean|pixel_full_global_v2|pixel_full_global|pixel_full|state_partial|state_full|state_entity|blind)"
    r"_env_(?P<scenario>.+?)_(?:s\.r\.alp_[\d.]+_)?see_(?P<seed>\d+)$"
)
KD6J_MIN_STEPS = 45_000_000


class ManifestError(RuntimeError):
    """The training runs do not cover exactly the cells and seeds an evaluation set needs."""


def _merge(*tables: dict[Cell, set[int]]) -> dict[Cell, set[int]]:
    out: dict[Cell, set[int]] = {}
    for table in tables:
        for cell, seeds in table.items():
            out.setdefault(cell, set()).update(seeds)
    return {cell: seeds for cell, seeds in out.items() if seeds}


def expected_kd6j_50m() -> dict[Cell, set[int]]:
    seeds = parse_seeds(DEFAULT_SEEDS, "DEFAULT_SEEDS")
    return _merge(expected_2x2(seeds), expected_full_obs(seeds), expected_clean(parse_seeds(DEFAULT_CLEAN_SEEDS, "DEFAULT_CLEAN_SEEDS")))


def expected_full_obs_global_v2() -> dict[Cell, set[int]]:
    return expected_v2(parse_seeds(DEFAULT_V2_SEEDS, "DEFAULT_V2_SEEDS"))


def expected_state_entity_dir() -> dict[Cell, set[int]]:
    """Runs of `heldout_runs_state_entity`: entity state on all seeds, Pixel-Partial only on the retrained seeds."""
    retrained = set(STATE_ENTITY_SEEDS) - set(STATE_ENTITY_PIXEL_REUSED_SEEDS)
    cells = {(s, a, "state_entity"): set(STATE_ENTITY_SEEDS) for s, a in STATE_ENTITY_CELLS}
    cells.update({(s, a, "pixel_partial"): set(retrained) for s, a in STATE_ENTITY_CELLS})
    return cells


@dataclass(frozen=True)
class EvalSet:
    name: str
    profiles: tuple[str, ...]
    out_dir: str
    target_steps: int
    min_steps: int
    run_interventions: bool
    on_scratch: bool
    expected: Callable[[], dict[Cell, set[int]]]


EVAL_SETS = {
    s.name: s
    for s in (
        EvalSet(
            "kd6j_50m",
            ("obs_modality_2x2", "blind_ablation", "visibility_telemetry", "full_obs_clean", "full_obs_global"),
            "heldout_runs", 50_000_000, KD6J_MIN_STEPS, True, False, expected_kd6j_50m,
        ),
        EvalSet(
            "full_obs_global_v2", ("full_obs_global_v2",),
            "heldout_runs", 100_000_000, V2_MIN_STEPS, False, True, expected_full_obs_global_v2,
        ),
        EvalSet(
            "state_entity", ("state_entity",),
            "heldout_runs_state_entity", 50_000_000, KD6J_MIN_STEPS, False, False, expected_state_entity_dir,
        ),
    )
}


@dataclass(frozen=True)
class ManifestRow:
    run_dir: Path
    scenario: str
    algorithm: str
    cell_tag: str
    train_seed: int
    checkpoint_steps: int
    run_interventions: bool
    target_steps: int
    output_json: Path

    def tsv(self) -> str:
        return "\t".join(str(v) for v in (
            self.run_dir, self.scenario, self.algorithm, self.cell_tag, self.train_seed,
            int(self.run_interventions), self.target_steps, self.output_json,
        ))


def default_train_root(eval_set: EvalSet) -> Path:
    if eval_set.on_scratch:
        return Path("/scratch-shared") / os.environ["USER"] / "ViZDoom" / "train_dir"
    return REPO_ROOT / "train_dir"


def load_superseded(path: Path = SUPERSEDED_RUNS) -> set[str]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    for entry in entries:
        if not entry.get("run_dir") or not entry.get("reason"):
            raise ManifestError(f"{path}: every superseded run needs a run_dir and a reason, got {entry}")
    return {entry["run_dir"] for entry in entries}


def _completed_steps(run_dir: Path, eval_set: EvalSet) -> int | None:
    if not (run_dir / "config.json").is_file() or not (run_dir / "checkpoint_p0").is_dir():
        return None
    try:
        _, steps = select_checkpoint(run_dir, eval_set.target_steps)
    except FileNotFoundError:
        return None
    return steps if steps >= eval_set.min_steps else None


def discover(eval_set: EvalSet, train_root: Path) -> dict[tuple[Cell, int], list[tuple[Path, int]]]:
    """Completed run directories of the set's profiles, keyed by (cell, seed)."""
    found: dict[tuple[Cell, int], list[tuple[Path, int]]] = {}
    for profile in eval_set.profiles:
        # The launch timestamp suffix (20YYMMDD_...) keeps full_obs_global from also matching full_obs_global_v2.
        for workdir in sorted(train_root.glob(f"mode3_comrad_{profile}_2*")):
            for run_dir in sorted(p for p in workdir.rglob("*") if p.is_dir() and RUN_DIR_RE.match(p.name)):
                steps = _completed_steps(run_dir, eval_set)
                if steps is None:
                    continue
                m = RUN_DIR_RE.match(run_dir.name)
                key = ((m["scenario"], m["algorithm"], m["cell"]), int(m["seed"]))
                found.setdefault(key, []).append((run_dir, steps))
    return found


def build_manifest(eval_set: EvalSet, train_root: Path, superseded: set[str], out_root: Path = RESULTS_ROOT) -> list[ManifestRow]:
    found = discover(eval_set, train_root)
    expected = eval_set.expected()
    problems = []
    rows = []
    for (cell, seed), candidates in sorted(found.items()):
        live = [(d, s) for d, s in candidates if d.relative_to(train_root).as_posix() not in superseded]
        if seed not in expected.get(cell, set()):
            problems.append(f"unexpected run {'/'.join(cell)} seed {seed}: {', '.join(str(d) for d, _ in live)}")
            continue
        if len(live) > 1:
            problems.append(
                f"duplicate runs for {'/'.join(cell)} seed {seed} (list all but one in {SUPERSEDED_RUNS.name}): "
                + ", ".join(str(d) for d, _ in live)
            )
            continue
        if not live:
            continue
        (run_dir, steps), (scenario, algorithm, cell_tag) = live[0], cell
        rows.append(ManifestRow(
            run_dir, scenario, algorithm, cell_tag, seed, steps,
            eval_set.run_interventions and cell_tag == "pixel_partial", eval_set.target_steps,
            out_root / eval_set.out_dir / f"{scenario}_{algorithm}_{cell_tag}_s{seed}.json",
        ))
    covered = {((r.scenario, r.algorithm, r.cell_tag), r.train_seed) for r in rows}
    for cell, seeds in sorted(expected.items()):
        missing = sorted(s for s in seeds if (cell, s) not in covered)
        if missing:
            problems.append(f"no completed run (>= {eval_set.min_steps} steps) for {'/'.join(cell)} seeds {missing}")
    if problems:
        raise ManifestError(f"{eval_set.name}: {len(problems)} problems:\n  " + "\n  ".join(problems))
    # Pixel-Partial runs with interventions take 3x the episodes: schedule them first to shorten the tail.
    return sorted(rows, key=lambda r: (not r.run_interventions, r.output_json.name))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-set", required=True, choices=sorted(EVAL_SETS))
    p.add_argument("--train-root", type=Path, help="Directory holding mode3_comrad_<profile>_<timestamp>.")
    p.add_argument("--out-root", type=Path, default=RESULTS_ROOT)
    p.add_argument("--pending-only", action="store_true", help="Omit runs whose JSON is already current.")
    args = p.parse_args()
    eval_set = EVAL_SETS[args.eval_set]
    train_root = args.train_root or default_train_root(eval_set)
    try:
        rows = build_manifest(eval_set, train_root, load_superseded(), args.out_root)
    except ManifestError as exc:
        print(exc, file=sys.stderr)
        return 1
    (args.out_root / eval_set.out_dir).mkdir(parents=True, exist_ok=True)
    for row in rows:
        if not (args.pending_only and is_current_result_file(row.output_json)):
            print(row.tsv())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
