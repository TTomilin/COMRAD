"""
Format the Reviewer C2K6 frozen-partner table from `experiments/iclr2027_rebuttal/eval_frozen_partner.py` outputs.

For every scenario the single checkpoint of the Table 3 winning method with the highest held-out `both_learned` mean
is selected (ties -> lower seed). This is a selected-checkpoint diagnostic: selection and reporting use the same
held-out layouts, so every candidate is published in the summary JSON. Partner no-op / random scores are the symmetric
mean over both replaced seats ((seat0 + seat1) / 2, equal episode counts); per-seat values are kept in the JSON. All
percentages follow the ceiling normalization of normalized Table 3: s~ = 100 * clip(s / s_max, 0, 1).

Per intervention k in {noop, random}: retention_k = (pct_k - random_floor_pct) / (both_pct - random_floor_pct), the
share of the above-random-floor score that survives the partner replacement (a heuristic, not an equivalence test).
  - near_floor (inconclusive): both_pct - random_floor_pct < FLOOR_MARGIN_PP (nothing above the floor to lose)
  - drops_to_random:     both retentions <= DROP_TO_RANDOM_RETENTION (<= 15% of the excess score retained)
  - partner_insensitive: both retentions >= PARTNER_INSENSITIVE_RETENTION
  - partner_dependent:   otherwise (at least one replacement removes a substantial part of the excess score)
Uncertainty: Student-t 95% CIs over the per-WAD means (5 layouts, df=4) and paired-by-WAD drops both - partner, all
conditional on the selected checkpoint (not over training seeds).

The script refuses to build the table (unless --allow-missing) when a candidate seed, a random floor, the expected
checkpoint budget, or the expected per-WAD episode counts are missing.

Outputs (in --out-dir): c2k6_frozen_partner_summary.json, c2k6_table.md, c2k6_table.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from comrad.utils.confidence import ci95_half_width
from experiments.iclr2027_rebuttal.generate_kd6j_tables import CEILING, NAMES, SCENARIOS, load_random, norm

# Table 3 winning method per scenario (MAPPO/QMIX ties on Foraging Commons and Armory Siege per rebuttal W1).
WINNERS = {
    "stag_hunt_arena": "MAPPO",
    "rhythm_sync_dense": "MAPPO",
    "foraging_commons": "MAPPO",
    "coop_puzzle": "MAPPO",
    "platform_chain": "MAPPO",
    "armory_siege": "QMIX",
    "coop_health_gathering": "IPPO",
    "lavapit": "MAPPO",
    "smart_enemies": "MAPPO",
    "dumb_enemies": "IPPO",
    "ammo_carrier": "IPPO",
    "stealth_labyrinth": "QMIX",
    "lava_maze": "MAPPO",
}
# Candidate seeds and checkpoint budget per scenario (reused kd6J 50M checkpoints / new 100M frozen_partner_winners).
EXPECTED_CANDIDATES = {
    "stag_hunt_arena": ((42,), 50_000_000),
    "rhythm_sync_dense": ((68,), 50_000_000),
    "foraging_commons": ((42,), 50_000_000),
    "coop_puzzle": ((68,), 50_000_000),
    "platform_chain": ((42,), 50_000_000),
    "armory_siege": ((68,), 50_000_000),
    "coop_health_gathering": ((42, 68), 100_000_000),
    "lavapit": ((42,), 50_000_000),
    "smart_enemies": ((42,), 50_000_000),
    "dumb_enemies": ((42, 68), 100_000_000),
    "ammo_carrier": ((42, 68), 100_000_000),
    "stealth_labyrinth": ((42, 68, 81), 100_000_000),
    "lava_maze": ((42,), 50_000_000),
}
STEP_TOLERANCE = 0.02
# Must equal eval_frozen_partner.PROTOCOL_VERSION (kept separate so validation does not import torch/ViZDoom).
PROTOCOL_VERSION = "c2k6-frozen-partner-v2"
EXPECTED_WADS = (1009, 2027, 3037, 4051, 5099)
BOTH_EPISODES_PER_WAD = 20
SEAT_EPISODES_PER_WAD = 10
SEAT_PASSES = {"noop": ("noop_seat0", "noop_seat1"), "random": ("random_seat0", "random_seat1")}
ALL_PASSES = ("both_learned", *SEAT_PASSES["noop"], *SEAT_PASSES["random"])
FLOOR_MARGIN_PP = 2.0
DROP_TO_RANDOM_RETENTION = 0.15
PARTNER_INSENSITIVE_RETENTION = 0.75
CATEGORY_LABELS = {
    "near_floor": "near the random floor already (inconclusive)",
    "drops_to_random": "drops to the random level",
    "partner_dependent": "partner-dependent",
    "partner_insensitive": "partner-insensitive",
}


def pass_mean(run: dict[str, Any], name: str) -> float:
    recs = run["passes"][name]["records"]
    if not recs:
        raise ValueError(f"{run.get('scenario')} s{run.get('train_seed')}: pass {name} has no episodes")
    return float(np.mean([float(r["true_objective"]) for r in recs]))


def wad_means(records: list[dict[str, Any]]) -> dict[int, float]:
    by_wad: dict[int, list[float]] = {}
    for r in records:
        by_wad.setdefault(int(r["wad_seed"]), []).append(float(r["true_objective"]))
    return {w: float(np.mean(v)) for w, v in sorted(by_wad.items())}


def t_ci95(values: list[float]) -> float:
    """Student-t 95% CI half-width over layout-level values (df = n_layouts - 1)."""
    return ci95_half_width(values)


def layout_ci95(records: list[dict[str, Any]]) -> float:
    return t_ci95(list(wad_means(records).values()))


def seat_records(run: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """Both replaced seats pooled; equal episodes per seat and WAD keep this the per-WAD seat mean."""
    return [r for name in SEAT_PASSES[kind] for r in run["passes"][name]["records"]]


def paired_drop(both: list[dict[str, Any]], partner: list[dict[str, Any]]) -> tuple[float, float]:
    """Mean and t-CI of the per-WAD drop both_learned - partner (raw score units)."""
    b, p = wad_means(both), wad_means(partner)
    diffs = [b[w] - p[w] for w in b]
    return float(np.mean(diffs)), t_ci95(diffs)


def validate_run(
    run: dict[str, Any],
    expected_wads: tuple[int, ...] = EXPECTED_WADS,
    both_eps: int = BOTH_EPISODES_PER_WAD,
    seat_eps: int = SEAT_EPISODES_PER_WAD,
    expect_run_dir: str | None = None,
    expect_steps: int | None = None,
) -> list[str]:
    """Protocol problems of one evaluation JSON (empty list = valid)."""
    problems = []
    scen, seed = run.get("scenario"), run.get("train_seed")
    if run.get("protocol_version") != PROTOCOL_VERSION:
        problems.append(f"protocol_version={run.get('protocol_version')!r} != {PROTOCOL_VERSION!r}")
    if [int(w) for w in run.get("heldout_seeds") or []] != list(expected_wads):
        problems.append(f"heldout_seeds={run.get('heldout_seeds')} != {list(expected_wads)}")
    if sorted((run.get("wad_sha256") or {}).keys()) != sorted(str(w) for w in expected_wads):
        problems.append("wad_sha256 missing for some held-out WADs")
    for key in ("checkpoint_sha256", "config_sha256"):
        if not run.get(key):
            problems.append(f"{key} missing")
    if expect_run_dir is not None and str(Path(run.get("run_dir", ""))) != str(Path(expect_run_dir).resolve()):
        problems.append(f"run_dir={run.get('run_dir')} != {expect_run_dir}")
    expected = EXPECTED_CANDIDATES.get(scen)
    target = expect_steps if expect_steps is not None else (expected[1] if expected is not None else None)
    if target is not None:
        steps = int(run.get("checkpoint_steps") or 0)
        if abs(steps - target) > STEP_TOLERANCE * target:
            problems.append(f"checkpoint_steps={steps} not within {STEP_TOLERANCE:.0%} of {target}")
    for name in ALL_PASSES:
        recs = ((run.get("passes") or {}).get(name) or {}).get("records") or []
        per_wad: dict[int, int] = {}
        for r in recs:
            per_wad[int(r["wad_seed"])] = per_wad.get(int(r["wad_seed"]), 0) + 1
        want = both_eps if name == "both_learned" else seat_eps
        if per_wad != {w: want for w in expected_wads}:
            problems.append(f"{name}: episodes per WAD {per_wad} != {want} on {list(expected_wads)}")
    return [f"{scen} s{seed}: {p}" for p in problems]


def missing_candidates(runs: list[dict[str, Any]]) -> list[str]:
    have = {(r["scenario"], r["algorithm"], int(r["train_seed"])) for r in runs}
    return [
        f"{scen} {WINNERS[scen]} s{seed}"
        for scen, (seeds, _) in EXPECTED_CANDIDATES.items()
        for seed in seeds
        if (scen, WINNERS[scen], seed) not in have
    ]


def unexpected_candidates(runs: list[dict[str, Any]]) -> list[str]:
    """Winner-method runs outside the declared candidate set, or duplicated (scenario, algorithm, seed) keys."""
    problems, seen = [], set()
    for r in runs:
        key = (r["scenario"], r["algorithm"], int(r["train_seed"]))
        if key in seen:
            problems.append(f"duplicate candidate: {key}")
        seen.add(key)
        if key[2] not in EXPECTED_CANDIDATES.get(key[0], ((), 0))[0]:
            problems.append(f"unexpected candidate: {key}")
    return problems


def select_top_runs(runs: list[dict[str, Any]], winners: dict[str, str] = WINNERS) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for run in runs:
        scen, algo = run["scenario"], run["algorithm"]
        if winners.get(scen) != algo:
            continue
        key = (pass_mean(run, "both_learned"), -int(run["train_seed"]))
        best = chosen.get(scen)
        if best is None or key > (pass_mean(best, "both_learned"), -int(best["train_seed"])):
            chosen[scen] = run
    return chosen


def classify(
    both_pct: float, noop_pct: float, random_partner_pct: float, floor_pct: float
) -> tuple[str, dict[str, float] | None]:
    headroom = both_pct - floor_pct
    if headroom < FLOOR_MARGIN_PP:
        return "near_floor", None
    retention = {
        "noop": (noop_pct - floor_pct) / headroom,
        "random": (random_partner_pct - floor_pct) / headroom,
    }
    if max(retention.values()) <= DROP_TO_RANDOM_RETENTION:
        return "drops_to_random", retention
    if min(retention.values()) >= PARTNER_INSENSITIVE_RETENTION:
        return "partner_insensitive", retention
    return "partner_dependent", retention


def scenario_row(scen: str, run: dict[str, Any], floor_raw: float, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    both_raw = pass_mean(run, "both_learned")
    seat = {name: pass_mean(run, name) for kind in SEAT_PASSES for name in SEAT_PASSES[kind]}
    noop_raw = 0.5 * (seat["noop_seat0"] + seat["noop_seat1"])
    rand_raw = 0.5 * (seat["random_seat0"] + seat["random_seat1"])
    floor_pct = norm(scen, floor_raw)
    both_recs = run["passes"]["both_learned"]["records"]
    row: dict[str, Any] = {
        "scenario": scen,
        "name": NAMES[scen],
        "algorithm": run["algorithm"],
        "train_seed": int(run["train_seed"]),
        "checkpoint": run.get("checkpoint"),
        "checkpoint_steps": run.get("checkpoint_steps"),
        "checkpoint_sha256": run.get("checkpoint_sha256"),
        "wad_sha256": run.get("wad_sha256"),
        "hidden_resets": run.get("hidden_resets", []),
        "crashed_episodes": run.get("crashed_episodes", []),
        "ceiling": CEILING[scen],
        "episodes": {name: len(run["passes"][name]["records"]) for name in ALL_PASSES},
        "raw": {"both_learned": both_raw, "partner_noop": noop_raw, "partner_random": rand_raw, **seat},
        "raw_ci95": {
            "both_learned": layout_ci95(both_recs),
            "partner_noop": layout_ci95(seat_records(run, "noop")),
            "partner_random": layout_ci95(seat_records(run, "random")),
        },
        "paired_drop_raw": {
            kind: dict(zip(("mean", "ci95"), paired_drop(both_recs, seat_records(run, kind)))) for kind in SEAT_PASSES
        },
        "layout_means_raw": {name: wad_means(run["passes"][name]["records"]) for name in ALL_PASSES},
        "pct": {
            "both_learned": norm(scen, both_raw),
            "partner_noop": norm(scen, noop_raw),
            "partner_random": norm(scen, rand_raw),
            **{name: norm(scen, v) for name, v in seat.items()},
        },
        "random_policy_raw": floor_raw,
        "random_policy_pct": floor_pct,
        "candidates": [
            {"train_seed": int(c["train_seed"]), "both_learned_raw": pass_mean(c, "both_learned"),
             "both_learned_pct": norm(scen, pass_mean(c, "both_learned")),
             "checkpoint_steps": c.get("checkpoint_steps")}
            for c in sorted(candidates, key=lambda c: int(c["train_seed"]))
        ],
    }
    category, retention = classify(
        row["pct"]["both_learned"], row["pct"]["partner_noop"], row["pct"]["partner_random"], floor_pct
    )
    row["category"] = category
    row["retention"] = retention
    return row


def render(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    md = [
        "| Scenario | Best method | Both learned (%) | Partner no-op (%) | Partner random (%) |",
        "|---|---|---|---|---|",
    ]
    tex = []
    for row in rows:
        p = row["pct"]
        cells = [f"{p['both_learned']:.1f}", f"{p['partner_noop']:.1f}", f"{p['partner_random']:.1f}"]
        md.append(f"| {row['name']} | {row['algorithm']} | " + " | ".join(cells) + " |")
        tex.append(f"    {row['name']} & {row['algorithm']} & " + " & ".join(cells) + " \\\\")
    return md, tex


def render_diagnostics(rows: list[dict[str, Any]]) -> list[str]:
    md = [
        "| Scenario | Seed (of candidates) | Steps | Random policy (%) | Both learned (%, t-CI) | "
        "No-op seat 0 / 1 (%) | Random seat 0 / 1 (%) | Drop no-op / random (pp, paired t-CI) | "
        "Retention no-op / random | Category | Discarded crash eps |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        p, c = row["pct"], CEILING[row["scenario"]]
        ret = "—" if row["retention"] is None else f"{row['retention']['noop']:.2f} / {row['retention']['random']:.2f}"
        # Means condition on crash-free episodes: ViZDoom crashes discard the interrupted episode (see crashed_episodes).
        crash_counts: dict[str, int] = {}
        for crash in row.get("crashed_episodes", []):
            crash_counts[crash["pass"]] = crash_counts.get(crash["pass"], 0) + 1
        crashes = ", ".join(f"{k} {v}" for k, v in sorted(crash_counts.items())) or "0"
        drops = " / ".join(
            f"{100 * d['mean'] / c:+.1f} ± {100 * d['ci95'] / c:.1f}" for d in row["paired_drop_raw"].values()
        )
        md.append(
            f"| {row['name']} | {row['train_seed']} (of {len(row['candidates'])}) | {row['checkpoint_steps']} | "
            f"{row['random_policy_pct']:.1f} | {p['both_learned']:.1f} ± {100 * row['raw_ci95']['both_learned'] / c:.1f} | "
            f"{p['noop_seat0']:.1f} / {p['noop_seat1']:.1f} | {p['random_seat0']:.1f} / {p['random_seat1']:.1f} | "
            f"{drops} | {ret} | {CATEGORY_LABELS[row['category']]} | {crashes} |"
        )
    return md


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(runs_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "passes" in payload and "scenario" in payload:
            payload["_path"] = str(path)
            runs.append(payload)
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="results/rebuttal_c2k6/heldout_runs")
    p.add_argument("--random-summary", default="results/rebuttal_kd6j/random_full/random_summary.json")
    p.add_argument("--out-dir", default="results/rebuttal_c2k6")
    p.add_argument("--allow-missing", action="store_true", help="build a partial table (smoke tests only)")
    p.add_argument("--validate-json", help="only validate one evaluation JSON (exit 1 on problems)")
    p.add_argument("--expect-run-dir")
    p.add_argument("--expect-steps", type=int)
    p.add_argument("--both-episodes-per-wad", type=int, default=BOTH_EPISODES_PER_WAD)
    p.add_argument("--seat-episodes-per-wad", type=int, default=SEAT_EPISODES_PER_WAD)
    args = p.parse_args()

    if args.validate_json:
        try:
            run = json.loads(Path(args.validate_json).read_text(encoding="utf-8"))
            problems = validate_run(
                run,
                both_eps=args.both_episodes_per_wad,
                seat_eps=args.seat_episodes_per_wad,
                expect_run_dir=args.expect_run_dir,
                expect_steps=args.expect_steps,
            )
        except Exception as exc:  # unreadable / truncated JSON
            problems = [f"{args.validate_json}: {exc!r}"]
        print("\n".join(problems) if problems else "ok")
        raise SystemExit(1 if problems else 0)

    random_path = Path(args.random_summary)
    rand = load_random(random_path)
    floors_missing = [scen for scen, _ in SCENARIOS if "mean_true_objective" not in rand.get(scen, {})]
    runs = [r for r in load_runs(Path(args.runs_dir)) if WINNERS.get(r["scenario"]) == r["algorithm"]]
    problems = [msg for r in runs for msg in validate_run(r)] + [f"no random floor: {s}" for s in floors_missing]
    problems += [f"missing candidate: {c}" for c in missing_candidates(runs)] + unexpected_candidates(runs)
    if problems and not args.allow_missing:
        raise SystemExit("Refusing to build the table:\n  " + "\n  ".join(problems))
    if args.allow_missing:
        runs = [r for r in runs if not validate_run(r) and not unexpected_candidates([r])]
    chosen = select_top_runs(runs)
    missing = [scen for scen, _ in SCENARIOS if scen not in chosen or scen in floors_missing]

    rows = []
    for scen, _ in SCENARIOS:
        if scen in missing:
            continue
        candidates = [r for r in runs if r["scenario"] == scen]
        rows.append(scenario_row(scen, chosen[scen], rand[scen]["mean_true_objective"], candidates))

    md, tex = render(rows)
    groups = {cat: [r["name"] for r in rows if r["category"] == cat] for cat in CATEGORY_LABELS}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "random_summary": str(random_path),
        "protocol": "selected-checkpoint test-time partner replacement without adaptation; checkpoint selected on the "
        "same held-out both_learned episodes; CIs over 5 layouts, conditional on the selected checkpoint",
        "thresholds": {
            "floor_margin_pp": FLOOR_MARGIN_PP,
            "drop_to_random_retention": DROP_TO_RANDOM_RETENTION,
            "partner_insensitive_retention": PARTNER_INSENSITIVE_RETENTION,
        },
        "rows": rows,
        "categories": groups,
        "missing": missing,
        "problems": problems,
    }
    (out_dir / "c2k6_frozen_partner_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text = md + ["", "Diagnostics (% of ceiling; CIs are Student-t over 5 held-out layouts):", ""]
    text += render_diagnostics(rows) + [""]
    text += [f"- {CATEGORY_LABELS[c]}: {', '.join(names) if names else '—'}" for c, names in groups.items()]
    (out_dir / "c2k6_table.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (out_dir / "c2k6_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
