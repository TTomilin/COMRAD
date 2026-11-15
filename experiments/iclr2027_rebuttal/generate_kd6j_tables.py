"""
Format the Reviewer kd6J rebuttal tables from the aggregated held-out summaries.

Inputs (written by `experiments/iclr2027_rebuttal/aggregate_kd6j_results.py` and `experiments/iclr2027_rebuttal/rebuttal_kd6j_eval.py --mode random`):
  results/rebuttal_kd6j/summary_2x2.json, summary_full_obs.json, summary_interventions.json,
  summary_full_obs_clean.json, summary_full_obs_global_v2.json,
  random_full/random_summary.json (20 episodes per held-out WAD x 5 eval seeds)

Main-paper Table 3 contains only the 100M-step / 5-seed baselines plus the Random lower bound; the matched 50M-step
observability ablations are reported in separate tables. Objective-visibility rates are not reported: ViZDoom's labels
buffer only labels Things/actors, never map geometry (switches, plates, bridges, doors), so they are undefined on
several scenarios.

Outputs:
  results/rebuttal_kd6j/kd6j_tables.md             Markdown tables for paper/iclr2027/rebuttal.md
  results/rebuttal_kd6j/kd6j_table3_normalized.tex LaTeX rows for the ceiling-normalized Table 3
  results/rebuttal_kd6j/kd6j_full_obs_global_v2.tex LaTeX rows: Full (global state v2) vs MAPPO Partial (Table 3)

Normalization follows Figure 4 of the paper: s~ = clip(s / s_max, 0, 1), with s_max = 4 mazes for Lava Maze.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from comrad.utils.confidence import t_critical_95

SCENARIOS = (
    ("stag_hunt_arena", "Stag Hunt Arena"),
    ("rhythm_sync_dense", "Rhythm Sync"),
    ("foraging_commons", "Foraging Commons"),
    ("coop_puzzle", "Co-op Puzzle"),
    ("platform_chain", "Platform Chain"),
    ("armory_siege", "Armory Siege"),
    ("coop_health_gathering", "Co-op Health Gathering"),
    ("lavapit", "Lava Pit"),
    ("smart_enemies", "Smart Enemies"),
    ("dumb_enemies", "Dumb Enemies"),
    ("ammo_carrier", "Ammo Carrier"),
    ("stealth_labyrinth", "Stealth Labyrinth"),
    ("lava_maze", "Lava Maze"),
)
NAMES = dict(SCENARIOS)
REPRESENTATIVE = ("rhythm_sync_dense", "armory_siege", "ammo_carrier", "stealth_labyrinth")
CEILING = {
    "stag_hunt_arena": 36, "rhythm_sync_dense": 1.0, "foraging_commons": 21000, "coop_puzzle": 5,
    "platform_chain": 47, "armory_siege": 100, "coop_health_gathering": 2100, "lavapit": 11,
    "smart_enemies": 84, "dumb_enemies": 90, "ammo_carrier": 8400, "stealth_labyrinth": 1.0, "lava_maze": 4,
}
# Paper Table 3 means (100M steps, 5 seeds), copied from paper/iclr2027/iclr2027_conference.tex.
TABLE3_ALGOS = ("IPPO", "MAPPO", "HAPPO", "IDQN", "VDN", "QMIX", "QPLEX-D", "QPLEX-Q")
TABLE3 = {
    "stag_hunt_arena": (17.41, 25.48, 2.63, 14.19, 18.11, 0.26, 1.53, 3.35),
    "rhythm_sync_dense": (0.57, 0.67, 0.21, 0.00, 0.00, 0.00, 0.00, 0.00),
    "foraging_commons": (2341.18, 2497.55, 2212.40, 2178.36, 2339.82, 2531.07, 1918.44, 2358.61),
    "coop_puzzle": (2.98, 3.82, 2.15, 0.02, 0.06, 0.00, 0.00, 0.05),
    "platform_chain": (1.96, 4.85, 4.59, 0.71, 1.09, 0.94, 3.33, 1.01),
    "armory_siege": (15.95, 19.20, -0.85, 10.50, 17.80, 20.45, 1.05, 13.60),
    "coop_health_gathering": (1092.44, 561.20, 201.36, 161.87, 566.41, 168.95, 190.42, 273.05),
    "lavapit": (0.90, 1.00, 1.00, 0.93, 0.78, 0.96, 0.00, 1.00),
    "smart_enemies": (11.39, 14.47, 6.71, 3.38, 4.77, 0.61, 0.00, 0.68),
    "dumb_enemies": (12.21, 2.35, 0.99, 1.11, 2.68, 1.14, 0.00, 1.86),
    "ammo_carrier": (7031.42, 1351.10, 1310.45, 3795.80, 2511.20, 2341.15, 1515.60, 2045.30),
    "stealth_labyrinth": (0.00, 0.00, 0.00, 0.00, 0.00, 0.21, 0.00, 0.00),
    "lava_maze": (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
}
# 95% CI half-widths of the MAPPO column of tab:baseline_results_raw (Student-t over the 5 seeds; per-seed scores of
# Table 3 are not archived locally, so the Full-vs-Partial comparison uses these summary statistics).
TABLE3_MAPPO_CI = {"rhythm_sync_dense": 0.04, "coop_puzzle": 0.11, "armory_siege": 2.45}
TABLE3_N_SEEDS = 5
# Scenarios whose two viewpoints play different roles; the both-agent mean hides the asymmetry, so they are broken down
# per agent (agent index -> role). Lava Maze's observer camera is scripted to track the navigator.
ASYMMETRIC_ROLES = {"lava_maze": ("Navigator", "Observer")}
CELLS_2X2 = (("pixel_partial", "Pixel / Partial"), ("pixel_full", "Pixel / Full"),
             ("state_partial", "State / Partial"), ("state_full", "State / Full"))


def norm(scenario: str, value: float | None) -> float | None:
    if value is None:
        return None
    return 100.0 * min(max(value / CEILING[scenario], 0.0), 1.0)


def _cell(summary: dict, scenario: str, algo: str, cell: str) -> dict | None:
    return ((summary.get(scenario) or {}).get(algo) or {}).get(cell)


def _metric(cell: dict | None, key: str) -> tuple[float, float] | None:
    if not cell or key not in cell:
        return None
    return cell[key]["mean"], cell[key]["ci95"]


def fmt_raw(m: tuple[float, float] | None, digits: int = 2) -> str:
    if m is None:
        return "—"
    mean, ci = m
    return f"{mean:.{digits}f} ± {ci:.{digits}f}" if ci else f"{mean:.{digits}f}"


def fmt_pct(scenario: str, m: tuple[float, float] | None, with_ci: bool = True) -> str:
    if m is None:
        return "—"
    mean, ci = m
    s = f"{norm(scenario, mean):.1f}"
    if with_ci and ci:
        s += f" ± {100.0 * ci / CEILING[scenario]:.1f}"
    return s


def rate(cell: dict | None, key: str) -> str:
    m = _metric(cell, key)
    return "—" if m is None else f"{100.0 * m[0]:.0f}"


def occluded_rate(cell: dict | None, in_key: str, vis_key: str) -> str:
    """Share of in-cone steps without rendered partner pixels, pooled over steps (visible steps are always in-cone).

    The stored `*_obscured_rate` is a per-episode ratio that counts episodes without any in-cone step as 0, which
    understates occlusion when the partner rarely enters the cone (e.g. the Lava Maze navigator).
    """
    in_cone, vis = _metric(cell, in_key), _metric(cell, vis_key)
    if in_cone is None or vis is None or in_cone[0] <= 0:
        return "—"
    return f"{100.0 * max(0.0, 1.0 - vis[0] / in_cone[0]):.0f}"


def digits_for(scenario: str) -> int:
    return 0 if CEILING[scenario] >= 1000 else 2


def load_random(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["scenario"]: row for row in payload["summaries"]}


def visibility_table(inter: dict, rand: dict) -> list[str]:
    lines = [
        "| Scenario | Teammate in FOV cone (%) | Teammate visible (%) | In-cone but occluded (%) | Teammate out of view (%) | Mutual visibility (%) | Teammate visible, random policy (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for scen, name in SCENARIOS:
        c = (inter["normal"].get(scen) or {}).get("MAPPO")
        r = rand.get(scen, {})
        r_vis = f"{100.0 * r['teammate_visible_rate']:.0f}" if "teammate_visible_rate" in r else "—"
        lines.append(
            f"| {name} | {rate(c, 'teammate_in_frustum_rate')} | {rate(c, 'teammate_visible_rate')} | "
            f"{occluded_rate(c, 'teammate_in_frustum_rate', 'teammate_visible_rate')} | "
            f"{rate(c, 'teammate_out_of_view_rate')} | "
            f"{rate(c, 'mutual_visibility_rate')} | {r_vis} |"
        )
    return lines


def per_agent_visibility_table(inter: dict, rand: dict) -> list[str]:
    """Per-viewpoint teammate visibility for asymmetric-role scenarios (rates of that agent's own steps)."""
    lines = [
        "| Scenario | Viewpoint | Teammate in FOV cone (%) | Teammate visible (%) | In-cone but occluded (%) | "
        "Teammate in FOV cone, random policy (%) | Teammate visible, random policy (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for scen, roles in ASYMMETRIC_ROLES.items():
        c = (inter["normal"].get(scen) or {}).get("MAPPO")
        r = rand.get(scen, {})
        for i, role in enumerate(roles):
            keys = [f"agent{i}_teammate_{k}_rate" for k in ("in_frustum", "visible", "obscured")]
            r_cells = [f"{100.0 * r[k]:.1f}" if k in r else "—" for k in keys[:2]]
            lines.append(
                f"| {NAMES[scen]} | {role} (agent {i}) | {rate(c, keys[0])} | {rate(c, keys[1])} | "
                f"{occluded_rate(c, keys[0], keys[1])} | "
                f"{r_cells[0]} | {r_cells[1]} |"
            )
    return lines


FULL_OBS_CLEAN_CELLS = (
    ("pixel_partial", "Partial"),
    ("pixel_full_clean", "Shared vision"),
    ("pixel_full_global", "Full"),
)


def _per_seed_pct(scen: str, cell: dict | None) -> dict[int, float]:
    if not cell or "true_objective" not in cell:
        return {}
    return {int(s): norm(scen, v) for s, v in zip(cell["seeds"], cell["true_objective"]["per_seed"])}


def full_obs_clean_table(clean: dict) -> tuple[list[str], list[str]]:
    """MAPPO Partial vs Shared vision (partner frame + pose) vs Full (global state), no task scalars, % of ceiling.

    Differences are taken against Partial on the seeds both cells share; per-seed scores are listed for each cell.
    """
    labels = [label for _, label in FULL_OBS_CLEAN_CELLS]
    md = [
        "| Scenario | Seeds (P/SV/F) | " + " | ".join(labels) + " | Δ Shared vision | Δ Full | Per-seed (P; SV; F) |",
        "|---|---|" + "---|" * len(labels) + "---|---|---|",
    ]
    tex = []
    tex_pm = "$\\pm$"
    for scen in ("rhythm_sync_dense", "armory_siege", "coop_puzzle"):
        cells = [_cell(clean, scen, "MAPPO", tag) for tag, _ in FULL_OBS_CLEAN_CELLS]
        metrics = [_metric(c, "true_objective") for c in cells]
        per_seed = [_per_seed_pct(scen, c) for c in cells]
        n = "/".join(str(c["n_seeds"]) if c else "0" for c in cells)
        deltas = []
        for other in per_seed[1:]:
            shared = sorted(set(per_seed[0]) & set(other))
            deltas.append(
                sum(other[s] - per_seed[0][s] for s in shared) / len(shared) if shared else None
            )
        delta_str = ["—" if d is None else f"{d:+.1f}" for d in deltas]
        per = "; ".join(", ".join(f"{v:.1f}" for _, v in sorted(p.items())) or "—" for p in per_seed)
        md.append(
            f"| {NAMES[scen]} | {n} | " + " | ".join(fmt_pct(scen, m) for m in metrics)
            + f" | {delta_str[0]} | {delta_str[1]} | {per} |"
        )
        tex_cells = [fmt_pct(scen, m).replace("±", tex_pm).replace("—", "--") for m in metrics]
        tex_deltas = [d.replace("—", "--") for d in delta_str]
        tex.append(f"    {NAMES[scen]} & {n} & " + " & ".join(tex_cells) + " & " + " & ".join(tex_deltas) + " \\\\")
    return md, tex


def _welch_p(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> float | None:
    try:
        from scipy.stats import ttest_ind_from_stats
    except ImportError:
        return None
    if s1 <= 0 and s2 <= 0:
        return None
    return float(ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False).pvalue)


def full_obs_global_v2_table(v2: dict) -> tuple[list[str], list[str]]:
    """Full (own RGB + global state v2, 100M frames, 5 seeds) vs MAPPO Partial from Table 3, % of ceiling.

    Partial is the published Table 3 MAPPO entry (mean +/- 95% CI over the same 5 seeds); the difference is Full minus
    Partial on the means, and p is a two-sided Welch t-test from the summary statistics of both cells.
    """
    md = [
        "| Scenario | Seeds | Partial (Table 3) | Full | Δ Full − Partial | Welch p | Per-seed Full |",
        "|---|---|---|---|---|---|---|",
    ]
    tex = []
    mappo = TABLE3_ALGOS.index("MAPPO")
    for scen in ("rhythm_sync_dense", "armory_siege", "coop_puzzle"):
        cell = _cell(v2, scen, "MAPPO", "pixel_full_global_v2")
        full = _metric(cell, "true_objective")
        partial = (TABLE3[scen][mappo], TABLE3_MAPPO_CI[scen])
        per = _per_seed_pct(scen, cell)
        per_s = ", ".join(f"{v:.1f}" for _, v in sorted(per.items())) or "—"
        n = cell["n_seeds"] if cell else 0
        if full is None:
            delta, p_s = "—", "—"
        else:
            delta = f"{norm(scen, full[0]) - norm(scen, partial[0]):+.1f}"
            sd_full = statistics.stdev(cell["true_objective"]["per_seed"]) if n > 1 else 0.0
            sd_partial = partial[1] * TABLE3_N_SEEDS ** 0.5 / t_critical_95(TABLE3_N_SEEDS)
            p = _welch_p(full[0], sd_full, n, partial[0], sd_partial, TABLE3_N_SEEDS) if n > 1 else None
            p_s = "—" if p is None else (f"{p:.3f}" if p >= 0.001 else "<0.001")
        cells = [fmt_pct(scen, partial), fmt_pct(scen, full)]
        md.append(f"| {NAMES[scen]} | {n} | " + " | ".join(cells) + f" | {delta} | {p_s} | {per_s} |")
        tex_cells = [c.replace("±", "$\\pm$").replace("—", "--") for c in cells]
        tex.append(
            f"    \\textit{{{NAMES[scen]}}} & " + " & ".join(tex_cells)
            + f" & {delta.replace('—', '--')} & {per_s.replace('—', '--')} \\\\"
        )
    return md, tex


def intervention_table(inter: dict, two: dict, rand: dict) -> list[str]:
    lines = [
        "| Scenario | Seeds | Normal RGB | Partner sprite masked | Zero-shot blind | Trained blind (t/H only) | Random |",
        "|---|---|---|---|---|---|---|",
    ]
    for scen, name in SCENARIOS:
        for algo in ("MAPPO", "QMIX"):
            normal = (inter["normal"].get(scen) or {}).get(algo)
            if not normal:
                continue
            occ = _cell(inter["partner_occlusion"], scen, algo, "pixel_partial")
            blind = _cell(inter["blind_eval"], scen, algo, "pixel_partial")
            trained = _cell(two, scen, algo, "blind")
            r = rand.get(scen)
            r_m = (r["mean_true_objective"], r["ci95_half_width"]) if r else None
            label = name if algo == "MAPPO" else f"{name} (QMIX)"
            lines.append(
                f"| {label} | {normal['n_seeds']} | {fmt_pct(scen, _metric(normal, 'true_objective'))} | "
                f"{fmt_pct(scen, _metric(occ, 'true_objective'))} | {fmt_pct(scen, _metric(blind, 'true_objective'))} | "
                f"{fmt_pct(scen, _metric(trained, 'true_objective'))} | {fmt_pct(scen, r_m)} |"
            )
    return lines


def two_by_two_table(two: dict) -> list[str]:
    labels = ["Pixel / Full*" if c == "pixel_full" else label for c, label in CELLS_2X2]
    lines = [
        "| Scenario | Algorithm | " + " | ".join(labels) + " |",
        "|---|---|" + "---|" * len(CELLS_2X2),
    ]
    for scen in REPRESENTATIVE:
        for algo in ("MAPPO", "QMIX"):
            cells = [fmt_raw(_metric(_cell(two, scen, algo, c), "true_objective"), digits_for(scen)) for c, _ in CELLS_2X2]
            lines.append(f"| {NAMES[scen]} | {algo} | " + " | ".join(cells) + " |")
    return lines


def normalized_table3(rand: dict) -> tuple[list[str], list[str]]:
    md = [
        "| Scenario | Random | " + " | ".join(TABLE3_ALGOS) + " |",
        "|---|---|" + "---|" * len(TABLE3_ALGOS),
    ]
    tex = []
    for scen, name in SCENARIOS:
        r = rand.get(scen)
        r_s = f"{norm(scen, r['mean_true_objective']):.1f}" if r else "--"
        base = [norm(scen, v) for v in TABLE3[scen]]
        best = max(base)
        base_md = [f"**{b:.1f}**" if b == best and best > 0 else f"{b:.1f}" for b in base]
        base_tex = [f"\\textbf{{{b:.1f}}}" if b == best and best > 0 else f"{b:.1f}" for b in base]
        md.append(f"| {name} | {r_s} | " + " | ".join(base_md) + " |")
        tex.append(f"    {name} & {r_s} & " + " & ".join(base_tex) + " \\\\")
    return md, tex


STATE_ENTITY_CELLS = (
    ("rhythm_sync_dense", "MAPPO"), ("armory_siege", "MAPPO"), ("armory_siege", "QMIX"), ("ammo_carrier", "MAPPO"),
    ("ammo_carrier", "QMIX"),
)


def state_entity_table(summary: dict) -> tuple[list[str], list[str]]:
    """Entity state (SMAC-style, sight radius) vs Pixel-Partial, % of ceiling, mean ± 95% CI and per-seed scores.

    The difference is State minus Pixel on the seeds both cells share.
    """
    md = [
        "| Scenario | Algorithm | Seeds (S/P) | State | Pixel | Δ State − Pixel | Per-seed State | Per-seed Pixel |",
        "|---|---|---|---|---|---|---|---|",
    ]
    tex = []
    for scen, algo in STATE_ENTITY_CELLS:
        cells = [_cell(summary, scen, algo, tag) for tag in ("state_entity", "pixel_partial")]
        metrics = [_metric(c, "true_objective") for c in cells]
        state, pixel = (_per_seed_pct(scen, c) for c in cells)
        shared = sorted(set(state) & set(pixel))
        delta = "—" if not shared else f"{sum(state[s] - pixel[s] for s in shared) / len(shared):+.1f}"
        n = "/".join(str(c["n_seeds"]) if c else "0" for c in cells)
        per = [", ".join(f"{v:.1f}" for _, v in sorted(p.items())) or "—" for p in (state, pixel)]
        md.append(
            f"| {NAMES[scen]} | {algo} | {n} | " + " | ".join(fmt_pct(scen, m) for m in metrics)
            + f" | {delta} | {per[0]} | {per[1]} |"
        )
        tex_cells = [fmt_pct(scen, m).replace("±", "$\\pm$").replace("—", "--") for m in metrics]
        tex.append(
            f"    \\textit{{{NAMES[scen]}}} & {algo} & {n} & " + " & ".join(tex_cells)
            + f" & {delta.replace('—', '--')} & {per[0].replace('—', '--')} & {per[1].replace('—', '--')} \\\\"
        )
    return md, tex


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results/rebuttal_kd6j")
    args = p.parse_args()
    d = Path(args.results_dir)
    two = json.loads((d / "summary_2x2.json").read_text(encoding="utf-8"))
    full = json.loads((d / "summary_full_obs.json").read_text(encoding="utf-8"))
    inter = json.loads((d / "summary_interventions.json").read_text(encoding="utf-8"))
    clean_path = d / "summary_full_obs_clean.json"
    clean = json.loads(clean_path.read_text(encoding="utf-8")) if clean_path.exists() else {}
    entity_path = d / "summary_state_entity.json"
    entity = json.loads(entity_path.read_text(encoding="utf-8")) if entity_path.exists() else {}
    v2_path = d / "summary_full_obs_global_v2.json"
    v2 = json.loads(v2_path.read_text(encoding="utf-8")) if v2_path.exists() else {}
    rand_path = d / "random_full" / "random_summary.json"
    rand = load_random(rand_path if rand_path.exists() else d / "random_summary.json")
    # Random-policy telemetry re-run with the fixed frustum test (ZDoom pitch sign, eye height) and per-agent rates
    rand_fov_path = d / "random_lava_maze_fov" / "random_summary.json"
    rand_fov = {**rand, **load_random(rand_fov_path)} if rand_fov_path.exists() else rand

    md_t3, tex_t3 = normalized_table3(rand)
    md_clean, tex_clean = full_obs_clean_table(clean)
    md_entity, tex_entity = state_entity_table(entity)
    md_v2, tex_v2 = full_obs_global_v2_table(v2)
    sections = [
        ("Placeholder 1: visibility telemetry (MAPPO Pixel-Partial, held-out maps; mean of both agents' viewpoints)",
         visibility_table(inter, rand_fov)),
        ("Placeholder 1b: per-agent visibility for asymmetric roles", per_agent_visibility_table(inter, rand_fov)),
        ("Q1: Partial vs. Shared vision vs. Full (global state) observability (MAPPO, % of ceiling, mean ± 95% CI)",
         md_clean),
        ("Full (own RGB + global state v2, layout pool, 100M) vs. MAPPO Partial from Table 3 (% of ceiling, mean ± 95% CI)",
         md_v2),
        ("Placeholder 2: partner occlusion and blind policies (% of ceiling)", intervention_table(inter, two, rand)),
        ("Placeholder 3: State vs. Pixel x Partial vs. Full (raw score, mean ± 95% CI; * = Full also adds task scalars)", two_by_two_table(two)),
        ("Placeholder 4: ceiling-normalized Table 3 (% of ceiling)", md_t3),
        ("State (entity encoding) vs. Pixel, Partial observability (% of ceiling, mean ± 95% CI)", md_entity),
    ]
    out = []
    for title, lines in sections:
        out += [f"### {title}", ""] + lines + [""]
    (d / "kd6j_tables.md").write_text("\n".join(out), encoding="utf-8")
    (d / "kd6j_table3_normalized.tex").write_text("\n".join(tex_t3) + "\n", encoding="utf-8")
    (d / "kd6j_full_obs_clean.tex").write_text("\n".join(tex_clean) + "\n", encoding="utf-8")
    (d / "kd6j_state_entity.tex").write_text("\n".join(tex_entity) + "\n", encoding="utf-8")
    (d / "kd6j_full_obs_global_v2.tex").write_text("\n".join(tex_v2) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"Wrote {d / 'kd6j_tables.md'} and {d / 'kd6j_table3_normalized.tex'}")


if __name__ == "__main__":
    main()
