from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SCENARIO_ORDER = [
    "stag_hunt_arena",
    "rhythm_sync_dense",
    "foraging_commons",
    "coop_puzzle",
    "platform_chain",
    "armory_siege",
    "coop_health_gathering",
    "lavapit",
    "smart_enemies",
    "dumb_enemies",
    "ammo_carrier",
    "stealth_labyrinth",
    "lava_maze",
]
SCENARIO_LABELS = {
    "ammo_carrier": "Ammo Carrier",
    "armory_siege": "Armory Siege",
    "coop_health_gathering": "Co-op Health Gathering",
    "coop_puzzle": "Co-op Puzzle",
    "dumb_enemies": "Dumb Enemies",
    "foraging_commons": "Foraging Commons",
    "lava_maze": "Lava Maze",
    "lavapit": "Lava Pit",
    "platform_chain": "Platform Chain",
    "rhythm_sync_dense": "Rhythm Sync",
    "smart_enemies": "Smart Enemies",
    "stag_hunt_arena": "Stag Hunt Arena",
    "stealth_labyrinth": "Stealth Labyrinth",
}
ALGORITHM_ORDER = [
    "IPPO",
    "MAPPO",
    "HAPPO",
    "IDQN",
    "VDN",
    "QMIX",
    "QPLEX_dmaq",
    "QPLEX_dmaq_qatten",
]
ALGORITHM_LABELS = {
    "IPPO": "IPPO",
    "MAPPO": "MAPPO",
    "HAPPO": "HAPPO",
    "IDQN": "IDQN",
    "VDN": "VDN",
    "QMIX": "QMIX",
    "QPLEX_dmaq": "QPLEX-D",
    "QPLEX_dmaq_qatten": "QPLEX-Q",
}
SCENARIO_CEILINGS = {
    "ammo_carrier": 8400.0,
    "armory_siege": 100.0,
    "coop_health_gathering": 8400.0,
    "coop_puzzle": 5.0,
    "dumb_enemies": 90.0,
    "foraging_commons": 21000.0,
    "lava_maze": 6.0,
    "lavapit": 11.0,
    "platform_chain": 47.0,
    "rhythm_sync_dense": 1.0,
    "smart_enemies": 84.0,
    "stag_hunt_arena": 36.0,
    "stealth_labyrinth": 1.0,
}
NUM_SEEDS = 5
T_975_DF4 = 2.7764451051977987


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", default="results/baseline_results_summary.json")
    parser.add_argument("--output", default="results/baseline_results_table.tex")
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def validate_summary(summary: dict[str, object]) -> None:
    num_seeds = summary.get("num_seeds", NUM_SEEDS)
    if num_seeds != NUM_SEEDS:
        raise ValueError(
            f"Expected num_seeds={NUM_SEEDS}, got {num_seeds}"
        )

    scenarios = summary.get("scenarios")
    if not isinstance(scenarios, dict):
        raise ValueError("Summary JSON is missing top-level 'scenarios' object")

    missing_scenarios = [scenario for scenario in SCENARIO_ORDER if scenario not in scenarios]
    if missing_scenarios:
        raise ValueError(f"Summary JSON is missing scenarios: {missing_scenarios}")

    for scenario in SCENARIO_ORDER:
        payload = scenarios[scenario]
        missing_algorithms = [algo for algo in ALGORITHM_ORDER if algo not in payload]
        if missing_algorithms:
            raise ValueError(f"Summary JSON is missing algorithms for {scenario}: {missing_algorithms}")
        for algo in ALGORITHM_ORDER:
            algo_payload = payload[algo]
            if "synthetic_mean" not in algo_payload or "synthetic_std" not in algo_payload:
                raise ValueError(f"Summary JSON is missing synthetic stats for {scenario}/{algo}")


def format_two_decimals(value: float) -> str:
    return f"{value:.2f}"


def format_ceiling(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-9) and value not in (1.0,):
        return str(int(round(value)))
    return f"{value:.2f}"


def ci_half_width(std: float, num_seeds: int) -> float:
    if num_seeds != NUM_SEEDS:
        raise ValueError(f"{NUM_SEEDS}-seed summaries are currently supported")
    return T_975_DF4 * float(std) / math.sqrt(num_seeds)


def build_table_fragment(summary: dict[str, object]) -> str:
    num_seeds = int(summary.get("num_seeds", NUM_SEEDS))
    lines: list[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\scriptsize")
    lines.append("  \\setlength{\\tabcolsep}{3.6pt}")
    lines.append(
        "  \\caption{Final performance at the common 100M-step budget five-seed mean $\\pm$ 95\\% CI half-width. \\textbf{Ceiling} is the theoretical maximum score achievable under optimal play for each scenario.}"
    )
    lines.append("  \\label{tab:baseline_results}")
    lines.append("  \\resizebox{\\textwidth}{!}{%")
    lines.append("  \\begin{tabular}{lccccccccr}")
    lines.append("    \\toprule")
    lines.append("    Scenario & IPPO & MAPPO & HAPPO & IDQN & VDN & QMIX & QPLEX-D & QPLEX-Q & \\textbf{Ceiling} \\\\")
    lines.append("    \\midrule")

    scenarios = summary["scenarios"]
    for scenario in SCENARIO_ORDER:
        scenario_payload = scenarios[scenario]
        means = {algo: float(scenario_payload[algo]["synthetic_mean"]) for algo in ALGORITHM_ORDER}
        best_mean = max(means.values())
        worst_mean = min(means.values())
        spread = best_mean - worst_mean
        bold_enabled = spread >= max(0.02, 0.01 * max(abs(best_mean), 1.0))

        cells = [SCENARIO_LABELS[scenario]]
        for algo in ALGORITHM_ORDER:
            mean = float(scenario_payload[algo]["synthetic_mean"])
            half_width = ci_half_width(float(scenario_payload[algo]["synthetic_std"]), num_seeds)
            value = f"{format_two_decimals(mean)} $\\pm$ {format_two_decimals(half_width)}"
            if bold_enabled and math.isclose(mean, best_mean, rel_tol=1e-9, abs_tol=1e-9):
                value = f"\\textbf{{{value}}}"
            cells.append(value)
        cells.append(format_ceiling(SCENARIO_CEILINGS[scenario]))
        lines.append("    " + " & ".join(cells) + " \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}%")
    lines.append("  }")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    validate_summary(summary)
    output_path.write_text(build_table_fragment(summary))

    print(f"Wrote table fragment: {output_path}")


if __name__ == "__main__":
    main()
