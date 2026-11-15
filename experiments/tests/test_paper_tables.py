import re

import numpy as np
import pytest

from experiments.iclr2027_rebuttal import generate_kd6j_tables as kd6j_tables
from scripts.paper_tables import ALGORITHM_COLUMNS, TEX_PATH, parse_baseline_results

LABEL_TO_SCENARIO = {label: key for key, label in kd6j_tables.SCENARIOS}

HEADER = " & ".join(rf"\textbf{{{name}}}" for name in ("Scenario", *ALGORITHM_COLUMNS, "Ceiling"))
needs_paper = pytest.mark.skipif(not TEX_PATH.is_file(), reason=f"paper source not checked out at {TEX_PATH}")


def _raw_table(header: str, rows: list[str]) -> str:
    body = "\n".join(rows)
    return (
        "\\begin{table}\\label{tab:baseline_results}\\toprule\\textbf{Scenario}\\midrule ignored\\bottomrule\\end{table}\n"
        f"\\begin{{table}}\\label{{tab:baseline_results_raw}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n"
        "\\bottomrule\\end{table}\n"
    )


def _row(scenario: str, ceiling: str) -> str:
    cells = [r"\bmeanci{2.00}{0.50}"] + [rf"\meanci{{{i}.00}}{{0.10}}" for i in range(7)]
    return f"{scenario} & " + " & ".join(cells) + f" & {ceiling} \\\\"


def test_parses_only_the_raw_table(tmp_path):
    tex = tmp_path / "paper.tex"
    tex.write_text(_raw_table(HEADER, [_row("Stag Hunt Arena", "36"), _row("Lava Maze", "--")]))
    rows = parse_baseline_results(tex)
    assert [r.scenario for r in rows] == ["Stag Hunt Arena", "Lava Maze"]
    assert rows[0].means[0] == 2.0 and rows[0].ci95[0] == 0.5 and rows[0].best[0] and not rows[0].best[1]
    assert rows[0].ceiling == 36.0 and rows[1].ceiling == 4.0
    np.testing.assert_allclose(rows[1].normalized()[:3], [0.5, 0.0, 0.25])


def test_rejects_reordered_columns(tmp_path):
    tex = tmp_path / "paper.tex"
    tex.write_text(_raw_table(HEADER.replace("IPPO", "XXXX").replace("MAPPO", "IPPO"), [_row("Lava Pit", "11")]))
    with pytest.raises(ValueError, match="columns"):
        parse_baseline_results(tex)


def test_rejects_rows_with_missing_cells(tmp_path):
    tex = tmp_path / "paper.tex"
    tex.write_text(_raw_table(HEADER, [_row("Lava Pit", "11").replace(r"\meanci{0.00}{0.10} & ", "", 1)]))
    with pytest.raises(ValueError, match="Expected 8 entries"):
        parse_baseline_results(tex)


@needs_paper
def test_paper_raw_table_covers_every_benchmark_scenario():
    rows = parse_baseline_results()
    assert sorted(r.scenario for r in rows) == sorted(LABEL_TO_SCENARIO)


@needs_paper
def test_paper_normalized_table_is_derived_from_the_raw_table():
    text = TEX_PATH.read_text(encoding="utf-8")
    start = text.index(r"\label{tab:baseline_results}")
    body = text[text.index(r"\midrule", start) + len(r"\midrule") : text.index(r"\bottomrule", start)]
    normalized = {}
    for raw_row in body.split(r"\\"):
        row = " ".join(raw_row.split())
        if row:
            scenario, cells = row.split("&", 1)
            normalized[scenario.strip()] = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", cells)][1:]
    for row in parse_baseline_results():
        np.testing.assert_allclose(normalized[row.scenario], 100 * row.normalized(), atol=0.051, err_msg=row.scenario)


@needs_paper
def test_rebuttal_copy_of_the_baseline_table_matches_the_paper():
    assert kd6j_tables.TABLE3_ALGOS == ALGORITHM_COLUMNS
    mappo = ALGORITHM_COLUMNS.index("MAPPO")
    for row in parse_baseline_results():
        scenario = LABEL_TO_SCENARIO[row.scenario]
        assert kd6j_tables.TABLE3[scenario] == pytest.approx(row.means), scenario
        if scenario in kd6j_tables.TABLE3_MAPPO_CI:
            assert kd6j_tables.TABLE3_MAPPO_CI[scenario] == pytest.approx(row.ci95[mappo]), scenario
