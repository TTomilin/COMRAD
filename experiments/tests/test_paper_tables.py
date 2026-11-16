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


REPO_ROOT = TEX_PATH.parents[2]
PAPER_SOURCES = [TEX_PATH, TEX_PATH.with_name("iclr2027_conference_reword.tex")]
REBUTTAL_PATH = TEX_PATH.with_name("rebuttal.md")
KD6J = REPO_ROOT / "results" / "rebuttal_kd6j"
# (paper label, generated LaTeX rows, number of key columns, first header cells of the rebuttal.md copy)
GENERATED_TABLES = [
    ("tab:baseline_results", KD6J / "kd6j_table3_normalized.tex", 1, "| Scenario|Random|IPPO|"),
    ("tab:po_visibility", KD6J / "kd6j_visibility.tex", 1, "| Scenario|Teammate in FOV cone (%)|"),
    ("tab:frozen_partner", REPO_ROOT / "results" / "rebuttal_c2k6" / "c2k6_table.tex", 2, "| Scenario|Method|Both learned"),
    ("tab:po_full_global", KD6J / "kd6j_full_obs_global_v2.tex", 1, "| Scenario|Partial (%)|Full (%)|"),
    ("tab:po_vision", KD6J / "kd6j_vision_controls.tex", 2, "| Scenario|Alg.|Seeds|Normal RGB (%)|"),
    ("tab:po_state", KD6J / "kd6j_state_entity.tex", 2, "| Scenario|Alg.|State (%)|"),
]
NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?")
MISSING = {"--", "—", ""}


def _clean(cell: str) -> str:
    cell = re.sub(r"\\text(?:it|bf)\{([^}]*)\}", r"\1", cell)
    return " ".join(cell.replace("*", "").replace("$", "").split())


def _keyed_rows(rows: list[list[str]], n_key: int) -> dict[tuple[str, ...], list[float | None]]:
    """Row key (scenario, method) -> every number of the row in column order; a missing cell counts as None."""
    out = {}
    for cells in rows:
        key = tuple(_clean(c) for c in cells[:n_key])
        assert key not in out, f"duplicate row {key}"
        values: list[float | None] = []
        for cell in map(_clean, cells[n_key:]):
            values += [None] if cell in MISSING else [float(v) for v in NUMBER.findall(cell)]
        out[key] = values
    return out


def _tex_rows(text: str) -> list[list[str]]:
    return [row.split("&") for row in (" ".join(r.split()) for r in text.split(r"\\")) if "&" in row]


def _tex_table(path, label: str) -> list[list[str]]:
    text = path.read_text(encoding="utf-8")
    start = text.index(rf"\label{{{label}}}")
    return _tex_rows(text[text.index(r"\midrule", start) + len(r"\midrule") : text.index(r"\bottomrule", start)])


def _md_tables(header: str) -> list[list[list[str]]]:
    tables, rows = [], None
    for line in REBUTTAL_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith(header):
            rows = []
            tables.append(rows)
        elif rows is not None and line.startswith("|"):
            if not line.startswith("|---"):
                rows.append(line.strip().strip("|").split("|"))
        else:
            rows = None
    return tables


def _expected(generated, n_key: int):
    return _keyed_rows(_tex_rows(generated.read_text(encoding="utf-8")), n_key)


def _assert_rows_match(found, expected, where: str):
    mismatched = {k: (found.get(k), v) for k, v in expected.items() if found.get(k) != pytest.approx(v)}
    assert not mismatched and set(found) == set(expected), (
        f"{where} differs from the generated rows (paper, generated): {mismatched}; "
        f"rows only in the paper: {sorted(set(found) - set(expected))}"
    )


@needs_paper
@pytest.mark.parametrize("source", PAPER_SOURCES, ids=lambda p: p.name)
@pytest.mark.parametrize("label,generated,n_key,_md_header", GENERATED_TABLES, ids=[t[0] for t in GENERATED_TABLES])
def test_paper_table_is_the_generated_table(source, label, generated, n_key, _md_header):
    _assert_rows_match(_keyed_rows(_tex_table(source, label), n_key), _expected(generated, n_key), f"{source.name} {label}")


@needs_paper
@pytest.mark.parametrize("label,generated,n_key,md_header", GENERATED_TABLES, ids=[t[0] for t in GENERATED_TABLES])
def test_rebuttal_table_is_the_generated_table(label, generated, n_key, md_header):
    tables = _md_tables(md_header)
    assert tables, f"no table with header {md_header!r} in {REBUTTAL_PATH.name}"
    expected = _expected(generated, n_key)
    _assert_rows_match(_keyed_rows(tables[0], n_key), expected, f"rebuttal.md copy of {label}")
    if label == "tab:baseline_results":
        # Later copies report held-out / training pairs; only their Random column is a generated number.
        for table in tables[1:]:
            random = {k: v[:1] for k, v in _keyed_rows(table, n_key).items()}
            _assert_rows_match(random, {k: v[:1] for k, v in expected.items()}, f"rebuttal.md Random column of {label}")


@needs_paper
def test_rebuttal_copy_of_the_baseline_table_matches_the_paper():
    assert kd6j_tables.TABLE3_ALGOS == ALGORITHM_COLUMNS
    mappo = ALGORITHM_COLUMNS.index("MAPPO")
    for row in parse_baseline_results():
        scenario = LABEL_TO_SCENARIO[row.scenario]
        assert kd6j_tables.TABLE3[scenario] == pytest.approx(row.means), scenario
        if scenario in kd6j_tables.TABLE3_MAPPO_CI:
            assert kd6j_tables.TABLE3_MAPPO_CI[scenario] == pytest.approx(row.ci95[mappo]), scenario
