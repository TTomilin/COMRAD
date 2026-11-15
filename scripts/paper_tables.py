"""Single parser for the raw baseline table of the paper (`tab:baseline_results_raw`).

The main-text `tab:baseline_results` reports ceiling-normalized percentages; every script that needs the raw
mean +- CI per (scenario, algorithm) reads them from here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEX_PATH = Path(__file__).resolve().parents[1] / "paper" / "iclr2027" / "iclr2027_conference.tex"
RAW_BASELINE_LABEL = r"\label{tab:baseline_results_raw}"
ALGORITHM_COLUMNS = ("IPPO", "MAPPO", "HAPPO", "IDQN", "VDN", "QMIX", "QPLEX-D", "QPLEX-Q")
NO_CEILING_S_MAX = {"Lava Maze": 4.0}

_MEANCI_RE = re.compile(r"\\(b?)meanci\{(-?[\d.]+)\}\{(-?[\d.]+)\}")
_HEADER_RE = re.compile(r"\\textbf\{([^}]*)\}")


@dataclass(frozen=True)
class BaselineRow:
    scenario: str
    means: tuple[float, ...]
    ci95: tuple[float, ...]
    best: tuple[bool, ...]
    ceiling: float

    def normalized(self) -> np.ndarray:
        return np.clip(np.asarray(self.means) / self.ceiling, 0.0, 1.0)


def _table_section(text: str, label: str, start_marker: str, end_marker: str) -> str:
    start = text.index(label)
    begin = text.index(start_marker, start) + len(start_marker)
    return text[begin : text.index(end_marker, begin)]


def parse_baseline_results(tex_path: Path = TEX_PATH) -> list[BaselineRow]:
    text = Path(tex_path).read_text(encoding="utf-8")
    header = _HEADER_RE.findall(_table_section(text, RAW_BASELINE_LABEL, r"\toprule", r"\midrule"))
    expected_header = ["Scenario", *ALGORITHM_COLUMNS, "Ceiling"]
    if header != expected_header:
        raise ValueError(f"{RAW_BASELINE_LABEL} columns {header} != {expected_header}")

    rows: list[BaselineRow] = []
    for raw_row in _table_section(text, RAW_BASELINE_LABEL, r"\midrule", r"\bottomrule").split(r"\\"):
        row = " ".join(raw_row.split())
        if not row:
            continue
        scenario = row.split("&", 1)[0].strip()
        cells = _MEANCI_RE.findall(row)
        if len(cells) != len(ALGORITHM_COLUMNS):
            raise ValueError(f"Expected {len(ALGORITHM_COLUMNS)} entries for {scenario!r}, got {len(cells)}")
        ceiling_cell = row.rsplit("&", 1)[1].strip()
        rows.append(
            BaselineRow(
                scenario=scenario,
                means=tuple(float(mean) for _, mean, _ in cells),
                ci95=tuple(float(ci) for _, _, ci in cells),
                best=tuple(flag == "b" for flag, _, _ in cells),
                ceiling=NO_CEILING_S_MAX[scenario] if ceiling_cell == "--" else float(ceiling_cell),
            )
        )
    return rows
