import re
from pathlib import Path

import numpy as np
import pytest

from comrad.utils.confidence import ci95_half_width, t_critical_95

REPO_ROOT = Path(__file__).resolve().parents[2]
HAND_ROLLED_CI = re.compile(r"1\.96\s*\*|\*\s*1\.96|,\s*1\.96\)|:\s*12\.706|=\s*2\.776\b")


@pytest.mark.parametrize("n, expected", [(2, 12.706), (3, 4.303), (5, 2.776), (30, 2.045)])
def test_t_critical_uses_n_minus_one_degrees_of_freedom(n, expected):
    assert t_critical_95(n) == pytest.approx(expected, abs=1e-3)


def test_t_critical_rejects_single_sample():
    with pytest.raises(ValueError):
        t_critical_95(1)


def test_ci_over_five_layouts_is_student_t_not_normal():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected = 2.7764451 * np.std(values, ddof=1) / np.sqrt(5)
    assert ci95_half_width(values) == pytest.approx(expected)
    assert ci95_half_width(np.asarray(values)) == pytest.approx(expected)
    assert ci95_half_width(values) > 1.96 * np.std(values, ddof=1) / np.sqrt(5)


def test_ci_is_zero_below_two_samples():
    assert ci95_half_width([]) == 0.0
    assert ci95_half_width([3.0]) == 0.0
    np.testing.assert_array_equal(ci95_half_width(np.ones((1, 4))), np.zeros(4))


def test_ci_along_axis_matches_per_column_ci():
    stack = np.arange(15, dtype=float).reshape(5, 3) ** 1.5
    np.testing.assert_allclose(ci95_half_width(stack), [ci95_half_width(stack[:, j]) for j in range(3)])


def test_no_hand_rolled_confidence_intervals_in_experiments_or_scripts():
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}"
        for folder in ("experiments", "scripts")
        for path in (REPO_ROOT / folder).rglob("*.py")
        if "wad2image" not in path.parts and "tests" not in path.parts
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if HAND_ROLLED_CI.search(line.split("#", 1)[0])
    ]
    assert offenders == [], "use comrad.utils.confidence.ci95_half_width / t_critical_95"
