"""95% confidence intervals for small samples (seeds, held-out layouts): Student-t with df = n - 1."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats


def t_critical_95(n: int) -> float:
    if n < 2:
        raise ValueError(f"a t interval needs at least 2 samples, got {n}")
    return float(stats.t.ppf(0.975, df=n - 1))


def ci95_half_width(values: Iterable[float] | np.ndarray, axis: int = 0) -> float | np.ndarray:
    """Half-width of the two-sided 95% Student-t CI of the mean along `axis` (0 when fewer than 2 samples)."""
    arr = np.asarray(values if isinstance(values, np.ndarray) else list(values), dtype=float)
    n = arr.shape[axis] if arr.ndim else 0
    if n < 2:
        return np.zeros(np.delete(arr.shape, axis)) if arr.ndim > 1 else 0.0
    half = t_critical_95(n) * arr.std(axis=axis, ddof=1) / np.sqrt(n)
    return float(half) if np.ndim(half) == 0 else half
