import multiprocessing
from typing import Optional
import numpy as np


class Curriculum:
    """
    Abstract base curriculum class for a fixed discrete batch of tasks (WADs).
    Requires multiprocessing arrays and locks for disjoint worker/learner access.
    """

    EMA_ALPHA = 0.1
    SCORE_MEAN_VALUE_L1   = "mean_value_l1"
    SCORE_MAX_VALUE_L1    = "max_value_l1"
    SCORE_MEAN_ADVANTAGE  = "mean_advantage"
    SCORE_MAX_ADVANTAGE   = "max_advantage"
    SCORE_MEAN_ENTROPY    = "mean_entropy"
    SCORE_MAX_ENTROPY     = "max_entropy"

    def __init__(self, n_tasks: int, seed: Optional[int] = None, **kwargs):
        self.ctx = multiprocessing.get_context("spawn")
        self._n = n_tasks
        self._rng = np.random.default_rng(seed)
        
        self._weights = self.ctx.Array('d', [1.0 / n_tasks] * n_tasks)
        self._counts = self.ctx.Array('l', [0] * n_tasks)
        self._lock = self.ctx.Lock()

    def update(self, task_idx: int, episode_return: float) -> None:
        """Called at end of each episode."""
        with self._lock:
            self._counts[task_idx] += 1
            self._update_logic(task_idx, float(episode_return))
            self._recompute_weights()

    def update_task_score(self, task_idx: int, score: dict) -> None:
        """Used by sample factory learners (e.g. PLR)"""
        pass

    def sample(self) -> int:
        with self._lock:
            task_idx = self._sample_logic()
        return task_idx

    def _sample_logic(self) -> int:
        """Isolated locked core for actually rolling the random sample"""
        weights = np.array(self._weights[:])
        return int(self._rng.choice(self._n, p=weights))

    def _update_logic(self, task_idx: int, episode_return: float) -> None:
        """Sub-classes override this to implement update behavior."""
        pass

    def _recompute_weights(self) -> None:
        """Sub-classes override this to update probabilities inside self._weights."""
        pass

    def task_weights(self) -> np.ndarray:
        with self._lock:
            return np.array(self._weights[:])

    def task_returns(self) -> np.ndarray:
        """Return fast moving average or appropriate returns proxy."""
        # By default just returns zeroes, overridden in LP/OMNI classes
        return np.zeros(self._n)

    def task_scores(self) -> np.ndarray:
        """PLR scores or similar metric."""
        return np.zeros(self._n)

    def metrics(self) -> dict:
        """General metrics reporting."""
        return {}
