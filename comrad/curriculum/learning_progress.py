import numpy as np
from comrad.curriculum.base import Curriculum

class LearningProgress(Curriculum):
    """
    Syllabus-inspired Learning Progress curriculum tracking fast and slow EMAs.
    Progress is measured by the absolute difference between these EMAs.
    """

    def __init__(self, n_tasks: int, **kwargs):
        super().__init__(n_tasks, **kwargs)
        self._p_fast = self.ctx.Array('d', [0.0] * n_tasks)
        self._p_slow = self.ctx.Array('d', [0.0] * n_tasks)

    def _update_logic(self, task_idx: int, episode_return: float) -> None:
        """Update EMAs for task returns."""
        alpha = self.EMA_ALPHA
        # Fast EMA updates based on actual returns
        self._p_fast[task_idx] = episode_return * alpha + self._p_fast[task_idx] * (1.0 - alpha)
        # Slow EMA updates as moving average of fast EMA
        self._p_slow[task_idx] = self._p_fast[task_idx] * alpha + self._p_slow[task_idx] * (1.0 - alpha)

    def _recompute_weights(self) -> None:
        fast = np.array(self._p_fast[:])
        slow = np.array(self._p_slow[:])
        
        # Compute learning progress as absolute difference
        learning_progress = np.abs(fast - slow)
        
        # Consider tasks with progress > 0
        posidxs = [i for i, lp in enumerate(learning_progress) if lp > 0]
        any_progress = len(posidxs) > 0
        
        if any_progress:
            subprobs = learning_progress[posidxs]
        else:
            subprobs = learning_progress

        std = np.std(subprobs)
        # z-score
        subprobs = (subprobs - np.mean(subprobs)) / (std if std else 1.0)
        # sigmoid
        subprobs = 1.0 / (1.0 + np.exp(-subprobs))
        # normalize
        subprobs = subprobs / (np.sum(subprobs) + 1e-8)

        if any_progress:
            task_dist = np.zeros(self._n)
            task_dist[posidxs] = subprobs
            # normalize just in case
            task_dist = task_dist / (np.sum(task_dist) + 1e-8)
            self._weights[:] = task_dist.tolist()
        else:
            if np.sum(subprobs) == 0:
                self._weights[:] = (np.ones(self._n) / self._n).tolist()
            else:
                self._weights[:] = subprobs.tolist()

    def task_returns(self) -> np.ndarray:
        with self._lock:
            return np.array(self._p_fast[:])
