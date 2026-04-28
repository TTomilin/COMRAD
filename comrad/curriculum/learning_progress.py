import numpy as np
from comrad.curriculum.base import Curriculum

class LearningProgress(Curriculum):
    """
    Syllabus-inspired Learning Progress curriculum tracking fast and slow EMAs.
    Progress is measured by the absolute difference between these EMAs.
    """

    def __init__(self, n_tasks: int, p_theta: float = 0.1, **kwargs):
        super().__init__(n_tasks, **kwargs)
        self.p_theta = p_theta
        self._p_fast = self.ctx.Array('d', [0.0] * n_tasks)
        self._p_slow = self.ctx.Array('d', [0.0] * n_tasks)
        self._p_true = self.ctx.Array('d', [0.0] * n_tasks)
        self._p_seen = self.ctx.Array('d', [0.0] * n_tasks)  # 0.0 = unseen, 1.0 = seen

    def _update_logic(self, task_idx: int, episode_return: float) -> None:
        """Update EMAs for task returns."""
        alpha = self.EMA_ALPHA
        if self._p_seen[task_idx] == 0.0: # init both EMAs to episode return on first observation to avoid learning progress spike differences
            self._p_fast[task_idx] = episode_return
            self._p_slow[task_idx] = episode_return
            self._p_seen[task_idx] = 1.0
        else:
            old_fast = self._p_fast[task_idx]    
            # Fast EMA updates based on actual returns
            self._p_fast[task_idx] = episode_return * alpha + old_fast * (1.0 - alpha)
            # Slow EMA updates as moving average of fast EMA based on value BEFORE this fast update
            self._p_slow[task_idx] = self._p_fast[task_idx] * alpha + self._p_slow[task_idx] * (1.0 - alpha)
        self._p_true[task_idx] = episode_return

    def _reweight(self, p: np.ndarray) -> np.ndarray:
        numerator = p * (1.0 - self.p_theta)
        denominator = p + self.p_theta * (1.0 - 2.0 * p)
        return numerator / denominator

    def _recompute_weights(self) -> None:
        fast = np.array(self._p_fast[:])
        slow = np.array(self._p_slow[:])
        true_rates = np.array(self._p_true[:])
        
        # Compute learning progress as absolute difference of reweighted EMAs
        learning_progress = np.abs(self._reweight(fast) - self._reweight(slow))
        
        # Consider tasks with progress > 0 or true rates > 0
        posidxs = [i for i, lp in enumerate(learning_progress) if lp > 0 or true_rates[i] > 0]
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
            if np.sum(subprobs) == 0: # shouldn't happen with sigmoid, but just in case
                self._weights[:] = (np.ones(self._n) / self._n).tolist()
            else:
                self._weights[:] = subprobs.tolist()

    def task_returns(self) -> np.ndarray:
        with self._lock:
            return np.array(self._p_fast[:])
