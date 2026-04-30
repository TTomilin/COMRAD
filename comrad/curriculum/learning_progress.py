import numpy as np
from comrad.curriculum.base import Curriculum

class LearningProgress(Curriculum):
    """
    Syllabus-inspired Learning Progress curriculum tracking fast and slow EMAs over
    normalized per-task success metrics. The raw terminal `true_objective` signal is
    tracked separately for diagnostics/checkpoint continuity.
    """

    def __init__(
        self,
        n_tasks: int,
        p_theta: float = 0.1,
        max_return: float = 100.0,
        min_return: float = -100.0,
        uniform_prob: float = 0.25,
        **kwargs,
    ):
        super().__init__(n_tasks, **kwargs)
        self.p_theta = p_theta
        self.max_return = max_return
        self.min_return = min_return
        self.uniform_prob = uniform_prob
        self._stale_dist = self.ctx.Value('b', True)
        self._p_fast = self.ctx.Array('d', [0.0] * n_tasks)
        self._p_slow = self.ctx.Array('d', [0.0] * n_tasks)
        self._p_true = self.ctx.Array('d', [0.0] * n_tasks)
        self._p_seen = self.ctx.Array('d', [0.0] * n_tasks)  # 0.0 = unseen, 1.0 = seen

    def _sample_logic(self) -> int:
        if self._stale_dist.value:
            self._recompute_weights()
            self._stale_dist.value = False
        weights = np.array(self._weights[:])
        return self._choice(self._n, p=weights)

    def update(self, task_idx: int, task_score: float) -> None:
        with self._lock:
            self._counts[task_idx] += 1
            self._update_logic(task_idx, float(task_score))
            self._stale_dist.value = True

    def _normalize(self, task_score: float) -> float:
        """Map raw `true_objective` success metrics to [0, 1]."""
        r_min = self.min_return
        r_max = self.max_return
        if r_max <= r_min:
            return 0.0
        normalized = (task_score - r_min) / (r_max - r_min)
        return float(np.clip(normalized, 0.0, 1.0))

    def _update_logic(self, task_idx: int, task_score: float) -> None:
        """Update normalized LP EMAs plus a raw `true_objective` EMA."""
        p = self._normalize(task_score)
        alpha = self.EMA_ALPHA

        if self._p_seen[task_idx] == 0.0:
            self._p_fast[task_idx] = p
            self._p_slow[task_idx] = p
            self._p_true[task_idx] = task_score
            self._p_seen[task_idx] = 1.0
        else:
            old_fast = self._p_fast[task_idx]
            old_true = self._p_true[task_idx]
            self._p_fast[task_idx] = p * alpha + old_fast * (1.0 - alpha)
            self._p_slow[task_idx] = self._p_fast[task_idx] * alpha + self._p_slow[task_idx] * (1.0 - alpha)
            self._p_true[task_idx] = task_score * alpha + old_true * (1.0 - alpha)

    def _reweight(self, p: np.ndarray) -> np.ndarray:
        numerator = p * (1.0 - self.p_theta)
        denominator = p + self.p_theta * (1.0 - 2.0 * p)
        return numerator / denominator

    def _recompute_weights(self) -> None:
        seen = np.array(self._p_seen[:]) > 0.0
        unseen = ~seen

        # LP is undefined until every task has at least some signal.
        # Keep the curriculum uniform during this cold-start phase instead of
        # forcing unseen-only sweeps or collapsing onto the first updated task.
        if unseen.any():
            self._weights[:] = (np.ones(self._n) / self._n).tolist()
            return

        fast = np.array(self._p_fast[:])
        slow = np.array(self._p_slow[:])
        # Compute learning progress as absolute difference of reweighted EMAs
        learning_progress = np.abs(self._reweight(fast) - self._reweight(slow))
        # Keep tasks with any normalized success signal eligible when LP is flat.
        # `_p_true` stores raw objectives for OMNI ordering, so using its sign here
        # would incorrectly bias sampling toward tasks whose reward scale is positive.
        posidxs = [i for i, lp in enumerate(learning_progress) if lp > 0.0 or fast[i] > 0.0]
        any_progress = len(posidxs) > 0
        subprobs = learning_progress[posidxs] if any_progress else learning_progress

        std = np.std(subprobs)
        # z-score
        subprobs = (subprobs - np.mean(subprobs)) / (std if std else 1.0)
        # sigmoid
        subprobs = 1.0 / (1.0 + np.exp(-subprobs))
        # normalize
        subprobs = subprobs / (np.sum(subprobs))

        if any_progress:
            task_dist = np.zeros(self._n)
            task_dist[posidxs] = subprobs
        else:
            task_dist = subprobs

        if self.uniform_prob > 0.0:
            uniform = np.ones(self._n) / self._n
            task_dist = (1.0 - self.uniform_prob) * task_dist + self.uniform_prob * uniform

        task_dist = task_dist / np.sum(task_dist)
        self._weights[:] = task_dist.tolist()

    def task_returns(self) -> np.ndarray:
        with self._lock:
            return np.array(self._p_fast[:])

    def _state_dict_locked(self) -> dict:
        return {
            "stale_dist": bool(self._stale_dist.value),
            "p_fast": list(self._p_fast[:]),
            "p_slow": list(self._p_slow[:]),
            "p_true": list(self._p_true[:]),
            "p_seen": list(self._p_seen[:]),
        }

    def _load_state_dict_locked(self, state: dict) -> None:
        self._stale_dist.value = bool(state.get("stale_dist", True))
        self._p_fast[:] = [float(value) for value in state.get("p_fast", self._p_fast[:])]
        self._p_slow[:] = [float(value) for value in state.get("p_slow", self._p_slow[:])]
        self._p_true[:] = [float(value) for value in state.get("p_true", self._p_true[:])]
        self._p_seen[:] = [float(value) for value in state.get("p_seen", self._p_seen[:])]
