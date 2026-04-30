import multiprocessing
import os
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

    def __init__(self, n_tasks: int, seed: Optional[int] = None, tasks: Optional[list] = None, **kwargs):
        self.ctx = multiprocessing.get_context("spawn")
        self._n = n_tasks
        self._base_seed = int(seed) if seed is not None else int.from_bytes(os.urandom(8), "little")
        self._rng_counter = self.ctx.Value("Q", 0)
        self._task_ids = tuple(tasks) if tasks is not None else None

        self._weights = self.ctx.Array('d', [1.0 / n_tasks] * n_tasks)
        self._counts = self.ctx.Array('l', [0] * n_tasks)
        self._lock = self.ctx.Lock()

    def update(self, task_idx: int, episode_return: float) -> None:
        """Called at end of each episode."""
        with self._lock:
            self._counts[task_idx] += 1
            self._update_logic(task_idx, float(episode_return))
            self._recompute_weights()

    def update_task_score(self, task_idx: int, score: dict, num_steps: int = 1) -> None:
        """Used by sample factory learners (e.g. PLR)"""
        pass

    def sample(self) -> int:
        with self._lock:
            task_idx = self._sample_logic()
        return task_idx

    def _sample_logic(self) -> int:
        """Isolated locked core for actually rolling the random sample"""
        weights = np.array(self._weights[:])
        return self._choice(self._n, p=weights)

    def _next_rng(self) -> np.random.Generator:
        """
        Generate randomness from a single shared draw counter so spawned workers
        do not each replay an identical local RNG stream.
        """
        counter = self._rng_counter.value
        self._rng_counter.value = counter + 1
        seed = (self._base_seed + 0x9E3779B97F4A7C15 * counter) & ((1 << 64) - 1)
        return np.random.default_rng(seed)

    def _choice(self, n: int, p=None) -> int:
        return int(self._next_rng().choice(n, p=p))

    def _random(self) -> float:
        return float(self._next_rng().random())

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

    def state_dict(self) -> dict:
        with self._lock:
            state = {
                "version": 1,
                "curriculum_type": type(self).__name__,
                "n_tasks": self._n,
                "base_seed": int(self._base_seed),
                "rng_counter": int(self._rng_counter.value),
                "weights": list(self._weights[:]),
                "counts": list(self._counts[:]),
            }
            if self._task_ids is not None:
                state["task_ids"] = list(self._task_ids)
            state.update(self._state_dict_locked())
            return state

    def load_state_dict(self, state: dict) -> None:
        expected_type = type(self).__name__
        actual_type = state.get("curriculum_type", expected_type)
        if actual_type != expected_type:
            raise ValueError(f"Expected curriculum state for {expected_type}, got {actual_type}")
        if int(state.get("n_tasks", self._n)) != self._n:
            raise ValueError(f"Expected curriculum state for {self._n} tasks, got {state.get('n_tasks')}")
        expected_task_ids = list(self._task_ids) if self._task_ids is not None else None
        actual_task_ids = state.get("task_ids", None)
        if expected_task_ids is not None and actual_task_ids != expected_task_ids:
            raise ValueError(f"Expected curriculum task ids {expected_task_ids}, got {actual_task_ids}")

        with self._lock:
            self._base_seed = int(state.get("base_seed", self._base_seed))
            self._rng_counter.value = int(state.get("rng_counter", 0))
            self._weights[:] = [float(weight) for weight in state.get("weights", self._weights[:])]
            self._counts[:] = [int(count) for count in state.get("counts", self._counts[:])]
            self._load_state_dict_locked(state)

    def _state_dict_locked(self) -> dict:
        return {}

    def _load_state_dict_locked(self, state: dict) -> None:
        pass
