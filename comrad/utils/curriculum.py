import multiprocessing
from collections import deque
from typing import Optional
import numpy as np


class BatchCurriculum:
    """
    Curriculum helper for a fixed discrete batch of tasks (WADs).

    Strategies:
        uniform           - equal probability for all tasks
        learning_progress - dual EMA tracking rate of return improvement
        plr               - Prioritized Level Replay: prioritize lowest-return tasks
        sequential        - advance through tasks in index order once rolling mean
                            return over last `seq_window` episodes exceeds
                            seq_threshold * seq_max_return
        omni              - LP masked by a task-interestingness graph using the
                            Syllabus OMNI algorithm which iterates over tasks by success rate
                            (highest first), marks tasks boring if a more-mastered
                            task declares them uninteresting; boring tasks get
                            weight *= 0.001. Falls back to pure LP if no graph given.
    """

    EMA_ALPHA = 0.1

    def __init__(
        self,
        n_tasks: int,
        strategy: str = "uniform",
        seed: Optional[int] = None,
        # sequential options
        seq_threshold: float = 0.8,
        seq_max_return: float = 1.0,
        seq_window: int = 100,
        # omni: dict[int, dict[int, bool]]
        # maps evaluated_task_idx -> {other_task_idx: is_interesting (bool)}
        # e.g. {0: {1: False, 2: True}} means mastering task 0 makes task 1 boring
        interestingness: Optional[dict[int, dict[int, bool]]] = None,
    ):
        assert strategy in ("uniform", "learning_progress", "plr", "sequential", "omni"), \
            f"Unknown strategy '{strategy}'. Choose: uniform, learning_progress, plr, sequential, omni"

        ctx = multiprocessing.get_context("spawn")
        self._n = n_tasks
        self._strategy = strategy
        self._seq_threshold = seq_threshold
        self._seq_max_return = seq_max_return
        self._seq_window = seq_window

        if interestingness is not None:
            self._interestingness = interestingness
        else:
            # Default: mastering task i makes task i boring (suppress it), all others stay interesting
            self._interestingness = {
                i: {j: (j != i) for j in range(n_tasks)}
                for i in range(n_tasks)
            }

        # Shared memory
        self._weights = ctx.Array('d', [1.0 / n_tasks] * n_tasks)
        self._counts  = ctx.Array('l', [0] * n_tasks)
        self._p_fast  = ctx.Array('d', [0.0] * n_tasks)
        self._p_slow  = ctx.Array('d', [0.0] * n_tasks)
        self._seq_idx = ctx.Array('l', [0])
        self._seq_buf     = ctx.Array('d', [0.0] * seq_window)
        self._seq_buf_pos = ctx.Array('l', [0])
        self._seq_buf_len = ctx.Array('l', [0])
        self._lock    = ctx.Lock()
        
        self._rng = np.random.default_rng(seed)

    def update(self, task_idx: int, episode_return: float) -> None:
        """Called at end of each episode. No-op for uniform."""
        if self._strategy == "uniform":
            return
        with self._lock:
            self._counts[task_idx] += 1
            r = float(episode_return)
            alpha = self.EMA_ALPHA

            self._p_fast[task_idx] = r * alpha + self._p_fast[task_idx] * (1.0 - alpha)
            self._p_slow[task_idx] = (
                self._p_fast[task_idx] * alpha
                + self._p_slow[task_idx] * (1.0 - alpha)
            )

            if self._strategy == "sequential":
                self._advance_sequential(task_idx, r)
            else:
                self._recompute_weights()

    def sample(self) -> int:
        with self._lock:
            weights = np.array(self._weights[:])
        return int(self._rng.choice(self._n, p=weights))

    def task_weights(self) -> np.ndarray:
        with self._lock:
            return np.array(self._weights[:])

    def task_returns(self) -> np.ndarray:
        with self._lock:
            return np.array(self._p_fast[:])

    def _recompute_weights(self) -> None:
        """Must be called under lock."""
        if self._strategy in ("learning_progress", "omni"):
            weights = self._lp_weights()
            if self._strategy == "omni":
                weights = self._apply_interestingness_mask(weights)
        elif self._strategy == "plr":
            weights = self._plr_weights()
        else:
            return  # uniform / sequential manage weights themselves

        self._weights[:] = weights.tolist()

    def _lp_weights(self) -> np.ndarray:
        fast   = np.array(self._p_fast[:])
        slow   = np.array(self._p_slow[:])
        scores = np.abs(fast - slow) + 1e-6
        return scores / scores.sum()

    def _plr_weights(self) -> np.ndarray:
        fast   = np.array(self._p_fast[:])
        scores = (fast.max() - fast) + 1e-6
        return scores / scores.sum()

    def _apply_interestingness_mask(self, lp_dist: np.ndarray) -> np.ndarray:
        """
        Syllabus-inspired OMNI algorithm
        """
        fast = np.array(self._p_fast[:])
        tasks_by_success = np.argsort(fast)[::-1]  # highest return first

        interesting: set = set()
        boring: set = set()

        for task_idx in tasks_by_success:
            if task_idx in interesting or task_idx in boring:
                continue
            interesting.add(task_idx)

            task_map = self._interestingness.get(task_idx, {})
            for other_idx in range(self._n):
                if other_idx in interesting or other_idx in boring:
                    continue
                # If the map says this other task is not interesting given task_idx mastery
                if not task_map.get(other_idx, True):
                    boring.add(other_idx)

        moi_weight = np.ones(self._n)
        for i in boring:
            moi_weight[i] = 0.001

        dist = lp_dist * moi_weight
        return dist / dist.sum()

    def _advance_sequential(self, task_idx: int, r: float) -> None:
        idx = self._seq_idx[0]
        if task_idx != idx:
            return
        pos = self._seq_buf_pos[0]
        self._seq_buf[pos] = r
        self._seq_buf_pos[0] = (pos + 1) % self._seq_window
        self._seq_buf_len[0] = min(self._seq_buf_len[0] + 1, self._seq_window)
        
        if self._seq_buf_len[0] > 0:
            valid = list(self._seq_buf[:self._seq_buf_len[0]])
            mean_return = sum(valid) / len(valid)
            normed = mean_return / max(self._seq_max_return, 1e-6)
            if normed >= self._seq_threshold and idx < self._n - 1:
                idx += 1
                self._seq_idx[0] = idx
                self._seq_buf_pos[0] = 0
                self._seq_buf_len[0] = 0

        w = np.zeros(self._n)
        w[idx] = 1.0
        self._weights[:] = w.tolist()
