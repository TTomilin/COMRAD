import multiprocessing
from typing import Optional
import numpy as np


class BatchCurriculum:
    """
    Curriculum helper for a fixed discrete batch of tasks (WADs).

    Strategies:
        uniform           - equal probability for all tasks
        learning_progress - dual EMA tracking rate of return improvement
        plr               - Prioritized Level Replay
        sequential        - advance through tasks in index order
        omni              - LP masked by a task-interestingness graph
    """

    EMA_ALPHA = 0.1
    SCORE_MEAN_VALUE_L1   = "mean_value_l1"
    SCORE_MAX_VALUE_L1    = "max_value_l1"
    SCORE_MEAN_ADVANTAGE  = "mean_advantage"
    SCORE_MAX_ADVANTAGE   = "max_advantage"
    SCORE_MEAN_ENTROPY    = "mean_entropy"
    SCORE_MAX_ENTROPY     = "max_entropy"

    def __init__(
        self,
        n_tasks: int,
        strategy: str = "uniform",
        seed: Optional[int] = None,
        # sequential options
        seq_threshold: float = 0.8,
        seq_max_return: float = 1.0,
        seq_window: int = 100,        
        # plr options
        staleness_coef: float = 0.1,
        score_transform: str = "rank",
        temperature: float = 0.1,
        staleness_transform: str = "power",
        staleness_temperature: float = 1.0,
        plr_score_key: str = "mean_value_l1",
        alpha: float = 1.0,
        replay_schedule: str = "proportionate",
        replay_prob: float = 0.5,
        rho: float = 1.0,
        max_score_coef: float = 0.0,
        # omni options
        # e.g. {0: {1: False, 2: True}} means mastering task 0 makes task 1 boring
        interestingness: Optional[dict[int, dict[int, bool]]] = None,
    ):
        assert strategy in ("uniform", "learning_progress", "plr", "sequential", "omni"), \
            f"Unknown strategy '{strategy}'. Choose: uniform, learning_progress, plr, sequential, omni"

        self.ctx = multiprocessing.get_context("spawn")
        self._n = n_tasks
        self._strategy = strategy
        self._seq_threshold = seq_threshold
        self._seq_max_return = seq_max_return
        self._seq_window = seq_window

        # PLR params
        self._staleness_coef = staleness_coef
        self._score_transform = score_transform
        self._temperature = temperature
        self._staleness_transform = staleness_transform
        self._staleness_temperature = staleness_temperature
        self._plr_score_key = plr_score_key
        self._alpha = alpha
        self._replay_schedule = replay_schedule
        self._replay_prob = replay_prob
        self._rho = rho
        self._max_score_coef = max_score_coef

        if interestingness is not None:
            self._interestingness = interestingness
        else:
            # Default: mastering task i makes task i boring (suppress it), all others stay interesting
            self._interestingness = {
                i: {j: (j != i) for j in range(n_tasks)}
                for i in range(n_tasks)
            }

        # Shared memory
        self._weights          = self.ctx.Array('d', [1.0 / n_tasks] * n_tasks)
        self._counts           = self.ctx.Array('l', [0] * n_tasks)
        self._p_fast           = self.ctx.Array('d', [0.0] * n_tasks)
        self._p_slow           = self.ctx.Array('d', [0.0] * n_tasks)
        self._seq_idx          = self.ctx.Array('l', [0])
        self._seq_buf          = self.ctx.Array('d', [0.0] * seq_window)
        self._seq_buf_pos      = self.ctx.Array('l', [0])
        self._seq_buf_len      = self.ctx.Array('l', [0])
        # PLR specific
        self._task_scores      = self.ctx.Array('d', [0.0] * n_tasks)
        self._task_staleness   = self.ctx.Array('d', [0.0] * n_tasks)
        self._unseen_task_weights = self.ctx.Array('d', [1.0] * n_tasks)  # 1.0=unseen, 0.0=seen

        self._lock             = self.ctx.Lock()

        self._rng = np.random.default_rng(seed)

    def update(self, task_idx: int, episode_return: float) -> None:
        """Called at end of each episode."""
        if self._strategy == "uniform":
            return
        with self._lock:
            self._counts[task_idx] += 1
            r = float(episode_return)
            alpha = self.EMA_ALPHA
            
            if self._strategy == "learning_progress" or self._strategy == "omni":
                self._p_fast[task_idx] = r * alpha + self._p_fast[task_idx] * (1.0 - alpha)
                self._p_slow[task_idx] = (
                    self._p_fast[task_idx] * alpha + self._p_slow[task_idx] * (1.0 - alpha)
                )

            if self._strategy == "sequential":
                self._advance_sequential(task_idx, r)
            else:
                self._recompute_weights()
                
    def update_task_score(self, task_idx: int, score: dict) -> None:
        if self._strategy != "plr":
            return

        raw = float(score.get(self._plr_score_key, 0.0))
        raw_max = float(score.get(self._plr_score_key.replace("mean_", "max_"), raw))
        total_score = self._max_score_coef * raw_max + (1.0 - self._max_score_coef) * raw
        with self._lock:
            # task_score = (1-alpha)*old + alpha*new
            old = self._task_scores[task_idx]
            self._task_scores[task_idx] = (1.0 - self._alpha) * old + self._alpha * total_score
            # Mark task as seen
            self._unseen_task_weights[task_idx] = 0.0
            self._recompute_weights()



    def sample(self) -> int:
        with self._lock:
            if self._strategy == "plr":
                self._running_sample_count[0] += 1
                task_idx = self._plr_sample()
            else:
                if self._strategy != "uniform":
                    self._recompute_weights()
                weights = np.array(self._weights[:])
                task_idx = int(self._rng.choice(self._n, p=weights))
        return task_idx

    def task_weights(self) -> np.ndarray:
        with self._lock:
            return np.array(self._weights[:])

    def task_returns(self) -> np.ndarray:
        with self._lock:
            return np.array(self._p_fast[:])

    def task_scores(self) -> np.ndarray:
        """PLR scores."""
        with self._lock:
            return np.array(self._task_scores[:])

    def metrics(self) -> dict:
        """Returns metrics for plr"""
        with self._lock:
            unseen = np.array(self._unseen_task_weights[:])
            n = self._n
            proportion_seen = (n - (unseen > 0).sum()) / float(n) if n > 0 else 0.0
            return {
                "task_scores": np.array(self._task_scores[:]),
                "unseen_task_weights": unseen,
                "task_staleness": np.array(self._task_staleness[:]),
                "proportion_seen": proportion_seen,
            }

    def _sample_replay_decision(self) -> bool:
        proportion_seen = self._proportion_seen()
        if self._replay_schedule == "fixed":
            if proportion_seen >= self._rho:
                if self._rng.random() < self._replay_prob or proportion_seen >= 1.0:
                    return True
            return False
        else:  # "proportionate"
            if proportion_seen >= self._rho and self._rng.random() < min(proportion_seen, self._replay_prob):
                return True
            return False

    def _proportion_seen(self) -> float:
        unseen = np.array(self._unseen_task_weights[:])
        num_unseen = (unseen > 0).sum()
        return (self._n - num_unseen) / self._n

    def _plr_sample(self) -> int:
        proportion_seen = self._proportion_seen()
        if proportion_seen < self._rho:
            # sample unseen since not enough seen
            unseen = np.array(self._unseen_task_weights[:])
            if (unseen > 0).any():
                return self._sample_unseen_level(unseen)
            # fallthrough to replay
        do_replay = self._sample_replay_decision()
        if do_replay:
            return self._sample_replay_level()
        else:
            unseen = np.array(self._unseen_task_weights[:])
            if (unseen > 0).any():
                return self._sample_unseen_level(unseen)
            return self._sample_replay_level()

    def _sample_replay_level(self) -> int:
        weights = self.sample_weights()
        task_idx = int(self._rng.choice(self._n, p=weights))
        self._update_staleness(task_idx)
        return task_idx

    def _sample_unseen_level(self, unseen: np.ndarray) -> int:
        unseen_w = unseen / unseen.sum()
        task_idx = int(self._rng.choice(self._n, p=unseen_w))
        self._update_staleness(task_idx)
        self._unseen_task_weights[task_idx] = 0.0  # Mark as seen
        return task_idx

    def _update_staleness(self, selected_idx: int) -> None:
        if self._staleness_coef > 0:
            staleness = np.array(self._task_staleness[:])
            staleness += 1.0
            staleness[selected_idx] = 0.0
            self._task_staleness[:] = staleness.tolist()

    def sample_weights(self) -> np.ndarray:
        """
        Combines score transform + optional staleness, masking unseen tasks.
        """
        scores   = np.array(self._task_scores[:])
        unseen   = np.array(self._unseen_task_weights[:])
        staleness = np.array(self._task_staleness[:])

        weights = self._apply_score_transform(self._score_transform, self._temperature, scores)
        weights *= (1.0 - unseen)   # zero out unseen
        z = weights.sum()
        if z > 0:
            weights /= z
        else:
            seen = (unseen == 0.0)
            weights = seen.astype(float)
            if seen.any():
                weights /= weights.sum()

        if self._staleness_coef > 0:
            staleness_w = self._apply_score_transform(
                self._staleness_transform, self._staleness_temperature, staleness
            )
            staleness_w *= (1.0 - unseen)
            z_s = staleness_w.sum()
            if z_s > 0:
                staleness_w /= z_s
            else:
                staleness_w = (1.0 / self._n) * (1.0 - unseen)
            weights = (1.0 - self._staleness_coef) * weights + self._staleness_coef * staleness_w

        total = weights.sum()
        return weights / total if total > 0 else weights

    def _apply_score_transform(
        self, transform: str, temperature: float, scores: np.ndarray
    ) -> np.ndarray:
        if transform == "rank":
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp, dtype=float)
            ranks[temp] = np.arange(len(temp)) + 1
            return 1.0 / (ranks ** (1.0 / temperature))
        elif transform == "power":
            eps = 0.0 if self._staleness_coef > 0 else 1e-3
            return (np.clip(scores, 0, None) + eps) ** (1.0 / temperature)
        elif transform == "softmax":
            return np.exp(scores / temperature)
        elif transform == "max":
            w = np.zeros_like(scores)
            scores_ = scores.copy()
            scores_[np.array(self._unseen_task_weights[:]) > 0] = -np.inf
            argmax = self._rng.choice(np.flatnonzero(np.isclose(scores_, scores_.max())))
            w[argmax] = 1.0
            return w
        else:  # "constant" or unknown
            return np.ones_like(scores)


    def _recompute_weights(self) -> None:
        """Must be called under lock."""
        if self._strategy in ("learning_progress", "omni"):
            weights = self._lp_weights()
            if self._strategy == "omni":
                weights = self._apply_interestingness_mask(weights)
        else:
            return
        self._weights[:] = weights.tolist()

    def _lp_weights(self) -> np.ndarray:
        fast = np.array(self._p_fast[:])
        slow = np.array(self._p_slow[:])
        scores = np.abs(fast - slow) + 1e-6
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
