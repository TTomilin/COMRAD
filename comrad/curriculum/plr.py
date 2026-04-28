import numpy as np
from comrad.curriculum.base import Curriculum

class PrioritizedLevelReplay(Curriculum):
    """
    Syllabus-inspired Prioritized Level Replay (PLR) tracking PPO rollout metrics natively.
    Replays tasks proportional to learning potential metrics and random unseen tasks.
    """

    def __init__(
        self,
        n_tasks: int,
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
        eps: float = 0.05,
        **kwargs
    ):
        super().__init__(n_tasks, **kwargs)
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
        self._eps = eps

        self._task_scores = self.ctx.Array('d', [0.0] * n_tasks)
        self._task_score_steps = self.ctx.Array('l', [0] * n_tasks)
        self._task_staleness = self.ctx.Array('d', [0.0] * n_tasks)
        self._unseen_task_weights = self.ctx.Array('d', [1.0] * n_tasks)

    def _update_logic(self, task_idx: int, episode_return: float) -> None:
        """Called conventionally by workers. PLR relies on central PPO learner metrics natively."""
        pass

    def update_task_score(self, task_idx: int, score: dict, num_steps: int = 1) -> None:
        """Inject from sample factory using native PLR learner loops."""
        raw = float(score.get(self._plr_score_key, 0.0))
        raw_max = float(score.get(self._plr_score_key.replace("mean_", "max_"), raw))
        total_score = self._max_score_coef * raw_max + (1.0 - self._max_score_coef) * raw
        num_steps = max(int(num_steps), 1)

        with self._lock:
            old = self._task_scores[task_idx]
            self._task_scores[task_idx] = (1.0 - self._alpha) * old + self._alpha * total_score
            self._task_score_steps[task_idx] += num_steps
            self._unseen_task_weights[task_idx] = 0.0
            self._recompute_weights()

    def _sample_logic(self) -> int:
        """Override to implement complex replay schedule logic directly inside lock."""
        proportion_seen = self._proportion_seen()

        if proportion_seen < self._rho:
            unseen = np.array(self._unseen_task_weights[:])
            if (unseen > 0).any():
                return self._sample_unseen_level(unseen)
            return self._sample_replay_level() # fallback to replay if all seen but still under rho

        do_replay = self._sample_replay_decision(proportion_seen)
        if do_replay:
            return self._sample_replay_level()
        else:
            unseen = np.array(self._unseen_task_weights[:])
            if (unseen > 0).any():
                return self._sample_unseen_level(unseen)
            return self._sample_replay_level()

    def _proportion_seen(self) -> float:
        unseen = np.array(self._unseen_task_weights[:])
        return (self._n - (unseen > 0).sum()) / self._n

    def _sample_replay_decision(self, proportion_seen: float) -> bool:
        if self._replay_schedule == "fixed":
            if proportion_seen >= self._rho:
                if self._random() < self._replay_prob or proportion_seen >= 1.0:
                    return True
            return False
        else:  # proportionate
            if proportion_seen >= self._rho and self._random() < min(proportion_seen, self._replay_prob):
                return True
            return False

    def _sample_replay_level(self) -> int:
        weights = self._calculate_score_weights()
        task_idx = self._choice(self._n, p=weights)
        self._update_staleness(task_idx)
        return task_idx

    def _sample_unseen_level(self, unseen: np.ndarray) -> int:
        unseen_w = unseen / unseen.sum()
        task_idx = self._choice(self._n, p=unseen_w)
        self._update_staleness(task_idx)
        return task_idx

    def _update_staleness(self, selected_idx: int) -> None:
        if self._staleness_coef > 0:
            staleness = np.array(self._task_staleness[:])
            staleness += 1.0
            staleness[selected_idx] = 0.0
            self._task_staleness[:] = staleness.tolist()
            self._recompute_weights()

    def _calculate_score_weights(self) -> np.ndarray:
        scores = np.array(self._task_scores[:])
        unseen = np.array(self._unseen_task_weights[:])
        staleness = np.array(self._task_staleness[:])

        weights = self._apply_score_transform(self._score_transform, self._temperature, scores)
        weights *= (1.0 - unseen)
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

    def _recompute_weights(self) -> None:
        self._weights[:] = self._calculate_score_weights().tolist()

    def _apply_score_transform(self, transform: str, temperature: float, scores: np.ndarray) -> np.ndarray:
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
            argmax_candidates = np.flatnonzero(np.isclose(scores_, scores_.max()))
            argmax = argmax_candidates[self._choice(len(argmax_candidates))]
            w[argmax] = 1.0
            return w
        elif transform == "eps_greedy":
            w = np.zeros_like(scores)
            w[scores.argmax()] = 1.0 - self._eps
            w += self._eps / self._n
            return w
        return np.ones_like(scores)

    def task_scores(self) -> np.ndarray:
        with self._lock:
            return np.array(self._task_scores[:])

    def metrics(self) -> dict:
        with self._lock:
            unseen = np.array(self._unseen_task_weights[:])
            return {
                "task_scores": np.array(self._task_scores[:]),
                "task_score_steps": np.array(self._task_score_steps[:]),
                "unseen_task_weights": unseen,
                "task_staleness": np.array(self._task_staleness[:]),
                "proportion_seen": self._proportion_seen(),
            }

    def _state_dict_locked(self) -> dict:
        return {
            "task_scores": list(self._task_scores[:]),
            "task_score_steps": list(self._task_score_steps[:]),
            "task_staleness": list(self._task_staleness[:]),
            "unseen_task_weights": list(self._unseen_task_weights[:]),
        }

    def _load_state_dict_locked(self, state: dict) -> None:
        self._task_scores[:] = [float(value) for value in state.get("task_scores", self._task_scores[:])]
        self._task_score_steps[:] = [int(value) for value in state.get("task_score_steps", self._task_score_steps[:])]
        self._task_staleness[:] = [float(value) for value in state.get("task_staleness", self._task_staleness[:])]
        self._unseen_task_weights[:] = [
            float(value) for value in state.get("unseen_task_weights", self._unseen_task_weights[:])
        ]
        self._recompute_weights()
