import numpy as np
from typing import Optional
from comrad.curriculum.learning_progress import LearningProgress

class OMNICurriculum(LearningProgress):
    """
    Syllabus-inspired OMNI curriculum tracking fast and slow EMAs masked by tasks dependencies
    suppressing mathematically boring tasks once prerequisites are met.
    """

    def __init__(self, n_tasks: int, tasks: list, interestingness: Optional[dict] = None, **kwargs):
        super().__init__(n_tasks, tasks=tasks, **kwargs)
        self.tasks = tasks or list(range(n_tasks))
        if interestingness is not None:
            bad = set(interestingness.keys()) - set(self.tasks)
            if bad:
                raise ValueError(
                    f"interestingness keys not in tasks: {bad}. "
                    f"Use task identifiers only."
            )
            self._interestingness = {
                task: {other: bool(is_interesting) for other, is_interesting in mapping.items()}
                for task, mapping in interestingness.items()
            }
        else:
            # Default: mastery of a task makes itself boring, but other tasks remain interesting.
            self._interestingness = {
                t: {other: (other != t) for other in self.tasks}
                for t in self.tasks
            }

    def _recompute_weights(self) -> None:
        """Combine LP weight calculation with "boring" penalty mask."""
        super()._recompute_weights() # Update weights purely by LP

        lp_dist = np.array(self._weights[:])
        mastery = np.array(self._p_fast[:])

        # OMNI prerequisite masking must be ordered by a comparable mastery signal.
        # Raw true_objective values can differ wildly across tasks, so use the
        # normalized fast EMA from LearningProgress instead.
        tasks_by_success = np.argsort(mastery)
        interesting_tasks = set()
        boring_tasks = set()

        for task_idx in tasks_by_success[::-1]:
            if task_idx in interesting_tasks or task_idx in boring_tasks:
                continue
            interesting_tasks.add(task_idx)

            task = self.tasks[task_idx]
            if task not in self._interestingness:
                continue

            for other_idx, other_task in enumerate(self.tasks):
                if other_task not in self._interestingness[task]:
                    boring_tasks.add(other_idx)
                    continue
                if (
                    other_idx not in interesting_tasks
                    and other_idx not in boring_tasks
                    and not self._interestingness[task][other_task]
                ):
                    boring_tasks.add(other_idx)

        # Build weight mask where boring items are heavily penalized
        moi_weight = np.ones(self._n)
        for i in boring_tasks:
            moi_weight[i] = 0.001

        dist = lp_dist * moi_weight
        norm_dist = dist / dist.sum() if dist.sum() > 0 else np.ones(self._n) / self._n

        self._weights[:] = norm_dist.tolist()
