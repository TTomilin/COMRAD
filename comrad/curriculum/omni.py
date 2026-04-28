import numpy as np
from typing import Optional
from comrad.curriculum.learning_progress import LearningProgress

class OMNICurriculum(LearningProgress):
    """
    Syllabus-inspired OMNI curriculum tracking fast and slow EMAs masked by tasks dependencies 
    suppressing mathematically boring tasks once prerequisites are met.
    """
    
    def __init__(self, n_tasks: int, interestingness: Optional[dict] = None, tasks: Optional[list] = None, **kwargs):
        super().__init__(n_tasks, **kwargs)
        self.tasks = tasks or list(range(n_tasks))
        if interestingness is not None:
            bad = set(interestingness.keys()) - set(self.tasks)
            if bad:
                raise ValueError(
                    f"interestingness keys not in tasks: {bad}. "
                    f"Use task identifiers only."
            )
            self._interestingness = interestingness
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
        true_rates = np.array(self._p_true[:])
        
        tasks_by_success = np.argsort(true_rates)[::-1]  # Highest success/return first
        interesting = set()
        boring = set()

        for task_idx in tasks_by_success:
            if task_idx in interesting or task_idx in boring:
                continue
            interesting.add(task_idx)

            task = self.tasks[task_idx]
            task_map = self._interestingness.get(task, {})
            # Given that we consider task_idx mastered, determine boring consequences
            for other_idx in range(self._n):
                if other_idx in interesting or other_idx in boring:
                    continue
                other_task = self.tasks[other_idx]
                # If map says this task is purely uninteresting knowing the mastered task's proficiency
                if not task_map.get(other_task, True):
                    boring.add(other_idx)

        # Build weight mask where boring items are heavily penalized
        moi_weight = np.ones(self._n)
        for i in boring:
            moi_weight[i] = 0.001

        dist = lp_dist * moi_weight
        norm_dist = dist / dist.sum() if dist.sum() > 0 else np.ones(self._n) / self._n
        
        self._weights[:] = norm_dist.tolist()
