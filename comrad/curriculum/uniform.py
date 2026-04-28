from comrad.curriculum.base import Curriculum

class UniformCurriculum(Curriculum):
    """Samples tasks purely uniformly."""

    def __init__(self, n_tasks: int, **kwargs):
        super().__init__(n_tasks, **kwargs)
        # Weights instantly set in base

    def update(self, task_idx: int, episode_return: float) -> None:
        # Uniform doesn't need to recalculate weights or even count episodes
        # But we still run base update to maintain counts
        super().update(task_idx, episode_return)

    def _update_logic(self, task_idx: int, episode_return: float) -> None:
        pass

    def _recompute_weights(self) -> None:
        # Array keeps original initialization values
        pass
