from comrad.curriculum.base import Curriculum
from comrad.curriculum.uniform import UniformCurriculum
from comrad.curriculum.sequential import SequentialCurriculum
from comrad.curriculum.learning_progress import LearningProgress
from comrad.curriculum.omni import OMNICurriculum
from comrad.curriculum.plr import PrioritizedLevelReplay

def make_curriculum(n_tasks: int, strategy: str = "uniform", **kwargs) -> Curriculum:
    """Factory to spin up proper curriculum based on configuration."""
    strategies = {
        "uniform": UniformCurriculum,
        "sequential": SequentialCurriculum,
        "learning_progress": LearningProgress,
        "omni": OMNICurriculum,
        "plr": PrioritizedLevelReplay,
    }
    
    assert strategy in strategies, f"Unknown strategy '{strategy}'. Available: {list(strategies.keys())}"
    
    return strategies[strategy](n_tasks=n_tasks, **kwargs)

__all__ = [
    "Curriculum",
    "UniformCurriculum",
    "SequentialCurriculum",
    "LearningProgress",
    "OMNICurriculum",
    "PrioritizedLevelReplay",
    "make_curriculum"
]
