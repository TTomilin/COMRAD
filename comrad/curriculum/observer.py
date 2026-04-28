import json
import os
from os.path import join

import numpy as np
import torch

from sample_factory.algo.runners.runner import AlgoObserver
from sample_factory.algo.learning.learner import Learner
from sample_factory.utils.utils import experiment_dir, log


def curriculum_state_file(cfg) -> str:
    return join(experiment_dir(cfg), "curriculum_state.json")


def save_curriculum_state(cfg, curriculum) -> None:
    if curriculum is None:
        return

    state_path = curriculum_state_file(cfg)
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w") as state_file:
        json.dump(curriculum.state_dict(), state_file, indent=2)
    os.replace(tmp_path, state_path)


def load_curriculum_state(cfg, curriculum) -> bool:
    if curriculum is None:
        return False

    state_path = curriculum_state_file(cfg)
    if not os.path.isfile(state_path):
        return False

    with open(state_path, "r") as state_file:
        state = json.load(state_file)

    curriculum.load_state_dict(state)
    return True


class CurriculumObserver(AlgoObserver):
    def __init__(self, curriculum):
        self.curriculum = curriculum

    def on_init(self, runner) -> None:
        return None

    def on_training_step(self, runner, training_iteration_since_resume: int) -> None:
        return None

    def on_stop(self, runner) -> None:
        return None

    def extra_summaries(self, runner, policy_id, writer, env_steps) -> None:
        if policy_id != 0 or self.curriculum is None:
            return

        weights = self.curriculum.task_weights()
        if weights.size > 0:
            nonzero_weights = weights[weights > 0]
            if nonzero_weights.size > 0:
                entropy = float(-(nonzero_weights * np.log(nonzero_weights)).sum())
                writer.add_scalar("curriculum/entropy", entropy, env_steps)
            writer.add_scalar("curriculum/nonzero_probability_tasks", int(np.count_nonzero(weights)), env_steps)

        task_returns = self.curriculum.task_returns()
        if task_returns.size > 0 and not np.allclose(task_returns, 0.0):
            writer.add_scalar("curriculum/mean_task_return", float(np.mean(task_returns)), env_steps)

        metrics = self.curriculum.metrics()
        for key, value in metrics.items():
            if np.isscalar(value):
                writer.add_scalar(f"curriculum/{key}", float(value), env_steps)
            elif isinstance(value, np.ndarray) and value.size > 0 and np.issubdtype(value.dtype, np.number):
                writer.add_scalar(f"curriculum/mean_{key}", float(np.mean(value)), env_steps)


def restore_curriculum_state(cfg, curriculum) -> None:
    if curriculum is None or cfg.restart_behavior != "resume":
        return

    checkpoint_dir = Learner.checkpoint_dir(cfg, 0)
    load_kind = getattr(cfg, "load_checkpoint_kind", "latest")
    pattern = f"{dict(latest='checkpoint', best='best')[load_kind]}_*"
    checkpoints = Learner.get_checkpoints(checkpoint_dir, pattern=pattern)
    if not checkpoints:
        return

    checkpoint_dict = Learner.load_checkpoint(checkpoints, torch.device("cpu"))
    if checkpoint_dict is None:
        return

    state = checkpoint_dict.get("curriculum_state", None)
    if state is None:
        raise RuntimeError("Checkpoint is missing curriculum_state; cannot safely resume curriculum-enabled training")

    curriculum.load_state_dict(state)
    log.info("Restored curriculum state from checkpoint %s", checkpoints[-1])
