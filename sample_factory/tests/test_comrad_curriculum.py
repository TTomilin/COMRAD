import argparse
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

import comrad.curriculum as comrad_curriculum
import comrad.train as comrad_train
from comrad.curriculum.base import Curriculum
from comrad.curriculum.learning_progress import LearningProgress
from comrad.curriculum.omni import OMNICurriculum
from comrad.curriculum.observer import load_curriculum_state, restore_curriculum_state, save_curriculum_state
from comrad.curriculum.plr import PrioritizedLevelReplay
from comrad.curriculum.sequential import SequentialCurriculum
from comrad.envs.doom_params import add_doom_env_args
from comrad.envs.multi_wad_env import MultiWADEnv
from comrad.envs.wad_catalog import WadBatch, WadInfo
from sample_factory.algo.learning.learner import Learner


class _SwapOnlyDummyEnv(gym.Env):
    def __init__(self):
        self.num_agents = 1
        self.is_multiagent = False
        self.observation_space = gym.spaces.Dict(
            {"obs": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)}
        )
        self.action_space = gym.spaces.Discrete(2)
        self.last_cfg = None

    def swap_scenario(self, cfg_path):
        self.last_cfg = cfg_path

    def reset(self, **kwargs):
        return {"obs": np.zeros((1,), dtype=np.float32)}, {"task_idx": 0}

    def step(self, action):
        return {"obs": np.zeros((1,), dtype=np.float32)}, 0.0, False, False, {"task_idx": 0}


def test_curriculum_state_round_trip(tmp_path):
    cfg = SimpleNamespace(train_dir=str(tmp_path), experiment="curriculum_state")

    curricula = []

    lp = LearningProgress(n_tasks=2, min_return=-10.0, max_return=10.0)
    lp.update(0, 6.0)
    lp.update(1, -4.0)
    curricula.append(lp)

    plr = PrioritizedLevelReplay(n_tasks=2)
    plr.update_task_score(
        0,
        {
            Curriculum.SCORE_MEAN_VALUE_L1: 1.5,
            Curriculum.SCORE_MAX_VALUE_L1: 2.5,
        },
        num_steps=3,
    )
    curricula.append(plr)

    sequential = SequentialCurriculum(n_tasks=3, seq_threshold=0.5, seq_max_return=10.0, seq_window=2)
    sequential.update(0, 8.0)
    sequential.update(0, 8.0)
    curricula.append(sequential)

    for curriculum in curricula:
        expected_state = curriculum.state_dict()
        save_curriculum_state(cfg, curriculum)

        restored = type(curriculum)(**_curriculum_ctor_kwargs(curriculum))
        assert load_curriculum_state(cfg, restored) is True
        assert restored.state_dict() == expected_state


def test_register_batch_env_forwards_sequential_args(monkeypatch):
    captured = {}
    curriculum = object()

    monkeypatch.setattr(comrad_train, "doom_env_by_name", lambda env_name: SimpleNamespace(name=env_name))
    monkeypatch.setattr(
        comrad_curriculum,
        "make_curriculum",
        lambda n_tasks, strategy="uniform", **kwargs: captured.update(
            {
                "n_tasks": n_tasks,
                "strategy": strategy,
                **kwargs,
            }
        )
        or curriculum,
    )

    import comrad.envs.wad_catalog as wad_catalog

    monkeypatch.setattr(
        wad_catalog.WadBatch,
        "from_dir",
        staticmethod(lambda _: SimpleNamespace(entries=[SimpleNamespace(name="wad_a"), SimpleNamespace(name="wad_b")])),
    )

    cfg = SimpleNamespace(
        env="doom_pitfall",
        wad_batch="batch_dir",
        wad_swap_every=5,
        curriculum="sequential",
        algo="APPO",
        seed=7,
        seq_threshold=0.9,
        seq_max_return=12.0,
        seq_window=17,
    )

    env_name, returned_curriculum = comrad_train.register_batch_env(cfg)

    assert env_name == "doom_pitfall_batch"
    assert returned_curriculum is curriculum
    assert captured["strategy"] == "sequential"
    assert captured["seq_threshold"] == pytest.approx(0.9)
    assert captured["seq_max_return"] == pytest.approx(12.0)
    assert captured["seq_window"] == 17


def test_register_batch_env_defaults_wad_swap_every_to_one(monkeypatch):
    captured = {}

    monkeypatch.setattr(comrad_train, "doom_env_by_name", lambda env_name: SimpleNamespace(name=env_name))
    monkeypatch.setattr(comrad_train, "register_env", lambda env_name, make_env_func: captured.update(
        {
            "env_name": env_name,
            "batch_spec": make_env_func.args[0],
        }
    ))
    monkeypatch.setattr(
        comrad_curriculum,
        "make_curriculum",
        lambda *args, **kwargs: object(),
    )

    import comrad.envs.wad_catalog as wad_catalog

    monkeypatch.setattr(
        wad_catalog.WadBatch,
        "from_dir",
        staticmethod(lambda _: SimpleNamespace(entries=[SimpleNamespace(name="wad_a")])),
    )

    cfg = SimpleNamespace(
        env="doom_pitfall",
        wad_batch="batch_dir",
        curriculum="uniform",
        algo="APPO",
        seed=7,
    )

    env_name, _ = comrad_train.register_batch_env(cfg)

    assert env_name == "doom_pitfall_batch"
    assert captured["env_name"] == "doom_pitfall_batch"
    assert captured["batch_spec"].swap_every == 1


class RecordingPLR(PrioritizedLevelReplay):
    def __init__(self):
        super().__init__(n_tasks=1)
        self.calls = []

    def update_task_score(self, task_idx: int, score: dict, num_steps: int = 1) -> None:
        self.calls.append((task_idx, dict(score), num_steps))


def test_plr_partial_scores_merge_across_updates_and_use_abs_advantage():
    learner = object.__new__(Learner)
    learner.curriculum = RecordingPLR()
    learner.cfg = SimpleNamespace(with_vtrace=False, rollout=4)
    learner.actor_critic = SimpleNamespace(action_space=gym.spaces.Discrete(2))
    learner._plr_partial_scores = {}

    first_gpu_buffer = {
        "returns": torch.tensor([1.0, -2.0, 3.0, -4.0]),
        "values": torch.zeros(4),
        "advantages": torch.tensor([1.0, -2.0, 3.0, -4.0]),
        "valids": torch.tensor([True, True, True, True]),
        "task_idx": torch.tensor([0, 0, 0, 0], dtype=torch.int32),
        "env_idx": torch.tensor([5, 5, 5, 5], dtype=torch.int32),
        "agent_idx": torch.tensor([1, 1, 1, 1], dtype=torch.int32),
        "dones": torch.tensor([False, False, False, False]),
        "action_logits": torch.zeros(4, 2),
    }
    second_gpu_buffer = {
        "returns": torch.tensor([5.0, -6.0, 0.0, 0.0]),
        "values": torch.zeros(4),
        "advantages": torch.tensor([5.0, -6.0, 0.0, 0.0]),
        "valids": torch.tensor([True, True, False, False]),
        "task_idx": torch.tensor([0, 0, -1, -1], dtype=torch.int32),
        "env_idx": torch.tensor([5, 5, -1, -1], dtype=torch.int32),
        "agent_idx": torch.tensor([1, 1, -1, -1], dtype=torch.int32),
        "dones": torch.tensor([False, True, False, False]),
        "action_logits": torch.zeros(4, 2),
    }

    Learner._compute_plr_task_scores(learner, first_gpu_buffer, experience_size=4)
    assert learner.curriculum.calls == []
    assert learner._plr_partial_scores

    Learner._compute_plr_task_scores(learner, second_gpu_buffer, experience_size=4)

    assert len(learner.curriculum.calls) == 1
    task_idx, score, num_steps = learner.curriculum.calls[0]
    assert task_idx == 0
    assert num_steps == 6
    assert score[Curriculum.SCORE_MEAN_VALUE_L1] == pytest.approx(3.5)
    assert score[Curriculum.SCORE_MEAN_ADVANTAGE] == pytest.approx(3.5)
    assert score[Curriculum.SCORE_MAX_VALUE_L1] == pytest.approx(6.0)
    assert score[Curriculum.SCORE_MAX_ADVANTAGE] == pytest.approx(6.0)
    assert learner._plr_partial_scores == {}


def test_plr_joint_env_updates_merge_across_agents():
    learner = object.__new__(Learner)
    learner.curriculum = RecordingPLR()
    learner.cfg = SimpleNamespace(with_vtrace=False, rollout=2)
    learner.actor_critic = SimpleNamespace(action_space=gym.spaces.Discrete(2))
    learner._plr_partial_scores = {}

    gpu_buffer = {
        "returns": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "values": torch.zeros(4),
        "advantages": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "valids": torch.tensor([True, True, True, True]),
        "task_idx": torch.tensor([0, 0, 0, 0], dtype=torch.int32),
        "env_idx": torch.tensor([5, 5, 5, 5], dtype=torch.int32),
        "agent_idx": torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        "dones": torch.tensor([False, True, False, True]),
        "action_logits": torch.zeros(4, 2),
    }

    Learner._compute_plr_task_scores(learner, gpu_buffer, experience_size=4)

    assert len(learner.curriculum.calls) == 1
    task_idx, score, num_steps = learner.curriculum.calls[0]
    assert task_idx == 0
    assert num_steps == 4
    assert score[Curriculum.SCORE_MEAN_VALUE_L1] == pytest.approx(2.5)


def test_plr_task_weights_track_live_replay_distribution():
    curriculum = PrioritizedLevelReplay(
        n_tasks=3,
        score_transform="power",
        temperature=1.0,
        staleness_coef=0.0,
        alpha=1.0,
    )

    curriculum.update_task_score(1, {Curriculum.SCORE_MEAN_VALUE_L1: 2.0})

    assert curriculum.task_weights() == pytest.approx([0.0, 1.0, 0.0])


def test_plr_staleness_updates_refresh_cached_weights():
    curriculum = PrioritizedLevelReplay(
        n_tasks=2,
        score_transform="power",
        temperature=1.0,
        staleness_coef=1.0,
        staleness_transform="power",
        staleness_temperature=1.0,
        alpha=1.0,
    )
    curriculum.update_task_score(0, {Curriculum.SCORE_MEAN_VALUE_L1: 1.0})
    curriculum.update_task_score(1, {Curriculum.SCORE_MEAN_VALUE_L1: 1.0})

    with curriculum._lock:
        curriculum._update_staleness(0)

    assert curriculum.task_weights() == pytest.approx([0.0, 1.0])


def test_plr_max_replay_redraws_ties_each_sample():
    curriculum = PrioritizedLevelReplay(
        n_tasks=3,
        score_transform="max",
        staleness_coef=0.0,
        alpha=1.0,
        seed=7,
    )
    for task_idx in range(3):
        curriculum.update_task_score(task_idx, {Curriculum.SCORE_MEAN_VALUE_L1: 1.0})

    samples = [curriculum.sample() for _ in range(20)]

    assert len(set(samples)) > 1


def test_learning_progress_is_uniform_while_some_tasks_are_unseen():
    curriculum = LearningProgress(n_tasks=3, min_return=0.0, max_return=1.0)
    curriculum.update(0, 1.0)

    with curriculum._lock:
        curriculum._recompute_weights()

    assert curriculum.task_weights() == pytest.approx([1.0 / 3.0] * 3)


def test_learning_progress_does_not_zero_negative_objective_tasks_when_lp_is_flat():
    curriculum = LearningProgress(n_tasks=2, min_return=-10.0, max_return=10.0)

    for _ in range(200):
        curriculum.update(0, -5.0)
        curriculum.update(1, 5.0)

    with curriculum._lock:
        curriculum._recompute_weights()

    assert curriculum.task_weights() == pytest.approx([0.5, 0.5])


def test_learning_progress_mixes_uniform_mass_for_zero_success_tasks():
    curriculum = LearningProgress(n_tasks=3, min_return=0.0, max_return=1.0, uniform_prob=0.25)

    with curriculum._lock:
        curriculum._p_seen[:] = [1.0, 1.0, 1.0]
        curriculum._p_fast[:] = [0.5, 0.0, 0.0]
        curriculum._p_slow[:] = [0.4, 0.0, 0.0]
        curriculum._recompute_weights()

    weights = curriculum.task_weights()
    assert weights[0] > weights[1]
    assert weights[1] > 0.0
    assert weights[2] > 0.0


def test_omni_sparse_interestingness_graph_defaults_missing_edges_to_boring():
    curriculum = OMNICurriculum(
        n_tasks=3,
        tasks=["a", "b", "c"],
        interestingness={"a": {"a": True, "b": False}},
        min_return=0.0,
        max_return=1.0,
    )

    with curriculum._lock:
        curriculum._p_seen[:] = [1.0, 1.0, 1.0]
        curriculum._p_fast[:] = [0.9, 0.2, 0.1]
        curriculum._p_slow[:] = [0.6, 0.1, 0.05]
        curriculum._p_true[:] = [0.9, 0.2, 0.1]
        curriculum._recompute_weights()

    weights = curriculum.task_weights()
    assert weights[0] > 0.99
    assert weights[1] < 0.01
    assert weights[2] < 0.01


def test_omni_orders_prerequisites_by_mastery_not_raw_objective_scale():
    curriculum = OMNICurriculum(
        n_tasks=2,
        tasks=["a", "b"],
        interestingness={
            "a": {"a": True, "b": False},
            "b": {"a": True, "b": True},
        },
        min_return=0.0,
        max_return=10.0,
        uniform_prob=0.0,
    )

    with curriculum._lock:
        curriculum._p_seen[:] = [1.0, 1.0]
        curriculum._p_fast[:] = [0.9, 0.2]
        curriculum._p_slow[:] = [0.8, 0.1]
        # Raw objective scale for task b is larger even though task a is more mastered.
        curriculum._p_true[:] = [1.0, 10.0]
        curriculum._recompute_weights()

    weights = curriculum.task_weights()
    assert weights[0] > 0.99
    assert weights[1] < 0.01


def test_restore_curriculum_state_rejects_task_order_mismatch(monkeypatch):
    cfg = SimpleNamespace(restart_behavior="resume", load_checkpoint_kind="latest", train_dir="/tmp", experiment="x")
    stateful = LearningProgress(n_tasks=2, tasks=["easy", "hard"], seed=3)
    stateful.update(0, 1.0)

    checkpoint = {"curriculum_state": stateful.state_dict()}

    monkeypatch.setattr(Learner, "checkpoint_dir", staticmethod(lambda cfg, policy_id: "/tmp/checkpoint_p0"))
    monkeypatch.setattr(Learner, "get_checkpoints", staticmethod(lambda checkpoint_dir, pattern="checkpoint_*": ["ckpt"]))
    monkeypatch.setattr(Learner, "load_checkpoint", staticmethod(lambda checkpoints, device: checkpoint))

    restored = LearningProgress(n_tasks=2, tasks=["hard", "easy"], seed=3)

    with pytest.raises(ValueError):
        restore_curriculum_state(cfg, restored)


def test_multi_wad_env_init_does_not_mutate_live_curriculum_state(tmp_path):
    base_cfg = tmp_path / "base.cfg"
    base_cfg.write_text("doom_scenario_path = placeholder.wad\n")

    wad_paths = []
    for idx in range(3):
        wad_path = tmp_path / f"task_{idx}.wad"
        wad_path.write_bytes(b"WAD")
        wad_paths.append(wad_path)

    batch = WadBatch([WadInfo(f"task_{idx}", str(path)) for idx, path in enumerate(wad_paths)])
    curriculum = PrioritizedLevelReplay(
        n_tasks=3,
        staleness_coef=1.0,
        score_transform="power",
        temperature=1.0,
        staleness_transform="power",
        staleness_temperature=1.0,
        alpha=1.0,
        seed=7,
    )

    before = curriculum.state_dict()
    env = MultiWADEnv(
        env=_SwapOnlyDummyEnv(),
        batch=batch,
        base_cfg=str(base_cfg),
        swap_every=1,
        seed=0,
        curriculum=curriculum,
    )
    env.close()

    after = curriculum.state_dict()
    assert after["rng_counter"] == before["rng_counter"]
    assert after["weights"] == before["weights"]
    assert after["task_staleness"] == before["task_staleness"]


def test_learner_checkpoint_includes_curriculum_state_but_not_partial_plr_fragments():
    learner = object.__new__(Learner)
    learner.train_step = 7
    learner.env_steps = 11
    learner.best_performance = 3.5
    learner.actor_critic = SimpleNamespace(state_dict=lambda: {"model": 1})
    learner.optimizer = SimpleNamespace(state_dict=lambda: {"optimizer": 1})
    learner.curr_lr = 1e-4
    learner.curriculum = LearningProgress(n_tasks=2, tasks=["a", "b"], seed=4)
    learner._plr_partial_scores = {5: {"task_idx": 0, "num_steps": 3}}

    checkpoint = Learner._get_checkpoint_dict(learner)

    assert "curriculum_state" in checkpoint
    assert checkpoint["curriculum_state"]["task_ids"] == ["a", "b"]
    assert "plr_partial_scores" not in checkpoint


def test_doom_args_default_to_per_episode_wad_swaps():
    parser = argparse.ArgumentParser()
    add_doom_env_args(parser)

    cfg = parser.parse_args([])

    assert cfg.wad_swap_every == 1


def _curriculum_ctor_kwargs(curriculum):
    kwargs = {"n_tasks": curriculum._n}
    if isinstance(curriculum, LearningProgress):
        kwargs.update(
            p_theta=curriculum.p_theta,
            max_return=curriculum.max_return,
            min_return=curriculum.min_return,
            uniform_prob=curriculum.uniform_prob,
        )
    if isinstance(curriculum, PrioritizedLevelReplay):
        kwargs.update(
            staleness_coef=curriculum._staleness_coef,
            score_transform=curriculum._score_transform,
            temperature=curriculum._temperature,
            staleness_transform=curriculum._staleness_transform,
            staleness_temperature=curriculum._staleness_temperature,
            plr_score_key=curriculum._plr_score_key,
            alpha=curriculum._alpha,
            replay_schedule=curriculum._replay_schedule,
            replay_prob=curriculum._replay_prob,
            rho=curriculum._rho,
            max_score_coef=curriculum._max_score_coef,
            eps=curriculum._eps,
        )
    if isinstance(curriculum, SequentialCurriculum):
        kwargs.update(
            seq_threshold=curriculum._seq_threshold,
            seq_max_return=curriculum._seq_max_return,
            seq_window=curriculum._seq_window,
        )
    return kwargs
