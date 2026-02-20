from __future__ import annotations

import numpy as np
import pytest
import torch

from sample_factory.algo.utils.joint_replay_buffer import JointReplayBuffer
from sample_factory.algo.utils.tensor_dict import TensorDict

NUM_AGENTS = 2
CAPACITY = 128
NUM_TRANSITIONS = 80
BATCH_SIZE = 16


def _make_joint_transition(step: int) -> TensorDict:
    return TensorDict(
        {
            "obs": torch.full((NUM_AGENTS, 1), float(step), dtype=torch.float32),
            "next_obs": torch.full((NUM_AGENTS, 1), float(step + 1), dtype=torch.float32),
            "actions": torch.tensor([step % 3, (step + 1) % 3], dtype=torch.int64),
            "rewards": torch.tensor([float(step), float(step) + 0.5], dtype=torch.float32),
            "dones": torch.tensor([False, False]),
            "time_outs": torch.tensor([False, False]),
        }
    )


def _build_buffer(*, seed: int, use_per: bool) -> JointReplayBuffer:
    replay_buffer = JointReplayBuffer(
        capacity=CAPACITY,
        num_agents=NUM_AGENTS,
        obs_space=None,
        action_space=None,
        device="cpu",
        share_memory=False,
        use_per=use_per,
        rng_seed=seed,
    )
    for step in range(NUM_TRANSITIONS):
        replay_buffer.add_joint(_make_joint_transition(step))
    return replay_buffer


def _sample_signature(replay_buffer: JointReplayBuffer, *, use_per: bool) -> torch.Tensor:
    batch, _, indices = replay_buffer.sample(BATCH_SIZE, "cpu")
    if use_per:
        return indices.cpu().clone()
    return batch["team_reward"].cpu().clone()


@pytest.mark.parametrize("use_per", [False, True])
def test_replay_sampling_reproducible_with_same_seed(use_per: bool) -> None:
    seed = 1729
    replay_a = _build_buffer(seed=seed, use_per=use_per)
    replay_b = _build_buffer(seed=seed, use_per=use_per)

    for _ in range(8):
        torch.randint(0, 1000, (17,))
        np.random.uniform(size=11)
        sig_a = _sample_signature(replay_a, use_per=use_per)

        torch.randint(0, 1000, (31,))
        np.random.uniform(size=23)
        sig_b = _sample_signature(replay_b, use_per=use_per)

        assert torch.equal(sig_a, sig_b)


@pytest.mark.parametrize("use_per", [False, True])
def test_replay_sampling_differs_with_different_seed(use_per: bool) -> None:
    replay_a = _build_buffer(seed=111, use_per=use_per)
    replay_b = _build_buffer(seed=222, use_per=use_per)

    any_difference = False
    for _ in range(12):
        sig_a = _sample_signature(replay_a, use_per=use_per)
        sig_b = _sample_signature(replay_b, use_per=use_per)
        if not torch.equal(sig_a, sig_b):
            any_difference = True
            break

    assert any_difference
