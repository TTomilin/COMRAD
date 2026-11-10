import gymnasium as gym
import numpy as np
import pytest

from comrad.envs.action_space import doom_action_space_lava_maze
from comrad.utils.doom_utils import LAVA_MAZE_ACTION_MASK, LAVA_MAZE_ORACLE_MSG_ACTION_MASK
from sample_factory.algo.utils.masked_sampling import sample_masked_action, split_flat_action_mask


def test_flat_mask_on_tuple_space_is_rejected_by_gymnasium():
    # reproduces the decorrelation crash in non_batched_sampling._reset
    space = doom_action_space_lava_maze()
    flat = np.asarray(LAVA_MAZE_ACTION_MASK[1]["masks_by_player_id"][1], dtype=np.float32)
    with pytest.raises(AssertionError):
        space.sample(flat)


@pytest.mark.parametrize("mask_spec", [LAVA_MAZE_ACTION_MASK, LAVA_MAZE_ORACLE_MSG_ACTION_MASK])
@pytest.mark.parametrize("player_id", [0, 1])
def test_sample_masked_action_respects_role_masks(mask_spec, player_id):
    space = doom_action_space_lava_maze()
    space.seed(0)
    flat = np.asarray(mask_spec[1]["masks_by_player_id"][player_id], dtype=np.float32)
    heads = split_flat_action_mask(space, flat)
    assert isinstance(heads, tuple) and len(heads) == len(space.spaces)

    offsets = np.cumsum([0] + [s.n for s in space.spaces])
    for _ in range(200):
        action = sample_masked_action(space, flat)
        assert space.contains(action)
        for head, value in enumerate(action):
            assert flat[offsets[head] + int(value)] == 1.0


def test_all_zero_head_falls_back_to_unmasked_and_discrete_and_none():
    space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(3)))
    heads = split_flat_action_mask(space, np.array([0, 0, 0, 1, 0], dtype=np.float32))
    assert heads[0] is None and heads[1].tolist() == [0, 1, 0]
    assert sample_masked_action(space, np.array([0, 0, 0, 1, 0]))[1] == 1

    discrete = gym.spaces.Discrete(3)
    assert sample_masked_action(discrete, np.array([0, 0, 1])) == 2
    assert split_flat_action_mask(discrete, None) is None
    assert space.contains(sample_masked_action(space, None))


def test_wrong_mask_size_raises():
    with pytest.raises(ValueError):
        split_flat_action_mask(doom_action_space_lava_maze(), np.ones(4))
    with pytest.raises(ValueError):
        split_flat_action_mask(gym.spaces.Discrete(3), np.ones(2))


def test_unsupported_space_raises():
    with pytest.raises(NotImplementedError):
        split_flat_action_mask(gym.spaces.Box(-1.0, 1.0, (2,)), np.ones(2))
