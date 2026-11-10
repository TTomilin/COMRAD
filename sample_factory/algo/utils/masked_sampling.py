from typing import Optional

import gymnasium as gym
import numpy as np


def split_flat_action_mask(action_space: gym.Space, mask: Optional[np.ndarray]):
    """
    Convert a flat action mask (concatenated over discrete heads, as produced by RoleActionMaskWrapper)
    into the structure expected by gymnasium's `Space.sample(mask=...)`: an int8 array for Discrete,
    a tuple of int8 arrays for Tuple(Discrete, ...). Heads whose mask is all zeros get None (unmasked),
    so sampling never fails on a fully masked head.
    """
    if mask is None:
        return None
    if isinstance(mask, tuple):
        return mask
    flat = np.asarray(mask).reshape(-1)

    if isinstance(action_space, gym.spaces.Discrete):
        if flat.shape[0] != action_space.n:
            raise ValueError(f"Flat action mask of size {flat.shape[0]} does not match action space size {action_space.n}")
        head = (flat > 0).astype(np.int8)
        return head if head.any() else None

    if isinstance(action_space, gym.spaces.Tuple) and all(
        isinstance(space, gym.spaces.Discrete) for space in action_space.spaces
    ):
        total = sum(space.n for space in action_space.spaces)
        if flat.shape[0] != total:
            raise ValueError(f"Flat action mask of size {flat.shape[0]} does not match action space size {total}")
        heads = []
        offset = 0
        for space in action_space.spaces:
            head = (flat[offset : offset + space.n] > 0).astype(np.int8)
            heads.append(head if head.any() else None)
            offset += space.n
        return tuple(heads)

    raise NotImplementedError(f"Action masks are only supported for Discrete and Tuple(Discrete, ...), got {action_space}")


def sample_masked_action(action_space: gym.Space, mask: Optional[np.ndarray] = None):
    return action_space.sample(split_flat_action_mask(action_space, mask))
