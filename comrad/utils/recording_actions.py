import numpy as np
import torch


def reshape_deterministic_actions(sampled_actions, deterministic_actions):
    """Restore multi-agent action layout after argmax flattens head-major actions."""
    if type(sampled_actions) is not type(deterministic_actions):
        return deterministic_actions

    sampled_shape = getattr(sampled_actions, "shape", None)
    deterministic_shape = getattr(deterministic_actions, "shape", None)
    if sampled_shape is None or deterministic_shape is None:
        return deterministic_actions
    if sampled_shape == deterministic_shape:
        return deterministic_actions

    sampled_numel = int(np.prod(sampled_shape))
    deterministic_numel = int(np.prod(deterministic_shape))
    if sampled_numel != deterministic_numel:
        return deterministic_actions

    if len(sampled_shape) == 2 and len(deterministic_shape) == 2 and deterministic_shape[0] == 1:
        num_agents, num_heads = sampled_shape
        head_major = deterministic_actions.reshape(num_heads, num_agents)
        if isinstance(head_major, torch.Tensor):
            return head_major.transpose(0, 1)
        return np.swapaxes(head_major, 0, 1)

    return deterministic_actions.reshape(sampled_shape)
