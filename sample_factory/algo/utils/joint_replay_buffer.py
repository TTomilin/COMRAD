"""
This extends replay_buffer as that flattens all transitions into independent samples
We also need to compute team_reward and joint_done

Learner reconstructs joint transitions from traj -> write into joint replay buffer -> sampling returns joint batches
"""

from __future__ import annotations

import threading
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.utils.replay_buffer import SumSegmentTree, MinSegmentTree
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.typing import Device
from sample_factory.utils.utils import log


class JointReplayBuffer:
    """
    Note:
    + Global state is not stored but computed on the go from obs to save RAM
    + I added a mutex so multi workers don't deadlock when sampling, though this might be inefficient if workers are proc
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_space,
        action_space,
        device: Device = "cpu",
        share_memory: bool = True,
        use_per: bool = False,
        per_omega: float = 0.6,
        per_beta_start: float = 0.4,
        per_epsilon: float = 1e-6,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if num_agents < 2:
            raise ValueError(f"num_agents must be >= 2 for multi-agent, got {num_agents}")

        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.device = torch.device(device)
        self.share_memory = share_memory

        # This pointer points at the next state in circular buffer to update priority
        self._ptr = 0
        self._size = 0

        # mutex
        self._lock = threading.Lock()

        # Storage tensors on first add
        self._storage = None
        self._initd = False

        # PER
        self.use_per = use_per
        if use_per:
            self._sum_tree = SumSegmentTree(capacity)
            self._min_tree = MinSegmentTree(capacity)
            self._max_priority = 1.0
            self._per_omega = per_omega
            self._per_beta = per_beta_start
            self._per_epsilon = per_epsilon
        else:
            self._sum_tree = None
            self._min_tree = None

    def __len__(self) -> int:
        return self._size

    def _init_storage(self, sample: TensorDict):
        if self._initd: return
        self._storage = TensorDict()

        def create_buffer(tensor: Tensor) -> Tensor:
            shape = (self.capacity,) + tensor.shape # [capacity, num_agents, ...]
            buf = torch.zeros(shape, dtype=tensor.dtype, device=self.device)
            if self.share_memory and not buf.is_cuda:
                buf.share_memory_()
            return buf

        def _init(src: TensorDict, dst: TensorDict):
            for key, val in src.items():
                if isinstance(val, TensorDict):
                    dst[key] = TensorDict()
                    _init(val, dst[key])
                else:
                    dst[key] = create_buffer(val)
        _init(sample, self._storage)

        self._storage['team_reward'] = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self._storage['joint_done'] = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self._storage['joint_time_out'] = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        if self.share_memory and not self._storage['team_reward'].is_cuda:
            self._storage['team_reward'] = self._storage['team_reward'].share_memory_()
            self._storage['joint_done'] = self._storage['joint_done'].share_memory_()
            self._storage['joint_time_out'] = self._storage['joint_time_out'].share_memory_()

        self._initd = True
        log.info(f"JointReplayBuffer: capacity={self.capacity}, num_agents={self.num_agents}, PER={self.use_per}")

    def add_joint(self, joint_transition: TensorDict):
        """
        Note: Agents should be ordered by agent_idx (0, 1, ..., N-1)
        """
        first_val = self._get_first_tensor(joint_transition)
        batch_agents = first_val.shape[0]
        if batch_agents != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} agents in joint transition, got {batch_agents}")

        with self._lock:
            if not self._initd:
                self._init_storage(joint_transition)

            self._write_at(self._ptr, joint_transition)

            if self.use_per:
                priority = self._max_priority ** self._per_omega # New transitions get max priority
                self._sum_tree.update(self._ptr, priority)
                self._min_tree.update(self._ptr, priority)

            # Next state
            self._ptr = (self._ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def add_joint_batch(self, joint_batch: TensorDict) -> int:
        first_val = self._get_first_tensor(joint_batch)
        batch_size = first_val.shape[0]
        if batch_size == 0: return 0

        for i in range(batch_size):
            single = self._extract_at(joint_batch, i)
            self.add_joint(single)

        return batch_size

    def _get_first_tensor(self, td: TensorDict) -> Tensor:
        for key, val in td.items():
            if isinstance(val, TensorDict):
                return self._get_first_tensor(val)
            return val

    def _extract_at(self, batch: TensorDict, idx: int):
        result = TensorDict()
        for key, val in batch.items():
            if isinstance(val, TensorDict):
                result[key] = self._extract_at(val, idx)
            else:
                result[key] = val[idx] # We dont clone, its copied to storage after anw
        return result

    def _write_at(self, idx: int, joint: TensorDict):
        def write(src: TensorDict, dst: TensorDict):
            for key, val in src.items():
                if key in ('team_reward', 'joint_done', 'joint_time_out'):
                    continue
                if isinstance(val, TensorDict):
                    write(val, dst[key])
                else:
                    dst[key][idx].copy_(val.to(self.device))

        write(joint, self._storage)

        if 'rewards' in joint:
            self._storage['team_reward'][idx] = joint['rewards'].sum()
        if 'dones' in joint:
            self._storage['joint_done'][idx] = joint['dones'].float().max()
        if 'time_outs' in joint:
            self._storage['joint_time_out'][idx] = joint['time_outs'].float().max()
        else:
            self._storage['joint_time_out'][idx] = 0.0

    def sample(self, batch_size: int, device) -> Tuple[TensorDict, Tensor, Tensor]:
        if self._size == 0:
            raise RuntimeError("Cannot sample from empty JointReplayBuffer")

        tdevice = device if device is not None else str(self.device)

        # return (batch, weights, indices)
        if self.use_per:
            return self._sample_per(batch_size, tdevice)
        else:
            batch = self._sample_uniform(batch_size, tdevice)
            return batch, None, None

    def _sample_uniform(self, batch_size: int, device: Device) -> TensorDict:
        # TODO: Im not sure if the RNG used here is uniform really
        # Each workers is a thread so they might the same RNG state and sample the same indice
        indices = torch.randint(0, self._size, (batch_size,), device='cpu')
        return self._gather(indices, device)

    def _sample_per(self, batch_size: int, device: Device) -> Tuple[TensorDict, Tensor, Tensor]:
        indices = self._sample_proportional(batch_size)
        weights = self._compute_is_weights(indices, device)
        indices_tensor = torch.from_numpy(indices).long()
        batch = self._gather(indices_tensor, device)

        return batch, weights, indices_tensor.to(device)

    def _sample_proportional(self, batch_size: int) -> np.ndarray:
        """Sample indices proportional to priority"""
        total_priority = self._sum_tree.sum(0, self._size)
        segment = total_priority / batch_size

        segment_starts = np.arange(batch_size) * segment
        segment_ends = segment_starts + segment
        prefixsums = np.random.uniform(segment_starts, segment_ends)

        indices = self._sum_tree.find_prefixsum_idx_batch(prefixsums)
        indices = np.clip(indices, 0, self._size - 1)
        return indices

    def _compute_is_weights(self, indices: np.ndarray, device: Device) -> Tensor:
        total_priority = self._sum_tree.sum(0, self._size)
        min_priority = self._min_tree.min(0, self._size)

        max_weight = (self._size * min_priority / total_priority) ** (-self._per_beta)
        priorities = self._sum_tree[indices]
        probs = priorities / total_priority
        weights = (self._size * probs) ** (-self._per_beta)
        weights = weights / max_weight

        return torch.from_numpy(weights.astype(np.float32)).to(device)

    def _gather(self, indices: Tensor, device: Device):
        batch = TensorDict()
        target_device = torch.device(device)

        def get_gather(src, dst):
            for key, val in src.items():
                if key in ('team_reward', 'joint_done', 'joint_time_out'):
                    continue
                if isinstance(val, TensorDict):
                    dst[key] = TensorDict()
                    get_gather(val, dst[key])
                else:
                    dst[key] = val[indices].to(target_device)
        get_gather(self._storage, batch)

        batch['team_reward'] = self._storage['team_reward'][indices].to(target_device)
        batch['joint_done'] = self._storage['joint_done'][indices].to(target_device)
        batch['joint_time_out'] = self._storage['joint_time_out'][indices].to(target_device)

        return batch

    def update_priorities(self, indices: Tensor, td_errors: Tensor):
        if not self.use_per:
            return

        indices_np = indices.cpu().numpy().astype(np.int64)
        td_errors_np = td_errors.detach().cpu().numpy()

        # Add epsilon and clamp
        priorities = np.clip(np.abs(td_errors_np) + self._per_epsilon, self._per_epsilon, 1e6)
        self._max_priority = max(self._max_priority, float(priorities.max()))

        priority_omega = priorities ** self._per_omega
        self._sum_tree.update_batch(indices_np, priority_omega)
        self._min_tree.update_batch(indices_np, priority_omega)

    def set_beta(self, beta: float) -> None:
        if self.use_per:
            self._per_beta = beta

    def get_stats(self) -> dict:
        stats = {
            'size': self._size,
            'capacity': self.capacity,
            'utilization': self._size / self.capacity,
            'num_agents': self.num_agents,
        }
        if self.use_per:
            stats['per_beta'] = self._per_beta
            stats['per_max_priority'] = self._max_priority
        return stats
