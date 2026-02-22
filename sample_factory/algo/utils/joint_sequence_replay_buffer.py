"""
JointReplayBuffer wihtout PER
TODO: Refactor to remove dupe code with JointReplayBuffer, or not
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.typing import Device
from sample_factory.utils.utils import log


class JointSequenceReplayBuffer:
    def __init__(
        self,
        capacity_sequences: int,
        seq_len: int,
        num_agents: int,
        obs_space,
        action_space,
        device: Device = "cpu",
        share_memory: bool = True,
        rnn_state_size: int = 0,
        rng_seed: Optional[int] = None,
    ):
        if capacity_sequences <= 0:
            raise ValueError(f"capacity_sequences must be positive, got {capacity_sequences}")
        if seq_len <= 0:
            raise ValueError(f'seq_len must be positive, got {seq_len}')
        if num_agents < 2:
            raise ValueError(f"num_agents must be >= 2 for multi-agent, got {num_agents}")

        self.capacity = capacity_sequences
        self.seq_len = int(seq_len)
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.device = torch.device(device)
        self.share_memory = share_memory
        self.rnn_state_size = int(rnn_state_size)

        seed_sequence = np.random.SeedSequence(rng_seed)
        _, torch_seed_sequence = seed_sequence.spawn(2)
        torch_seed = int(torch_seed_sequence.generate_state(1, dtype=np.uint32)[0])
        self._torch_rng = torch.Generator(device="cpu")
        self._torch_rng.manual_seed(torch_seed)

        # This pointer points at the next state in circular buffer to update priority
        self._ptr = 0
        self._size = 0

        # mutex
        self._lock = threading.Lock()

        # Storage tensors on first add
        self._storage = None
        self._initd = False

    def __len__(self) -> int:
        return self._size

    def _first_tensor(self, td: TensorDict) -> Tensor:
        for _, val in td.items():
            if isinstance(val, TensorDict):
                return self._first_tensor(val)
            return val
        raise RuntimeError('TensorDict is empty')

    def _init_storage(self, sample_batch: TensorDict):
        if self._initd: return
        self._storage = TensorDict()

        def create_buffer(tensor: Tensor) -> Tensor:
            shape = (self.capacity,) + tuple(tensor.shape[1:])
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
        _init(sample_batch, self._storage)

        self._initd = True
        log.info(f"JointSequenceReplayBuffer: capacity={self.capacity}, seq_len={self.seq_len}, num_agents={self.num_agents}")

    def _slice_batch(self, batch: TensorDict, start: int, end: int) -> TensorDict:
        out = TensorDict()
        for key, val in batch.items():
            if isinstance(val, TensorDict):
                out[key] = self._slice_batch(val, start, end)
            else:
                out[key] = val[start:end]
        return out

    def _validate_batch_shapes(self, batch: TensorDict):
        if 'obs' not in batch:
            raise ValueError('Missing key in sequence batch: obs')

        def obs_is_valid(obs_td: TensorDict):
            for _, obs_val in obs_td.items():
                if isinstance(obs_val, TensorDict):
                    obs_is_valid(obs_val)
                else:
                    if obs_val.dim() < 3:
                        raise ValueError(f'obs leaf must be [B,T+1,N,...], got {tuple(obs_val.shape)}')
                    if obs_val.shape[1] != self.seq_len + 1:
                        raise ValueError(f'obs time dim must be T+1={self.seq_len + 1}, got {obs_val.shape[1]}')
                    if obs_val.shape[2] != self.num_agents:
                        raise ValueError(f'obs agent dim must be N={self.num_agents}, got {obs_val.shape[2]}')

        obs_is_valid(batch['obs'])

        for key in ('actions', 'rewards', 'dones', 'time_outs'):
            if key not in batch:
                raise ValueError(f'Missing key in sequence batch: {key}')

            val = batch[key]
            if val.dim() < 3:
                raise ValueError(f'{key} must be [B,T,N,...], got {tuple(val.shape)}')
            if val.shape[1] != self.seq_len:
                raise ValueError(f'{key} time dim must be T={self.seq_len}, got {val.shape[1]}')
            if val.shape[2] != self.num_agents:
                raise ValueError(f'{key} agent dim must be N={self.num_agents}, got {val.shape[2]}')

        if 'rnn_states' not in batch:
            raise ValueError('Missing key in sequence batch: rnn_states')
        rnn_states = batch['rnn_states']
        if rnn_states.dim() != 3:
            raise ValueError(f'rnn_states must be [B,N,R], got {tuple(rnn_states.shape)}')
        if rnn_states.shape[1] != self.num_agents:
            raise ValueError(f'rnn_states agent dim must be N={self.num_agents}, got {rnn_states.shape[1]}')
        if self.rnn_state_size > 0 and rnn_states.shape[2] != self.rnn_state_size:
            raise ValueError(f'rnn_states feature dim must be R={self.rnn_state_size}, got {rnn_states.shape[2]}')

    def add_sequence(self, sequence_dict: TensorDict) -> int:
        def r_unsqueeze(src: TensorDict) -> TensorDict:
            out = TensorDict()
            for key, val in src.items():
                if isinstance(val, TensorDict):
                    out[key] = r_unsqueeze(val)
                else:
                    out[key] = val.unsqueeze(0)
            return out
        sequence_batch = r_unsqueeze(sequence_dict)
        return self.add_sequence_batch(sequence_batch)

    def add_sequence_batch(self, batch_dict: TensorDict) -> int:
        first = self._first_tensor(batch_dict)
        batch_size = int(first.shape[0])
        if batch_size == 0: return 0

        self._validate_batch_shapes(batch_dict)

        if batch_size > self.capacity:
            start = batch_size - self.capacity
            batch_dict = self._slice_batch(batch_dict, start, batch_size)
            batch_size = self.capacity

        with self._lock:
            if not self._initd: self._init_storage(batch_dict)
            indices = (torch.arange(batch_size, dtype=torch.long) + self._ptr) % self.capacity
            indices = indices.to(self.device)

            def write_r(src: TensorDict, dst: TensorDict):
                for key, val in src.items():
                    if isinstance(val, TensorDict):
                        write_r(val, dst[key])
                    else:
                        dst[key].index_copy_(0, indices, val.to(self.device))
            write_r(batch_dict, self._storage)

            self._ptr = (self._ptr + batch_size) % self.capacity
            self._size = min(self._size + batch_size, self.capacity)
        return batch_size

    def _gather(self, indices: Tensor, device: Device) -> TensorDict:
        target_device = torch.device(device)
        idx = indices.to(self.device)
        batch = TensorDict()
        def gatherr(src: TensorDict, dst: TensorDict):
            for key, val in src.items():
                if isinstance(val, TensorDict):
                    dst[key] = TensorDict()
                    gatherr(val, dst[key])
                else:
                    dst[key] = val.index_select(0, idx).to(target_device)
        gatherr(self._storage, batch)
        return batch

    def sample(self, batch_size: int, device: Optional[Device] = None) -> Tuple[TensorDict, None, None]:
        if self._size == 0:
            raise RuntimeError('Cannot sample from empty JointSequenceReplayBuffer')

        target_device = device if device is not None else str(self.device)
        with self._lock:
            indices = torch.randint(0, self._size, (batch_size,), device='cpu', generator=self._torch_rng)
            batch = self._gather(indices, target_device)
            return batch, None, None

    def set_beta(self, beta: float):
        return None

    def update_priorities(self, indices: Tensor, td_errors: Tensor):
        return None

    def get_stats(self) -> dict:
        return {
            'size': self._size,
            'capacity': self.capacity,
            'utilization': self._size / self.capacity,
            'seq_len': self.seq_len,
            'num_agents': self.num_agents,
        }
