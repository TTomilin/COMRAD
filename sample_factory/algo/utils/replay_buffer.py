from __future__ import annotations

from typing import Optional
import torch
from torch import Tensor
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.typing import Device
from sample_factory.utils.utils import log

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_space,
        action_space,
        device: Device = "cpu", # Store on RAM
        share_memory: bool = True, # For async
    ):
        self.capacity = capacity
        self.obs_space = obs_space
        self.action_space = action_space
        self.device = torch.device(device)
        self.share_memory = share_memory

        self._ptr = 0
        self._size = 0

        self._storage: Optional[TensorDict] = None
        self._initialized = False

    def _init_storage(self, sample_batch: TensorDict) -> None:
        if self._initialized: return

        self._storage = TensorDict()

        def _create_buffer(tensor: Tensor) -> Tensor:
            shape = (self.capacity,) + tensor.shape[1:] # Replace batch dim with capacity
            dtype = tensor.dtype
            buf = torch.zeros(shape, dtype=dtype, device=self.device)
            if self.share_memory and not buf.is_cuda:
                buf.share_memory_()
            return buf

        # Create buffers for each sample in the sample batch
        for key, value in sample_batch.items():
            if isinstance(value, TensorDict):
                self._storage[key] = TensorDict()
                for k, v in value.items():
                    self._storage[key][k] = _create_buffer(v)
            elif isinstance(value, Tensor):
                self._storage[key] = _create_buffer(value)

        self._initialized = True
        log.debug(f"ReplayBuffer created: {self.capacity}")

    def add(self, batch: TensorDict) -> int:
        """Add batch to buffer"""
        if not self._initialized:
            self._init_storage(batch)

        batch_size = self._get_batch_size(batch)
        if batch_size == 0: return 0

        # Copy data to buffer
        if self._ptr + batch_size <= self.capacity:
            self._copy_to_storage(batch, self._ptr, self._ptr + batch_size)
        else:
            first_part = self.capacity - self._ptr
            second_part = batch_size - first_part
            first_batch = self._slice_batch(batch, 0, first_part)
            second_batch = self._slice_batch(batch, first_part, batch_size)

            self._copy_to_storage(first_batch, self._ptr, self.capacity)
            self._copy_to_storage(second_batch, 0, second_part)

        self._ptr = (self._ptr + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)

        return batch_size

    def _get_batch_size(self, batch: TensorDict) -> int:
        for key, value in batch.items():
            if isinstance(value, TensorDict):
                for k, v in value.items():
                    return v.shape[0]
            elif isinstance(value, Tensor):
                return value.shape[0]
        return 0
    
    def _slice_batch(self, batch: TensorDict, start: int, end: int) -> TensorDict:
        """Slice along first dim"""
        result = TensorDict()
        for key, value in batch.items():
            if isinstance(value, TensorDict):
                result[key] = TensorDict()
                for k, v in value.items():
                    result[key][k] = v[start:end]
            elif isinstance(value, Tensor):
                result[key] = value[start:end]
        return result

    def _copy_to_storage(self, batch: TensorDict, start: int, end: int) -> None:
        if self._storage is None:
            return
        for key, value in batch.items():
            if isinstance(value, TensorDict):
                for k, v in value.items():
                    self._storage[key][k][start:end].copy_(v.to(self.device))
            elif isinstance(value, Tensor):
                self._storage[key][start:end].copy_(value.to(self.device))

    def sample(self, batch_size: int, device: Optional[Device] = None) -> Optional[TensorDict]:
        if self._size < batch_size or self._storage is None:
            return None

        target_device = device if device is not None else str(self.device)
        indices = torch.randint(0, self._size, (batch_size,), device="cpu")
        return self._index_storage(indices, target_device)

    def _index_storage(self, indices: Tensor, device: Device) -> TensorDict:
        result = TensorDict()
        target_device = torch.device(device)

        if self._storage is None: return result

        # Index w/ indices and move to device
        for key, value in self._storage.items():
            if isinstance(value, TensorDict):
                result[key] = TensorDict()
                for k, v in value.items():
                    result[key][k] = v[indices].to(target_device)
            elif isinstance(value, Tensor):
                result[key] = value[indices].to(target_device)

        return result

    def __len__(self) -> int:
        return self._size
