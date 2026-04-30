import numpy as np
from comrad.curriculum.base import Curriculum

class SequentialCurriculum(Curriculum):
    """
    Advances linearly through tasks when a success threshold is met
    over a rolling window of recent episodes.
    """
    def __init__(
        self,
        n_tasks: int,
        seq_threshold: float = 0.8,
        seq_max_return: float = 1.0,
        seq_window: int = 100,
        **kwargs
    ):
        super().__init__(n_tasks, **kwargs)
        self._seq_threshold = seq_threshold
        self._seq_max_return = seq_max_return
        self._seq_window = seq_window

        self._seq_idx = self.ctx.Array('l', [0])
        self._seq_buf = self.ctx.Array('d', [0.0] * seq_window)
        self._seq_buf_pos = self.ctx.Array('l', [0])
        self._seq_buf_len = self.ctx.Array('l', [0])

        # Initialize first task weight
        w = np.zeros(self._n)
        w[0] = 1.0
        self._weights[:] = w.tolist()

    def _update_logic(self, task_idx: int, episode_return: float) -> None:
        idx = self._seq_idx[0]
        if task_idx != idx:
            return

        pos = self._seq_buf_pos[0]
        self._seq_buf[pos] = episode_return
        self._seq_buf_pos[0] = (pos + 1) % self._seq_window
        self._seq_buf_len[0] = min(self._seq_buf_len[0] + 1, self._seq_window)

    def _recompute_weights(self) -> None:
        idx = self._seq_idx[0]

        if self._seq_buf_len[0] > 0:
            valid = list(self._seq_buf[:self._seq_buf_len[0]])
            mean_return = sum(valid) / len(valid)
            normed = mean_return / max(self._seq_max_return, 1e-6)

            if normed >= self._seq_threshold and idx < self._n - 1:
                idx += 1
                self._seq_idx[0] = idx
                self._seq_buf_pos[0] = 0
                self._seq_buf_len[0] = 0

                w = np.zeros(self._n)
                w[idx] = 1.0
                self._weights[:] = w.tolist()

    def _state_dict_locked(self) -> dict:
        return {
            "seq_idx": int(self._seq_idx[0]),
            "seq_buf": list(self._seq_buf[:]),
            "seq_buf_pos": int(self._seq_buf_pos[0]),
            "seq_buf_len": int(self._seq_buf_len[0]),
        }

    def _load_state_dict_locked(self, state: dict) -> None:
        self._seq_idx[0] = int(state.get("seq_idx", self._seq_idx[0]))
        self._seq_buf[:] = [float(value) for value in state.get("seq_buf", self._seq_buf[:])]
        self._seq_buf_pos[0] = int(state.get("seq_buf_pos", self._seq_buf_pos[0]))
        self._seq_buf_len[0] = int(state.get("seq_buf_len", self._seq_buf_len[0]))
