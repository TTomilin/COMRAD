refs:
https://arxiv.org/abs/1803.11485 (qmix)
https://arxiv.org/abs/1706.05296 (vdn)

todos:
- add qmix args and validation in `cfg.py` (embed dim, hypernet hidden, batch size, ensure buffer > learning starts).
- make `JointReplayBuffer` in `joint_replay_buffer.py`: rip out the `_pending` dict, stop storing state/next_state (compute on the fly instead), and fix PER tree to use `.update()` instead of `__setitem__`.
- make mixers in `qmix_model.py`: `VDNMixer` (just sum) and `QMixMixer` (hypernets with `torch.abs()` for monotonic weights).
- make `QMixLearner` in `learner_qmix.py`:
  - vectorize agent forward pass by flattening agents into batch dim `[B*N, ...]`.
  - build `_compute_global_state()` to construct global state from obs on-demand.
  - cast actions to `int64` before using `.gather()`.
  - shove in a NaN gradient check right after `loss.backward()`.
  - fix `train()` to return a proper dict with stats (not None) when the buffer is still warming up.
- patch `_QMixActorCriticWrapper` it needs the missing sf methods (`device_for_input_tensor`, `type_for_input_tensor`) and a proper `obs_normalizer` init.

notes:
- computing global state on demand saves ~3gb of buffer mem. do not store it.
- strict assumption: synchronous stepping only. joint transitoins must have all agents data at once.
- vectorized forward is 2-4x faster than looping over agents, keep it flat.
- always cast actions to int64 or `.gather()` will blow up.
- per segment trees must use `update()`, `__setitem__` is broken.
- returning None when buffer is warming up breaks the whole pipeline, always return the dict with env_steps and policy_id.
- nan grad check is critical, loop thru named_parameters and log/abort step if nans found.
