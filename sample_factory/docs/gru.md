todos:
- fix `get_rnn_size()` in `qmix_model.py` to handle multi-layer gru
- add `forward_decomposed()` to spit out encoder outs
- add `qmix_sequence_batch_size` in `cfg.py`
- update `arguments.py` checks: reject lstm (gru only), require rollout >= 2, force per=False, enforce shared weights
- make `JointSequenceReplayBuffer` in a new file
- sequence storage (T+1 obs, T actions), uniform smaple onlym, vectorized add
- add `_prepare_joint_sequences` to `learner_qmix.py`, grouping by env/agent idx
- add `_sequential_agent_forward` for T+1 unrolling, zero out hidden states on `dones`
- add `_calculate_qmix_loss_sequential`, two fwd passes (online + target w/ no_grad)
- detach encoder outs before feeding into mixer.
- wire up init/train loops for rnn mode.

notes:
- gru only. lstm is explicitly unsupported and will throw config error.
- no PER for sequence mode. uniform only.
- target pass MUST be wrapped in `torch.no_grad()` so you dont blow up gpu mem.
- detach enc outputs before mixer (encoder only gets q-head grads).
- dont do `max(done) * (1-max(timeout))` for TD cutoff. use `max(dones * (1 - time_outs))` to handle mixed agent outcomes properly.
- keep all schedule math (train freq, warmup) in flat agent-transiton units so it behaves identically to non-rnn mode.
- sequence batch size is completely seprate from regular batch size so math doesnt silently change on users.
