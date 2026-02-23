refs:
https://arxiv.org/pdf/2109.11251
https://github.com/PKU-MARL/HARL/blob/main/harl/algorithms/actors/happo.py
https://github.com/PKU-MARL/HARL/blob/main/harl/runners/on_policy_ha_runner.py
https://github.com/cyanrain7/TRPO-in-MARL/blob/master/algorithms/happo_trainer.py

todos:
- happo args in `cfg.py` (agent order, clmap, critic hidden sizes)
- validate in `arguments.py`: requries batched sampling, num_polcies=1, hardcode normalize_input_keys to 'obs')
- make `AgentIDWrapper` to shove one-hot agent id into obs, then hook it up in `doom_utils.py` (careful with player_id=-1 temp env, manually fix obs space)
- make `HAPPOActorCritic` in `happo_model.py`, seprate encoder/core/decoder, centralized crtic needs its own seprate encoders (no RNN) + MLP
- make `HAPPOLearner` in `happo_learner.py`, sequential traning, custom `init()` like qmix, n agent optimizers + 1 critic optimzier,
- Track factor M (advatnage * unclipped post update ratios)
- register models and fix auto-naming in `train.py`, `enjoy.py`, and `record_video.py` (also fix the missing mappo/qmix regs in enjoy/record)
- hook up new learner in `learner_worker.py`

notes:
- true seq is slowr. might need more inferece workers to stop starvation.
- factor M can explod with many agents, use clamp if needed.
- custom init() in learner is fragil, gotta replicate all base setup steps exactly.
- absolutly need batched_samplng=True so env_idx is there for grouping.
- warn if per-agent rewrads differ (happo theorem assumes joint team reward).
- rnn chunk sizes gotta align perfectly for chunked full buffer eval.
- lr schduler gets upadted num_epochs times per iter. dont set self.curr_lr in _apply_lr.
- use asymetric ppo cliping to match base SF.
- no batch reordring needed, just use agent mask.
