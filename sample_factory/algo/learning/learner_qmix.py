from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import (
    LEARNER_ENV_STEPS,
    POLICY_ID_KEY,
    STATS_KEY,
    TRAIN_STATS,
    memory_stats,
)
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.joint_replay_buffer import JointReplayBuffer
from sample_factory.algo.utils.joint_sequence_replay_buffer import JointSequenceReplayBuffer
from sample_factory.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_factory.algo.utils.torch_utils import synchronize, to_scalar
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, InitModelData, PolicyID
from sample_factory.utils.utils import log

from sf.doom.qmix_model import QMixAgentNet, QMixActorCritic


class QMixLearner(Learner):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
        global_env_steps_tensor: Optional[Tensor] = None,
    ):
        super().__init__(cfg, env_info, policy_versions_tensor, policy_id, param_server)

        self.num_agents = getattr(cfg, 'num_agents', 2)
        self.use_rnn = bool(getattr(cfg, 'use_rnn', False))
        if self.num_agents < 2:
            raise ValueError(f"QMIX requires num_agents >= 2, got {self.num_agents}")

        self.global_env_steps_tensor = global_env_steps_tensor
        self.replay_buffer = None

        # Networks
        self.agent_net = None
        self.target_agent_net = None
        self.mixer = None
        self.target_mixer = None
        self.obs_normalizer = None

        # Training
        self.total_env_steps_for_training = 0
        self.last_target_update_step = 0
        self.last_train_env_steps = 0

        # PER
        self.use_per = getattr(cfg, 'per', False)
        self.per_beta_start = getattr(cfg, 'per_beta_start', 0.4)
        self.per_beta_frames = getattr(cfg, 'per_beta_frames', 100000)

        # debuggin
        self._invalid_sequence_groups = 0
        # Stats
        self._last_q_tot_mean = 0.0
        self._last_q_tot_max = 0.0
        self._last_td_error_mean = 0.0
        self._last_grad_norm = 0.0
        # More stats
        self._last_agent_qs_mean = 0.0
        self._last_agent_qs_max = 0.0
        self._last_agent_qs_min = 0.0
        self._last_target_q_tot_mean = 0.0
        self._last_target_before_clamp = 0.0
        self._last_q_std_across_actions = 0.0
        self._last_done_ratio = 0.0
        self._last_timeout_ratio = 0.0

    def init(self) -> InitModelData:
        from sample_factory.algo.utils.shared_buffers import policy_device
        import numpy as np

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        self.device = policy_device(self.cfg, self.policy_id)
        log.info(f"QMixLearner device: {self.device}")
        self.actor_critic = QMixActorCritic(
            self.cfg,
            self.env_info.obs_space,
            self.env_info.action_space,
            num_agents=self.num_agents,
        )
        self.actor_critic.model_to_device(self.device)

        # ref
        self.agent_net = self.actor_critic.agent_net
        self.mixer = self.actor_critic.mixer
        self.obs_normalizer = self.actor_critic.obs_normalizer

        # Target net
        self.target_agent_net = copy.deepcopy(self.agent_net)
        self.target_agent_net.eval()
        self.target_mixer = copy.deepcopy(self.mixer)
        self.target_mixer.eval()

        if self.use_rnn:
            cfg_rnn_size = get_rnn_size(self.cfg)
            model_rnn_size = self.agent_net.get_rnn_size()
            log.info(f"QMIX RNN size check: model={model_rnn_size}, cfg={cfg_rnn_size}")
            if model_rnn_size != cfg_rnn_size:
                raise ValueError(f"QMIX RNN size mismatch: model={model_rnn_size} cfg={cfg_rnn_size}")

        self.state_dim = self._calc_state_dim()
        log.info(f"QMIX state_dim={self.state_dim}")
        q_clamp = getattr(self.cfg, 'q_value_clamp', 100.0)
        tau = getattr(self.cfg, 'target_update_tau', 1.0)
        use_huber = getattr(self.cfg, 'use_huber_loss', True)
        log.info(f"QMIX: q_clamp={q_clamp}, target_tau={tau}, use_huber={use_huber}")

        replay_buffer_seed = None if self.cfg.seed is None else int(self.cfg.seed) + int(self.policy_id)
        if self.use_rnn:
            transitions_per_sequence = self.num_agents * self.cfg.rollout
            capacity_sequences = self.cfg.replay_buffer_size // transitions_per_sequence
            if capacity_sequences < 1:
                raise ValueError(f"QMIX/VDN RNN: replay_buffer_size={self.cfg.replay_buffer_size} is too small for num_agents={self.num_agents}, rollout={self.cfg.rollout}. Need at least {transitions_per_sequence} transitions")

            self.replay_buffer = JointSequenceReplayBuffer(
                capacity_sequences=capacity_sequences,
                seq_len=self.cfg.rollout,
                num_agents=self.num_agents,
                obs_space=self.env_info.obs_space,
                action_space=self.env_info.action_space,
                device='cpu',
                share_memory=not self.cfg.serial_mode,
                rnn_state_size=self.agent_net.get_rnn_size(),
                rng_seed=replay_buffer_seed,
            )
            log.info(f"JointSequenceReplayBuffer: capacity={capacity_sequences} sequences, {capacity_sequences * transitions_per_sequence} transitions")
            log.info(f"QMIX RNN updates: qmix_sequence_batch_size={self._qmix_batch_size()} (effective transitions/update={self._qmix_batch_size() * self.cfg.rollout * self.num_agents})")
        else:
            buffer_capacity = self.cfg.replay_buffer_size // self.num_agents
            self.replay_buffer = JointReplayBuffer(
                capacity=buffer_capacity,
                num_agents=self.num_agents,
                obs_space=self.env_info.obs_space,
                action_space=self.env_info.action_space,
                device='cpu',
                share_memory=not self.cfg.serial_mode,
                use_per=self.use_per,
                per_omega=getattr(self.cfg, 'per_omega', 0.6),
                per_beta_start=self.per_beta_start,
                rng_seed=replay_buffer_seed,
            )
            log.info(f"JointReplayBuffer: capacity={buffer_capacity}, {buffer_capacity * self.num_agents} transitions")

        # Adam optimizer
        params = list(self.agent_net.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.cfg.learning_rate)

        self.curr_lr = self.cfg.learning_rate
        return self._get_init_model_data()

    def _get_init_model_data(self) -> InitModelData:
        # (policy_id, state_dict, device, policy_version)
        return (self.policy_id, None if self.cfg.serial_mode else self.actor_critic.state_dict(), self.device, 0)

    def _calc_state_dim(self) -> int:
        return self.agent_net.encoder_out_size * self.num_agents

    def _first_tensor(self, td: TensorDict) -> Tensor:
        """Copy from Joint seq replay buffer"""
        for _, val in td.items():
            if isinstance(val, TensorDict):
                return self._first_tensor(val)
            return val
        raise RuntimeError('TensorDict is empty')

    def _compute_global_state(self, obs: TensorDict, encoder_net = None) -> Tensor:
        if encoder_net is None:
            encoder_net = self.agent_net
        if isinstance(obs, dict):
            sample_tensor = self._first_tensor(obs)
        else:
            sample_tensor = obs
        batch_size = sample_tensor.shape[0]
        num_agents = sample_tensor.shape[1]

        def flatten_dct(d):
            if isinstance(d, dict):
                return {k: v.flatten(end_dim=1) for k, v in d.items()}
            return d.flatten(end_dim=1)

        flat_obs = flatten_dct(obs) # [B, N, ...] -> [B*N, ...]
        with torch.no_grad():
            encoded = encoder_net.encode(flat_obs) # [B*N, encoder_out]
        encoder_out_size = encoded.shape[-1]
        encoded = encoded.view(batch_size, num_agents, encoder_out_size) # [B*N, encoder_out] -> [B, N, encoder_out]
        return encoded.flatten(start_dim=1) # [B, N, encoder_out] -> [B, N*encoder_out]

    def _update_target_networks(self, tau: float = 1.0):
        if tau < 1.0:
            with torch.no_grad():
                for t, o in zip(self.target_agent_net.parameters(), self.agent_net.parameters()):
                    t.data.mul_(1 - tau).add_(o.data, alpha=tau)
                for t, o in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                    t.data.mul_(1 - tau).add_(o.data, alpha=tau)
        else:
            if self.train_step - self.last_target_update_step >= self.cfg.target_update_interval:
                self.target_agent_net.load_state_dict(self.agent_net.state_dict())
                self.target_mixer.load_state_dict(self.mixer.state_dict())
                self.last_target_update_step = self.train_step
                log.debug(f"Hard updated target networks at step {self.train_step}")

    def _vectorized_agent_forward(self, obs: TensorDict, agent_net: QMixAgentNet) -> Tensor:
        """Do vectorized forward pass"""
        batch_size = self._first_tensor(obs).shape[0] if isinstance(obs, dict) else obs.shape[0]

        def flatten_td(td):
            result = TensorDict()
            for key, val in td.items():
                if isinstance(val, TensorDict): result[key] = flatten_td(val)
                else: result[key] = val.flatten(end_dim=1)
            return result
        flat_obs = flatten_td(obs)

        # Dummy state for non-sequential transition path
        num_flat = batch_size * self.num_agents
        rnn_size = agent_net.get_rnn_size()
        rnn_states = None
        if rnn_size > 0:
            rnn_states = torch.zeros(num_flat, rnn_size, device=self.device)
        q_values, _ = agent_net(flat_obs, rnn_states=rnn_states) # [B*N, num_actions]
        num_actions = q_values.shape[-1]
        q_values = q_values.view(batch_size, self.num_agents, num_actions) # [B*N, A] -> [B, N, A]

        return q_values

    def _sequential_agent_forward(self, obs, dones, rnn_states, agent_net) -> Tuple[Tensor, Tensor]:
        sample_obs = self._first_tensor(obs) if isinstance(obs, dict) else obs
        t_steps = dones.shape[1]
        if sample_obs.shape[1] != t_steps + 1:
            raise ValueError(f"Expected T+1={t_steps + 1} obs steps, got {sample_obs.shape[1]}")
        if dones.shape[1] != t_steps:
            raise ValueError(f"Expected T={t_steps} done steps, got {dones.shape[1]}")
        if rnn_states.shape[-1] != agent_net.get_rnn_size():
            raise ValueError(f"Expected rnn_states feature dim={agent_net.get_rnn_size()}, got {rnn_states.shape[-1]}")

        batch_size = sample_obs.shape[0]
        obs_steps = sample_obs.shape[1]
        num_agents = sample_obs.shape[2]
        if num_agents != self.num_agents: raise ValueError(f"Expected num_agents={self.num_agents}, got {num_agents}")

        def flatten_step_obs(step_obs: TensorDict) -> TensorDict:
            flat = TensorDict()
            for key, val in step_obs.items():
                if isinstance(val, TensorDict):
                    flat[key] = flatten_step_obs(val)
                else:
                    flat[key] = val.reshape(val.shape[0] * val.shape[1], *val.shape[2:])
            return flat

        rnn_flat = rnn_states.reshape(batch_size * num_agents, -1)
        q_values_list = []
        encoder_outs_list = []
        for t in range(obs_steps):
            step_obs = obs[:, t]
            flat_obs = flatten_step_obs(step_obs)

            q_flat, new_rnn_flat, encoder_flat = agent_net.forward_decomposed(flat_obs, rnn_flat)
            num_actions = q_flat.shape[-1]
            encoder_dim = encoder_flat.shape[-1]
            q_values_list.append(q_flat.view(batch_size, num_agents, num_actions))
            encoder_outs_list.append(encoder_flat.view(batch_size, num_agents, encoder_dim))
            if t < t_steps:
                done_mask = dones[:, t, :].reshape(batch_size * num_agents, 1).to(new_rnn_flat.dtype)
                new_rnn_flat = new_rnn_flat * (1.0 - done_mask)
            rnn_flat = new_rnn_flat

        q_values = torch.stack(q_values_list, dim=1)
        encoder_outs = torch.stack(encoder_outs_list, dim=1)
        assert q_values.shape[1] == t_steps + 1
        assert encoder_outs.shape[1] == t_steps + 1
        return q_values, encoder_outs

    def _calculate_qmix_loss_sequential(self, batch: TensorDict, weights = None) -> Tuple[Tensor, Tensor]:
        # Mostly copied from _calculate_qmix_loss
        # TODO: Remove dupe code, or changes should be synced correctly
        obs = batch['obs']
        actions = batch['actions'].long()
        rewards = batch['rewards'].float()
        dones = batch['dones'].float()
        time_outs = batch['time_outs'].float() if 'time_outs' in batch else torch.zeros_like(dones)
        rnn_states = batch['rnn_states']
        t_steps = actions.shape[1]
        obs_steps = self._first_tensor(obs).shape[1]
        if obs_steps != t_steps + 1:
            raise ValueError(f"Expected T+1={t_steps + 1} obs steps, got {obs_steps}")
        if dones.shape[1] != t_steps:
            raise ValueError(f"Expected T={t_steps} done steps, got {dones.shape[1]}")
        if time_outs.shape[1] != t_steps:
            raise ValueError(f"Expected T={t_steps} timeout steps, got {time_outs.shape[1]}")
        if not torch.all(time_outs <= dones + 1e-6):
            raise ValueError('QMIX RNN invariant violated: time_outs must be <= dones elementwise')

        if self.obs_normalizer is not None:
            flat_obs = TensorDict()
            obs_shapes = {}
            for key, val in obs.items():
                if isinstance(val, TensorDict):
                    raise ValueError('QMIX RNN normalization expects flat top-level obs dict')
                obs_shapes[key] = val.shape
                flat_obs[key] = val.reshape(-1, *val.shape[3:])
            norm_flat = self.obs_normalizer(flat_obs)
            obs = TensorDict({key: norm_flat[key].view(obs_shapes[key]) for key in obs_shapes})

        q_online_all, enc_online_all = self._sequential_agent_forward(obs, dones, rnn_states, self.agent_net)
        with torch.no_grad():
            q_target_all, enc_target_all = self._sequential_agent_forward(obs, dones, rnn_states, self.target_agent_net)
        q_online = q_online_all[:, :t_steps]
        q_online_next = q_online_all[:, 1:]
        q_target_next = q_target_all[:, 1:]
        is_compound_action = actions.dim() > 3

        def greedy_compound_actions(q_logits: Tensor) -> Tensor:
            action_heads = []
            offset = 0
            for head_size in self.agent_net.action_sizes:
                head_q = q_logits[..., offset:offset + head_size]
                action_heads.append(head_q.argmax(dim=-1))
                offset += head_size
            return torch.stack(action_heads, dim=-1)

        def evaluate_compound_q(q_logits: Tensor, head_actions: Tensor) -> Tensor:
            flat_q = q_logits.reshape(-1, q_logits.shape[-1])
            flat_actions = head_actions.reshape(-1, head_actions.shape[-1])
            flat_q_values = self.agent_net.get_q_for_actions(flat_q, flat_actions)
            return flat_q_values.view(q_logits.shape[0], q_logits.shape[1], q_logits.shape[2])

        # Double DQN
        if is_compound_action:
            agent_qs = evaluate_compound_q(q_online, actions)
            if getattr(self.cfg, 'double_dqn', True):
                best_actions = greedy_compound_actions(q_online_next)
                target_agent_qs = evaluate_compound_q(q_target_next, best_actions)
            else:
                greedy_target_actions = greedy_compound_actions(q_target_next)
                target_agent_qs = evaluate_compound_q(q_target_next, greedy_target_actions)
        else:
            agent_qs = q_online.gather(3, actions.unsqueeze(-1)).squeeze(-1)
            if getattr(self.cfg, 'double_dqn', True):
                best_actions = q_online_next.argmax(dim=-1, keepdim=True)
                target_agent_qs = q_target_next.gather(3, best_actions).squeeze(-1)
            else:
                target_agent_qs = q_target_next.max(dim=-1)[0]

        batch_size = actions.shape[0]
        enc_dim = enc_online_all.shape[-1]
        state = enc_online_all[:, :t_steps].detach().reshape(batch_size, t_steps, self.num_agents * enc_dim)
        next_state = enc_target_all[:, 1:].detach().reshape(batch_size, t_steps, self.num_agents * enc_dim)

        # Target Q_tot and Q_tot
        q_tot = self.mixer(
            agent_qs.reshape(batch_size * t_steps, self.num_agents),
            state.reshape(batch_size * t_steps, -1),
        ).view(batch_size, t_steps)
        with torch.no_grad():
            target_q_tot = self.target_mixer(
                target_agent_qs.reshape(batch_size * t_steps, self.num_agents),
                next_state.reshape(batch_size * t_steps, -1),
            ).view(batch_size, t_steps)

            team_reward = rewards.sum(dim=2)
            effective_done = (dones * (1.0 - time_outs)).amax(dim=2)
            # same as learner_dqn
            gamma = getattr(self.cfg, 'gamma', 0.99)
            target_before_clamp = team_reward + gamma * (1.0 - effective_done) * target_q_tot
            target = target_before_clamp
            # Clamp target Q val
            q_clamp = getattr(self.cfg, 'q_value_clamp', 100.0)
            if q_clamp > 0:
                target = target.clamp(-q_clamp, q_clamp)

        # This kind of becomes funny if not cloned, I think it got used in shared memory tensors
        # So clone to make sure its correct
        td_error = (target - q_tot).clone()

        # Loss
        use_huber = getattr(self.cfg, 'use_huber_loss', True)
        if use_huber:
            abs_td = td_error.abs()
            elementwise_loss = torch.where(abs_td < 1.0, 0.5 * td_error.pow(2), abs_td - 0.5)
        else:
            elementwise_loss = td_error.pow(2)

        # Weights
        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(-1)
            elementwise_loss = elementwise_loss * weights

        loss = elementwise_loss.mean()

        # loss metric from detached td_error
        _td_d = td_error.detach()
        _agent_qs_d = agent_qs.detach()
        _q_online_d = q_online.detach()
        self._last_loss_value = loss.detach().item()
        self._last_q_tot_mean = q_tot.detach().mean().item()
        self._last_q_tot_max = q_tot.detach().max().item()
        self._last_td_error_mean = _td_d.abs().mean().item()
        self._last_agent_qs_mean = _agent_qs_d.mean().item()
        self._last_agent_qs_max = _agent_qs_d.max().item()
        self._last_agent_qs_min = _agent_qs_d.min().item()
        self._last_target_q_tot_mean = target_q_tot.mean().item()
        self._last_target_before_clamp = target_before_clamp.mean().item()
        self._last_q_std_across_actions = _q_online_d.std(dim=-1).mean().item()
        self._last_done_ratio = effective_done.mean().item()
        self._last_timeout_ratio = time_outs.mean().item()

        td_per_sequence = _td_d.abs().mean(dim=1)
        return loss, td_per_sequence

    def _calculate_qmix_loss(self, batch: TensorDict, weights: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        obs = batch['obs'] # [B, N, ...]
        next_obs = batch['next_obs']
        actions = batch['actions'] # [B, N]/[B, N, heads]
        team_reward = batch['team_reward'] # [B]

        def normalize_joint_obs(joint_obs):
            if self.obs_normalizer is None:
                return joint_obs
            if isinstance(joint_obs, TensorDict):
                flat_obs = TensorDict()
                obs_shapes = {}
                for key, val in joint_obs.items():
                    if isinstance(val, TensorDict):
                        raise ValueError('QMIX normalization expects flat top-level obs dict')
                    obs_shapes[key] = val.shape
                    flat_obs[key] = val.flatten(end_dim=1)

                norm_flat = self.obs_normalizer(flat_obs)
                return TensorDict({key: norm_flat[key].view(obs_shapes[key]) for key in obs_shapes})
            else:
                original_shape = joint_obs.shape
                flat = joint_obs.flatten(end_dim=1)
                norm = self.obs_normalizer(flat)
                return norm.view(original_shape)

        obs = normalize_joint_obs(obs)
        next_obs = normalize_joint_obs(next_obs)

        # We recompute instead of storing in buffer
        # otherwise buffer's filled quite quickly -> timeout
        state = self._compute_global_state(obs)
        next_state = self._compute_global_state(next_obs, encoder_net=self.target_agent_net)

        all_q = self._vectorized_agent_forward(obs, self.agent_net)  # [B, N, A]

        # Get Q values
        # [B, N]
        actions = actions.long()
        if actions.dim() == 2:
            agent_qs = all_q.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        else:
            agent_qs = []
            for i in range(self.num_agents):
                agent_q = all_q[:, i, :] # [B, A]
                agent_action = actions[:, i, :] # [B, heads]
                q_val = self.agent_net.get_q_for_actions(agent_q, agent_action)
                agent_qs.append(q_val)
            agent_qs = torch.stack(agent_qs, dim=1)

        # Canonical effective_done: per-agent done*(1-timeout), then max across agents.
        # This correctly handles mixed timeout/terminal outcomes (e.g. one agent
        # terminates while another times out).
        per_agent_dones = batch['dones'].float()
        per_agent_timeouts = batch['time_outs'].float() if 'time_outs' in batch else torch.zeros_like(per_agent_dones)
        effective_done = (per_agent_dones * (1.0 - per_agent_timeouts)).amax(dim=-1)

        # Target Q_tot and Q_tot
        q_tot = self.mixer(agent_qs, state) # [B]
        with torch.no_grad():
            all_target_q = self._vectorized_agent_forward(next_obs, self.target_agent_net)
            is_compound_action = actions.dim() > 2

            def greedy_compound_actions(q_logits: Tensor) -> Tensor:
                action_heads = []
                offset = 0
                for head_size in self.agent_net.action_sizes:
                    head_q = q_logits[..., offset:offset + head_size]
                    action_heads.append(head_q.argmax(dim=-1))
                    offset += head_size
                return torch.stack(action_heads, dim=-1)

            def evaluate_compound_q(q_logits: Tensor, head_actions: Tensor) -> Tensor:
                q_values = []
                for agent_idx in range(self.num_agents):
                    q_val = self.agent_net.get_q_for_actions(
                        q_logits[:, agent_idx, :], head_actions[:, agent_idx, :]
                    )
                    q_values.append(q_val)
                return torch.stack(q_values, dim=1)

            # Double DQN
            if getattr(self.cfg, 'double_dqn', True):
                all_online_q = self._vectorized_agent_forward(next_obs, self.agent_net)
                if is_compound_action:
                    best_actions = greedy_compound_actions(all_online_q) # [B, N, H]
                    target_agent_qs = evaluate_compound_q(all_target_q, best_actions)
                else:
                    best_actions = all_online_q.argmax(dim=-1, keepdim=True) # [B, N, 1]
                    target_agent_qs = all_target_q.gather(2, best_actions).squeeze(-1)
            else:
                if is_compound_action:
                    greedy_target_actions = greedy_compound_actions(all_target_q)
                    target_agent_qs = evaluate_compound_q(all_target_q, greedy_target_actions)
                else:
                    target_agent_qs = all_target_q.max(dim=-1)[0] # [B, N]

            target_q_tot = self.target_mixer(target_agent_qs, next_state)

            # same as learner_dqn
            gamma = getattr(self.cfg, 'gamma', 0.99)
            target_before_clamp = team_reward + gamma * (1 - effective_done) * target_q_tot
            target = target_before_clamp

            # Clamp target Q val
            q_clamp = getattr(self.cfg, 'q_value_clamp', 100.0)
            if q_clamp > 0: target = target.clamp(-q_clamp, q_clamp)

        # This kind of becomes funny if not cloned, I think it got used in shared memory tensors
        # So clone to make sure its correct
        td_error = (target - q_tot).clone()

        # Loss
        use_huber = getattr(self.cfg, 'use_huber_loss', True)
        if use_huber:
            abs_td = td_error.abs()
            elementwise_loss = torch.where(abs_td < 1.0, 0.5 * td_error.pow(2), abs_td - 0.5)
        else:
            elementwise_loss = td_error.pow(2)

        # PER weights
        if weights is not None:
            elementwise_loss = elementwise_loss * weights

        loss = elementwise_loss.mean()

        # loss metric from detached td_error
        _td_d = td_error.detach()
        self._last_loss_value = loss.detach().item()
        self._last_q_tot_mean = q_tot.detach().mean().item()
        self._last_q_tot_max = q_tot.detach().max().item()
        self._last_td_error_mean = _td_d.abs().mean().item()
        self._last_agent_qs_mean = agent_qs.mean().item()
        self._last_agent_qs_max = agent_qs.max().item()
        self._last_agent_qs_min = agent_qs.min().item()
        self._last_target_q_tot_mean = target_q_tot.mean().item() if not effective_done.all() else 0.0
        self._last_target_before_clamp = target_before_clamp.mean().item()
        self._last_q_std_across_actions = all_q.std(dim=-1).mean().item() # std across actions should be > 0
        self._last_done_ratio = effective_done.mean().item()
        self._last_timeout_ratio = per_agent_timeouts.mean().item()

        return loss, td_error.abs().detach()

    def _train_on_batch(self, batch: TensorDict, weights: Optional[Tensor] = None, indices: Optional[Tensor] = None):
        self.agent_net.train()
        self.mixer.train()

        if self.use_rnn:
            loss, td_errors = self._calculate_qmix_loss_sequential(batch, weights)
        else:
            loss, td_errors = self._calculate_qmix_loss(batch, weights)
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.max_grad_norm > 0:
            params = list(self.agent_net.parameters()) + list(self.mixer.parameters())
            _gnorm_sq = sum(p.grad.detach().pow(2).sum().item() for p in params if p.grad is not None)
            self._last_grad_norm = _gnorm_sq ** 0.5
            torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
        with self.param_server.policy_lock:
            self.optimizer.step()
        self._after_optimizer_step()
        tau = getattr(self.cfg, 'target_update_tau', 1.0)
        self._update_target_networks(tau)
        synchronize(self.cfg, self.device)
        self.policy_versions_tensor[self.policy_id] = self.train_step

        # Stats
        stats = AttrDict()
        stats.loss = getattr(self, '_last_loss_value', to_scalar(loss))
        stats.lr = self.curr_lr
        stats.q_tot_mean = self._last_q_tot_mean
        stats.q_tot_max = self._last_q_tot_max
        stats.td_error_mean = self._last_td_error_mean
        stats.grad_norm = self._last_grad_norm
        stats.agent_qs_mean = getattr(self, '_last_agent_qs_mean', 0)
        stats.agent_qs_max = getattr(self, '_last_agent_qs_max', 0)
        stats.agent_qs_min = getattr(self, '_last_agent_qs_min', 0)
        stats.target_q_tot_mean = getattr(self, '_last_target_q_tot_mean', 0)
        stats.target_before_clamp = getattr(self, '_last_target_before_clamp', 0)
        stats.q_std_across_actions = getattr(self, '_last_q_std_across_actions', 0)
        stats.done_ratio = getattr(self, '_last_done_ratio', 0)
        stats.timeout_ratio = getattr(self, '_last_timeout_ratio', 0)

        return stats, td_errors

    def _prepare_joint_transitions(self, batch: TensorDict) -> TensorDict:
        with torch.no_grad():
            buff = shallow_recursive_copy(batch)
            num_traj = buff['rewards'].shape[0]
            row_index = None

            if 'env_idx' in buff and 'agent_idx' in buff:
                traj_env_idx = buff['env_idx'][:, 0].long().cpu()
                traj_agent_idx = buff['agent_idx'][:, 0].long().cpu()
                unique_envs = torch.unique(traj_env_idx)
                expected_agents = torch.arange(self.num_agents, dtype=torch.long)
                grouped_rows = []
                metadata_valid = True
                for env_id in unique_envs.tolist():
                    if env_id < 0:
                        metadata_valid = False
                        break
                    rows = torch.nonzero(traj_env_idx == env_id, as_tuple=False).squeeze(-1)
                    if rows.numel() < self.num_agents:
                        metadata_valid = False
                        break
                    rows = rows[torch.argsort(rows)]
                    agent_to_rows = []
                    occurrence_count = None
                    for agent_id in expected_agents.tolist():
                        agent_rows = rows[traj_agent_idx[rows] == agent_id]
                        if agent_rows.numel() == 0:
                            metadata_valid = False
                            break
                        if occurrence_count is None:
                            occurrence_count = agent_rows.numel()
                        elif occurrence_count != agent_rows.numel():
                            metadata_valid = False
                            break
                        agent_to_rows.append(agent_rows)
                    if not metadata_valid or occurrence_count is None:
                        break

                    for occ in range(occurrence_count):
                        group = torch.tensor(
                            [agent_to_rows[agent_id][occ].item() for agent_id in expected_agents.tolist()],
                            dtype=torch.long,
                        )
                        group_agents = traj_agent_idx[group]
                        if not torch.equal(group_agents, expected_agents):
                            metadata_valid = False
                            break
                        grouped_rows.append(group)
                    if not metadata_valid:
                        metadata_valid = False
                        break

                if metadata_valid and grouped_rows:
                    row_index = torch.stack(grouped_rows, dim=0).to(buff['rewards'].device)
                else:
                    log.warning("QMIX: invalid env_idx/agent_idx grouping")

            if row_index is None:
                num_envs = num_traj // self.num_agents
                if num_envs * self.num_agents != num_traj:
                    log.warning(f"num_traj={num_traj} not divisible by num_agents={self.num_agents}")
                    num_traj = num_envs * self.num_agents

            def reshape_for_joint(tensor: Tensor) -> Tensor:
                #[num_traj, T, ...] -> [num_joint, num_agents, ...]
                T = tensor.shape[1]
                rest = tensor.shape[2:]
                if row_index is not None:
                    x = tensor[row_index]# [num_envs, num_agents, T, ...]
                    perm = [0, 2, 1] + list(range(3, 3 + len(rest)))
                    x = x.permute(*perm) # [num_envs, T, num_agents, ...]
                    return x.reshape(x.shape[0] * T, self.num_agents, *rest) # [num_envs*T, num_agents, ...]
                x = tensor[:num_traj].view(num_envs, self.num_agents, T, *rest)
                perm = [0, 2, 1] + list(range(3, 3 + len(rest)))
                x = x.permute(*perm)
                return x.reshape(num_envs * T, self.num_agents, *rest)

            # T+1
            joint = TensorDict()
            joint['obs'] = TensorDict()
            joint['next_obs'] = TensorDict()
            for key, val in buff['obs'].items():
                current = val[:, :-1] # [num_traj, T, ...]
                next_val = val[:, 1:]# [num_traj, T, ...]
                joint['obs'][key] = reshape_for_joint(current)
                joint['next_obs'][key] = reshape_for_joint(next_val)

            # T steps
            joint['actions'] = reshape_for_joint(buff['actions'])
            joint['rewards'] = reshape_for_joint(buff['rewards'])
            joint['dones'] = reshape_for_joint(buff['dones']).float()
            if 'time_outs' in buff:
                joint['time_outs'] = reshape_for_joint(buff['time_outs']).float()
            else:
                joint['time_outs'] = torch.zeros_like(joint['dones'])

            return joint

    def _prepare_joint_sequences(self, batch: TensorDict) -> Optional[TensorDict]:
        with torch.no_grad():
            buff = shallow_recursive_copy(batch)
            num_traj = buff['rewards'].shape[0]
            row_index = None
            metadata_present = 'env_idx' in buff and 'agent_idx' in buff
            if metadata_present:
                env_idx_raw = buff['env_idx']
                agent_idx_raw = buff['agent_idx']
                traj_env_idx = env_idx_raw[:, 0].long().cpu() if env_idx_raw.dim() > 1 else env_idx_raw.long().cpu()
                traj_agent_idx = (agent_idx_raw[:, 0].long().cpu() if agent_idx_raw.dim() > 1 else agent_idx_raw.long().cpu())

                expected_agents = torch.arange(self.num_agents, dtype=torch.long)
                grouped_rows = []
                invalid_env_groups = 0
                for env_id in torch.unique(traj_env_idx).tolist():
                    if env_id < 0:
                        invalid_env_groups += 1
                        continue

                    rows = torch.nonzero(traj_env_idx == env_id, as_tuple=False).squeeze(-1)
                    rows = rows[torch.argsort(rows)]
                    if rows.numel() < self.num_agents:
                        invalid_env_groups += 1
                        continue

                    agent_to_rows = []
                    occurrence_count = None
                    env_valid = True
                    for agent_id in expected_agents.tolist():
                        agent_rows = rows[traj_agent_idx[rows] == agent_id]
                        if agent_rows.numel() == 0:
                            env_valid = False
                            break
                        if occurrence_count is None:
                            occurrence_count = agent_rows.numel()
                        elif occurrence_count != agent_rows.numel():
                            env_valid = False
                            break
                        agent_to_rows.append(agent_rows)

                    if not env_valid or occurrence_count is None:
                        invalid_env_groups += 1
                        continue

                    for occ in range(occurrence_count):
                        group = torch.tensor(
                            [agent_to_rows[agent_id][occ].item() for agent_id in expected_agents.tolist()],
                            dtype=torch.long,
                        )
                        if not torch.equal(traj_agent_idx[group], expected_agents):
                            env_valid = False
                            break
                        grouped_rows.append(group)

                    if not env_valid:
                        invalid_env_groups += 1

                self._invalid_sequence_groups += invalid_env_groups
                if invalid_env_groups > 0:
                    log.warning(f"QMIX RNN: dropped {invalid_env_groups} invalid env_idx/agent_idx groups")
                if len(grouped_rows) == 0:
                    log.warning("QMIX RNN: no valid env_idx/agent_idx groups. Skip learner update")
                    return None
                row_index = torch.stack(grouped_rows, dim=0).to(buff['rewards'].device)

            num_envs = None
            if row_index is None:
                log.debug("QMIX RNN: no env_idx/agent_idx metadata. Group by index")
                num_envs = num_traj // self.num_agents
                if num_envs * self.num_agents != num_traj:
                    log.warning(f"QMIX RNN: num_traj={num_traj} not divisible by num_agents={self.num_agents}")
                    num_traj = num_envs * self.num_agents
                if num_envs == 0: return None

            def reshape_for_seq(tensor: Tensor, expected_t: int) -> Tensor:
                rest = tensor.shape[2:]
                if tensor.shape[1] != expected_t:
                    raise ValueError(f"Expected time dim={expected_t}, got {tensor.shape[1]}")
                if row_index is not None:
                    x = tensor[row_index]
                else:
                    x = tensor[:num_traj].view(num_envs, self.num_agents, expected_t, *rest)
                perm = [0, 2, 1] + list(range(3, 3 + len(rest)))
                return x.permute(*perm).contiguous()

            joint = TensorDict()
            joint['obs'] = TensorDict()

            def reshape_obsr(src: TensorDict, dst: TensorDict):
                for key, val in src.items():
                    if isinstance(val, TensorDict):
                        dst[key] = TensorDict()
                        reshape_obsr(val, dst[key])
                    else:
                        dst[key] = reshape_for_seq(val, self.cfg.rollout + 1)
            reshape_obsr(buff['obs'], joint['obs'])

            joint['actions'] = reshape_for_seq(buff['actions'], self.cfg.rollout).long()
            joint['rewards'] = reshape_for_seq(buff['rewards'], self.cfg.rollout).float()
            joint['dones'] = reshape_for_seq(buff['dones'], self.cfg.rollout).float()
            if 'time_outs' in buff:
                joint['time_outs'] = reshape_for_seq(buff['time_outs'], self.cfg.rollout).float()
            else:
                joint['time_outs'] = torch.zeros_like(joint['dones'])

            if 'rnn_states' not in buff:
                raise ValueError('QMIX RNN requires rnn_states in learner batch')

            rnn_seq = buff['rnn_states']
            if rnn_seq.dim() != 3:
                raise ValueError(f"Unexpected rnn_states shape: {tuple(rnn_seq.shape)}")
            if rnn_seq.shape[1] != self.cfg.rollout + 1:
                raise ValueError(f"Expected rnn_states time dim={self.cfg.rollout + 1}, got {rnn_seq.shape[1]}")
            if rnn_seq.shape[2] != self.agent_net.get_rnn_size():
                raise ValueError(f"Expected rnn_states feature dim={self.agent_net.get_rnn_size()}, got {rnn_seq.shape[2]}")
            rnn_init = rnn_seq[:, 0, :]

            if row_index is not None:
                joint['rnn_states'] = rnn_init[row_index].contiguous()
            else:
                joint['rnn_states'] = rnn_init[:num_traj].view(num_envs, self.num_agents, -1).contiguous()
            return joint

    def _qmix_batch_size(self) -> int:
        if self.use_rnn:
            batch_size = int(getattr(self.cfg, 'qmix_sequence_batch_size', 8))
        else:
            batch_size = int(getattr(self.cfg, 'qmix_buffer_batch_size', 32))
        if batch_size <= 0:
            raise ValueError(f'QMIX batch size must be > 0, got {batch_size}')
        return batch_size

    def train(self, batch: TensorDict) -> Optional[Dict]:
        with self.timing.add_time('misc'):
            self._maybe_update_cfg()
            self._maybe_load_policy()

        # Tracking for GRU
        transitions_added = 0
        sequence_batch = None
        joint_transitions = None
        with self.timing.add_time('prepare_batch'):
            raw_num_traj = int(batch['rewards'].shape[0])
            raw_t = int(batch['rewards'].shape[1])
            num_agent_transitions = raw_num_traj * raw_t
            if self.cfg.summaries_use_frameskip:
                self.env_steps += num_agent_transitions * self.env_info.frameskip
            else:
                self.env_steps += num_agent_transitions
            if self.global_env_steps_tensor is not None:
                self.global_env_steps_tensor[self.policy_id] = self.env_steps

            if self.use_rnn:
                sequence_batch = self._prepare_joint_sequences(batch)
            else:
                joint_transitions = self._prepare_joint_transitions(batch)

        with self.timing.add_time('add_to_buffer'):
            if self.use_rnn:
                if sequence_batch is not None and sequence_batch['rewards'].shape[0] > 0:
                    num_sequences = self.replay_buffer.add_sequence_batch(sequence_batch)
                    transitions_added = num_sequences * self.num_agents * self.cfg.rollout
                    log.debug(f"QMIX RNN replay insert: sequences={num_sequences}, transitions={transitions_added}")
            else:
                num_joint = self.replay_buffer.add_joint_batch(joint_transitions)
                transitions_added = num_joint * self.num_agents  # each agent transitions
                # This leads to twice as much updates tho, which is fine

        if self.use_rnn:
            buffer_transitions = len(self.replay_buffer) * self.num_agents * self.cfg.rollout
        else:
            buffer_transitions = len(self.replay_buffer) * self.num_agents

        learning_starts = int(getattr(self.cfg, 'learning_starts', 5000))
        if buffer_transitions < learning_starts:
            return {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}

        self.total_env_steps_for_training += transitions_added
        train_freq = int(getattr(self.cfg, 'train_frequency', 4))
        if train_freq <= 0:
            log.warning('QMIX: train_frequency <= 0, forcing to 1')
            train_freq = 1
        num_updates = self.total_env_steps_for_training // train_freq
        if num_updates == 0:
            return {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}

        max_updates = getattr(self.cfg, 'dqn_max_updates_per_batch', 4)
        if max_updates > 0:
            num_updates = min(num_updates, max_updates)

        # Minus after capping to trainig upds dont get removed
        self.total_env_steps_for_training -= num_updates * train_freq

        train_stats = None
        with self.timing.add_time('train'):
            for _ in range(num_updates):
                # Anneal PER beta
                if self.use_per:
                    if self.per_beta_frames <= 0:
                        beta = 1.0
                    else:
                        progress = min(1.0, self.env_steps / self.per_beta_frames)
                        beta = self.per_beta_start + progress * (1 - self.per_beta_start)
                    self.replay_buffer.set_beta(beta)

                # Sample
                try:
                    sampled, weights, indices = self.replay_buffer.sample(self._qmix_batch_size(), str(self.device))
                except RuntimeError as e:
                    log.warning(f"Sampling failed: {e}")
                    continue

                # Train
                train_stats, td_errors = self._train_on_batch(sampled, weights, indices)

                # Update PER
                if self.use_per and indices is not None and td_errors is not None:
                    self.replay_buffer.update_priorities(indices, td_errors)
        self.last_train_env_steps = self.env_steps

        stats = {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
        if train_stats is not None:
            train_stats.env_steps = self.env_steps
            train_stats.num_updates = num_updates
            train_stats.buffer_size = len(self.replay_buffer)

            # Debug
            log_interval = getattr(self.cfg, 'qmix_log_interval', 100)
            if self.train_step % log_interval == 0:
                log.info(f"QMIX step={self.train_step}, loss={train_stats.loss:.4f}, q_tot={train_stats.q_tot_mean:.3f}, td_err={train_stats.td_error_mean:.3f}, grad={train_stats.grad_norm:.4f}, buf={train_stats.buffer_size}")
                log.info(f"+ agent_qs: mean={train_stats.agent_qs_mean:.3f}, max={train_stats.agent_qs_max:.3f}, min={train_stats.agent_qs_min:.3f}")
                log.info(f"+ target: q_tot={train_stats.target_q_tot_mean:.3f}, before_clamp={train_stats.target_before_clamp:.3f}")
                log.info(f"+ q_std={train_stats.q_std_across_actions:.4f}, done={train_stats.done_ratio:.3f}, timeout={train_stats.timeout_ratio:.3f}")
            stats[TRAIN_STATS] = train_stats
            stats[STATS_KEY] = memory_stats(f'learner{self.policy_id}', self.device)
        return stats

    def _get_checkpoint_dict(self) -> Dict:
        checkpoint = super()._get_checkpoint_dict()

        if self.target_agent_net is not None:
            checkpoint['target_agent_net'] = self.target_agent_net.state_dict()
        if self.target_mixer is not None:
            checkpoint['target_mixer'] = self.target_mixer.state_dict()

        checkpoint['last_train_env_steps'] = self.last_train_env_steps
        checkpoint['last_target_update_step'] = self.last_target_update_step
        if self.replay_buffer is not None:
            checkpoint['replay_buffer_size'] = len(self.replay_buffer)
        return checkpoint

    def _load_state(self, checkpoint_dict: Dict, load_progress: bool = True) -> None:
        super()._load_state(checkpoint_dict, load_progress)

        if 'target_agent_net' in checkpoint_dict and self.target_agent_net is not None:
            self.target_agent_net.load_state_dict(checkpoint_dict['target_agent_net'])
            log.info("Loaded target_agent_net from checkpoint")
        if 'target_mixer' in checkpoint_dict and self.target_mixer is not None:
            self.target_mixer.load_state_dict(checkpoint_dict['target_mixer'])
            log.info("Loaded target_mixer from checkpoint")
        if load_progress:
            if 'last_train_env_steps' in checkpoint_dict:
                self.last_train_env_steps = checkpoint_dict['last_train_env_steps']
            if 'last_target_update_step' in checkpoint_dict:
                self.last_target_update_step = checkpoint_dict['last_target_update_step']
