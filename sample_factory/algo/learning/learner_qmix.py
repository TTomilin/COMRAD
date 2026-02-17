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
from sample_factory.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_factory.algo.utils.torch_utils import synchronize, to_scalar
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
        
        self.state_dim = self._calc_state_dim()
        log.info(f"QMIX state_dim={self.state_dim}")
        q_clamp = getattr(self.cfg, 'q_value_clamp', 100.0)
        tau = getattr(self.cfg, 'target_update_tau', 1.0)
        use_huber = getattr(self.cfg, 'use_huber_loss', True)
        log.info(f"QMIX: q_clamp={q_clamp}, target_tau={tau}, use_huber={use_huber}")
        
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
    
    def _compute_global_state(self, obs: TensorDict) -> Tensor:
        if isinstance(obs, dict):
            sample_tensor = next(iter(obs.values()))
        else:
            sample_tensor = obs
        batch_size = sample_tensor.shape[0]
        num_agents = sample_tensor.shape[1]
        
        def flatten_dct(d):
            if isinstance(d, dict):
                return {k: v.flatten(end_dim=1) for k, v in d.items()}
            return d.flatten(end_dim=1)
        
        flat_obs = flatten_dct(obs) # [B, N, ...] -> [B*N, ...]
        with torch.no_grad(): # Encode through agent net
            encoded = self.agent_net.encode(flat_obs) # [B*N, encoder_out]
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
        batch_size = next(iter(obs.values())).shape[0] if isinstance(obs, dict) else obs.shape[0]
        
        def flatten_td(td):
            result = TensorDict()
            for key, val in td.items():
                if isinstance(val, TensorDict): result[key] = flatten_td(val)
                else: result[key] = val.flatten(end_dim=1)
            return result
        flat_obs = flatten_td(obs)
        
        # Dummy state
        num_flat = batch_size * self.num_agents
        rnn_states = torch.zeros(
            num_flat, 
            agent_net.core.get_out_size(),
            device=self.device
        )
        q_values, _ = agent_net(flat_obs, rnn_states=rnn_states) # [B*N, num_actions]
        num_actions = q_values.shape[-1]
        q_values = q_values.view(batch_size, self.num_agents, num_actions) # [B*N, A] -> [B, N, A]
        
        return q_values
    
    def _calculate_qmix_loss(self, batch: TensorDict, weights: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        obs = batch['obs'] # [B, N, ...]
        next_obs = batch['next_obs']
        actions = batch['actions'] # [B, N]/[B, N, heads]
        team_reward = batch['team_reward'] # [B]
        joint_done = batch['joint_done'] # [B]
        joint_time_out = batch['joint_time_out'] if 'joint_time_out' in batch else None
        
        def normalize_joint_obs(joint_obs):
            if self.obs_normalizer is None: return joint_obs
            
            if isinstance(joint_obs, TensorDict):
                normalized = TensorDict()
                for key, val in joint_obs.items():
                    if isinstance(val, TensorDict): normalized[key] = normalize_joint_obs(val)
                    else:
                        # [B, N, ...] -> [B*N, ...]
                        original_shape = val.shape
                        flat = val.flatten(end_dim=1)
                        
                        # Normalize
                        temp_td = TensorDict({key: flat})
                        norm_td = self.obs_normalizer(temp_td)

                        # [B*N, ...] -> [B, N, ...]
                        norm_val = norm_td[key]
                        normalized[key] = norm_val.view(original_shape)
                return normalized
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
        next_state = self._compute_global_state(next_obs)
        
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
        
        if joint_time_out is not None:
            effective_done = joint_done * (1 - joint_time_out)
        else:
            effective_done = joint_done
        
        # Target Q_tot
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
        _td_sq = _td_d.pow(2)
        self._last_loss_value = 0.5 * _td_sq.mean().item() # Huber approx for |td|<1
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
        if joint_time_out is not None:
            self._last_timeout_ratio = joint_time_out.mean().item()
        else:
            self._last_timeout_ratio = 0.0
        
        return loss, td_error.abs().detach()
    
    def _train_on_batch(self, batch: TensorDict, weights: Optional[Tensor] = None, indices: Optional[Tensor] = None):
        self.agent_net.train()
        self.mixer.train()
        
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
            joint['dones'] = joint['dones'] * (1 - joint['time_outs'])
            
            return joint
    
    def _qmix_batch_size(self) -> int:
        return getattr(self.cfg, 'qmix_buffer_batch_size', 32)
    
    def train(self, batch: TensorDict) -> Optional[Dict]:
        with self.timing.add_time('misc'):
            self._maybe_update_cfg()
            self._maybe_load_policy()
        
        with self.timing.add_time('prepare_batch'):
            joint_transitions = self._prepare_joint_transitions(batch)
            num_joint = joint_transitions['rewards'].shape[0]
            num_agent_transitions = num_joint * self.num_agents
            if self.cfg.summaries_use_frameskip:
                self.env_steps += num_agent_transitions * self.env_info.frameskip
            else:
                self.env_steps += num_agent_transitions
            if self.global_env_steps_tensor is not None:
                self.global_env_steps_tensor[self.policy_id] = self.env_steps
        
        with self.timing.add_time('add_to_buffer'):
            self.replay_buffer.add_joint_batch(joint_transitions)
        
        learning_starts = getattr(self.cfg, 'learning_starts', 5000) // self.num_agents
        if len(self.replay_buffer) < learning_starts:
            return {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
        
        self.total_env_steps_for_training += num_joint
        train_freq = getattr(self.cfg, 'train_frequency', 4)
        num_updates = self.total_env_steps_for_training // train_freq
        if num_updates == 0: return {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
        self.total_env_steps_for_training -= num_updates * train_freq
        
        max_updates = getattr(self.cfg, 'dqn_max_updates_per_batch', 4)
        if max_updates > 0:
            num_updates = min(num_updates, max_updates)
        
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
