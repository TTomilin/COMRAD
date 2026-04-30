"""
happo/factor_m_mean            # Mean M after all agent updates
happo/factor_m_max             # Max M
happo/agent_order              # of current iteration
happo/agent_i_policy_loss      # Policy losses
happo/agent_i_grad_norm        # grad norms each agents
happo/agent_i_post_ratio_mean  # Mean post-update ratio, should be near 1
happo/critic_grad_norm
value_loss                     # Critic value loss
lr
version_diff
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.learning.learner import Learner, model_initialization_data, get_lr_scheduler
from sample_factory.algo.learning.rnn_utils import build_rnn_inputs, build_core_out_from_seq
from sample_factory.algo.utils.action_distributions import get_action_distribution, is_continuous_action_space
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.algo.utils.torch_utils import synchronize, to_scalar
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import InitModelData
from sample_factory.utils.utils import log

from comrad.models.happo_model import _group_by_env


class HAPPOLearner(Learner):
    """
    Loop:
    1. Prepare batch, GAE advantages
    2. Init M = advantages
    3. For each agent in random order (can be set to fixed order as well):
        1. snapshot old log probs
        2. train agent for K epochs
            for epoch: for minibatch: agent_loss(M).backward(); agent_optimizer.step()
        3. eval new log probs
        4. M *= unclipped post-update ratio
    4. Train centralized critic separately
        for epoch: for minibatch: critic_loss.backward(); agent_optimizer.step()
    """

    def __init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server):
        super().__init__(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        self.n_agents = cfg.num_agents
        self._warned_on_non_shared_rewards = False

    def init(self) -> InitModelData:
        # Note: Cant use super().init(), everything needs to be updated
        # so many codes are mostly copied entirely from Learner

        # loss func
        if self.cfg.exploration_loss_coeff == 0.0:
            self.exploration_loss_func = lambda action_distr, valids, num_invalids: 0.0
        elif self.cfg.exploration_loss == "entropy":
            self.exploration_loss_func = self._entropy_exploration_loss
        elif self.cfg.exploration_loss == "symmetric_kl":
            self.exploration_loss_func = self._symmetric_kl_exploration_loss
        else:
            raise NotImplementedError(f"{self.cfg.exploration_loss} not supported!")

        if self.cfg.kl_loss_coeff == 0.0:
            if is_continuous_action_space(self.env_info.action_space):
                # For continuous action tasks, the return is normal dist
                # so continuous probability densities can exceed 1 if dist is narrow -> M will explode
                log.warning("HAPPO: You should enable Fixed KL loss (set --kl_loss_coeff=0.1 for example)")
            self.kl_loss_func = lambda action_space, action_logits, distribution, valids, num_invalids: (None, 0.0)
        else:
            self.kl_loss_func = self._kl_loss

        # seed
        if self.cfg.seed is None:
            log.info("Starting seed is not provided")
        else:
            # seed separately since didnt call super().init()
            log.info("Setting fixed seed %d", self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # initialize device
        self.device = policy_device(self.cfg, self.policy_id)

        # model
        log.debug("Initializing HAPPO actor-critic model on device %s", self.device)
         # trainable torch module
        self.actor_critic = create_actor_critic(self.cfg, self.env_info.obs_space, self.env_info.action_space)
        log.debug("Created Actor Critic model with architecture:")
        log.debug(self.actor_critic)
        self.actor_critic.model_to_device(self.device)

        # Shared mem for inference workers
        def share_mem(t):
            if t is not None and not t.is_cuda:
                return t.share_memory_()
            return t

        # noinspection PyProtectedMember
        self.actor_critic._apply(share_mem)
        self.actor_critic.train()

        # Critic optimizers
        self.agent_optimizers = []
        for i in range(self.n_agents):
            agent_params = self._get_agent_params(i)
            opt = torch.optim.Adam(
                agent_params,
                lr=self.cfg.learning_rate,
                betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
                eps=self.cfg.adam_eps,
            )
            self.agent_optimizers.append(opt)

        critic_params = self._get_critic_params()
        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=self.cfg.learning_rate,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            eps=self.cfg.adam_eps,
        )

        # checkpoint
        self.load_from_checkpoint(self.policy_id)

        # Param server init
        self.param_server.init(self.actor_critic, self.train_step, self.device)
        self.policy_versions_tensor[self.policy_id] = self.train_step

        # lr scheduler
        self.lr_scheduler = get_lr_scheduler(self.cfg)
        self.curr_lr = self.cfg.learning_rate if self.curr_lr is None else self.curr_lr
        self._apply_lr(self.curr_lr)

        self.is_initialized = True
        return model_initialization_data(self.cfg, self.policy_id, self.actor_critic, self.train_step, self.device)


    # =================================
    # Overriding from Learner

    def _get_checkpoint_dict(self):
        """save per agent, critic optimizer states"""
        checkpoint = {
            "model": self.actor_critic.state_dict(),
            "agent_optimizers": [opt.state_dict() for opt in self.agent_optimizers],
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "env_steps": self.env_steps,
            "train_step": self.train_step,
            "curr_lr": self.curr_lr,
            "best_performance": self.best_performance,
        }
        if self.curriculum is not None:
            checkpoint["curriculum_state"] = self.curriculum.state_dict()
        return checkpoint

    def _load_state(self, checkpoint_dict, load_progress=True):
        """restore per agent,critic optimizer states"""
        try:
            self.actor_critic.load_state_dict(checkpoint_dict["model"])
        except RuntimeError as e:
            if "critic_cores" in str(e) or "critic_projection" in str(e) or "centralized_critic" in str(e):
                log.warning("Checkpoint has different happo_critic_rnn setting. Loading with strict=False, critic layers will reinitialize.")
                self.actor_critic.load_state_dict(checkpoint_dict["model"], strict=False)
            else:
                raise
        if "agent_optimizers" in checkpoint_dict:
            for i, opt_state in enumerate(checkpoint_dict["agent_optimizers"]):
                self.agent_optimizers[i].load_state_dict(opt_state)
        if "critic_optimizer" in checkpoint_dict:
            try:
                self.critic_optimizer.load_state_dict(checkpoint_dict["critic_optimizer"])
            except ValueError:
                log.warning("Critic optimizer state mismatch") # could be happo_critic_rnn
        if load_progress:
            self.train_step = checkpoint_dict.get("train_step", 0)
            self.env_steps = checkpoint_dict.get("env_steps", 0)
            self.best_performance = checkpoint_dict.get("best_performance", -1e9)
        if "curr_lr" in checkpoint_dict:
            self.curr_lr = checkpoint_dict["curr_lr"]
        self._plr_partial_scores = {}
        log.info(f"Loaded HAPPO experiment state at {self.train_step=}, {self.env_steps=}")

    def _optimizer_lr(self):
        """return LR from first agent optimizer."""
        return self.agent_optimizers[0].param_groups[0]["lr"]

    def _apply_lr(self, lr):
        """Change learning rate in the optimizer."""
        for opt in self.agent_optimizers:
            for param_group in opt.param_groups:
                param_group["lr"] = lr
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = lr

    def _maybe_update_cfg(self):
        if self.new_cfg is not None:
            for key, val in self.new_cfg.items():
                if self.cfg[key] != val:
                    log.debug("Learner %d replacing cfg parameter %r with new value %r", self.policy_id, key, val)
                    self.cfg[key] = val

            if self.cfg.lr_schedule == "constant" and self.curr_lr != self.cfg.learning_rate:
                # PBT-optimized learning rate, only makes sense if we use constant LR
                # in case of more advanced LR scheduling we should update the parameters of the scheduler, not the
                # learning rate directly
                log.debug(f"Updating LR from {self.curr_lr} to {self.cfg.learning_rate}")
                self.curr_lr = self.cfg.learning_rate
                self._apply_lr(self.curr_lr)

            for opt in self.agent_optimizers + [self.critic_optimizer]:
                for param_group in opt.param_groups:
                    param_group["betas"] = (self.cfg.adam_beta1, self.cfg.adam_beta2)

            self.new_cfg = None

    def _record_summaries(self, train_stats):
        self.last_summary_time = time.time()
        stats = train_stats
        stats["lr"] = self.curr_lr
        stats["env_steps"] = self.env_steps
        # version_diff computed at start of _train() with access to gpu_buffer

        if hasattr(self.actor_critic, "summaries"):
            actor_critic_stats = self.actor_critic.summaries()
            stats.update(actor_critic_stats)

        for key, value in stats.items():
            stats[key] = to_scalar(value)

        return stats

    # =======================================


    def _prepare_batch(self, batch):
        """
        Use Learner's _prepare_batch() to compute advatnage GAE
        """
        buff, dataset_size, num_invalids = super()._prepare_batch(batch)

        # check env_idx
        assert "env_idx" in buff, "HAPPO requires env_idx in buffer (batched_sampling=True)"
        assert buff["env_idx"].dim() == 1, (f"env_idx should be 1D after _prepare_batch, got dim={buff['env_idx'].dim()}")
        assert buff["env_idx"].shape[0] == dataset_size, (f"env_idx size {buff['env_idx'].shape[0]} != dataset_size {dataset_size}")
        assert (buff["env_idx"] >= 0).all(), "HAPPO: env_idx contains negative values"

        self._warn_if_rewards_not_shared(buff, dataset_size)

        return buff, dataset_size, num_invalids

    # ===========================================================================

    def _warn_if_rewards_not_shared(self, buff, dataset_size):
        '''
        This is here for when shared reward is enabled as HAPPO assumes agents in the same transition saw the same scalar reward
        With alpha=0.5 and two agents getting r_0=10, r_1=0, the blended rewards are r_0=7.5, r_1=2.5, this breaks the assumption
        as GAE will compute diff returns for agents

        The original HAPPO paper assumes single shared team reward signal (so alpha = 1.0). However, alpha < 1 is fine to mix in individual rewards, thus here's just a warning
        '''
        if self._warned_on_non_shared_rewards:
            return

        rewards = buff.get("rewards")
        normalized_obs = buff.get("normalized_obs")
        if rewards is None or rewards.dim() != 1 or normalized_obs is None:
            return
        if "agent_id" not in normalized_obs or "env_idx" not in buff:
            return

        agent_idx = normalized_obs["agent_id"].argmax(dim=-1).long()
        env_idx = buff["env_idx"].long()
        group_idx = self._compute_env_group_idx(agent_idx, env_idx, dataset_size)
        n_transitions = group_idx.max().item() + 1

        reward_sum = torch.zeros(n_transitions, dtype=rewards.dtype, device=rewards.device)
        reward_sq_sum = torch.zeros_like(reward_sum)
        reward_count = torch.zeros_like(reward_sum)

        reward_sum.index_add_(0, group_idx, rewards)
        reward_sq_sum.index_add_(0, group_idx, rewards * rewards)
        reward_count.index_add_(0, group_idx, torch.ones_like(rewards))

        reward_mean = reward_sum / reward_count.clamp_min(1.0)
        reward_var = reward_sq_sum / reward_count.clamp_min(1.0) - reward_mean.square()
        reward_var.clamp_(min=0.0)

        if reward_var.max().item() <= 1e-6:
            return

        max_abs_diff = (rewards - reward_mean[group_idx]).abs().max().item()
        log.warning(f"HAPPO assumes joint team rewards, but this batch contains per-agent reward differences within the same transition (max abs diff {max_abs_diff})")
        self._warned_on_non_shared_rewards = True

    # ==========================================================================

    def _train(self, gpu_buffer, batch_size, experience_size, num_invalids):
        stats = AttrDict()
        assert self.actor_critic.training

        self._compute_plr_task_scores(gpu_buffer, experience_size)

        # Check if data stale from previous training
        if "policy_version" in gpu_buffer:
            curr_version = self.train_step
            version_diff = curr_version - gpu_buffer["policy_version"]
            stats["version_diff_avg"] = version_diff.float().mean().item()
            stats["version_diff_max"] = version_diff.max().item()

        # get agent_idx and env_group_idx
        agent_idx = gpu_buffer["normalized_obs"]["agent_id"].argmax(dim=-1).long()
        env_idx = gpu_buffer["env_idx"].long()
        env_group_idx = self._compute_env_group_idx(agent_idx, env_idx, experience_size)
        n_transitions = env_group_idx.max().item() + 1

        # ccalcutate advnatage
        adv = gpu_buffer["advantages"].clone()
        valids = gpu_buffer["valids"]
        with torch.no_grad():
            valid_mask = valids.bool()
            valid_adv = adv[valid_mask]
            adv_mean = valid_adv.mean()
            adv_std = valid_adv.std()
            adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)

        # M
        M = adv.clone().detach()

        # Either random order or fixed
        if self.cfg.happo_agent_order == "random":
            rng = torch.Generator().manual_seed(self.train_step)
            agent_order = torch.randperm(self.n_agents, generator=rng).tolist()
        else:
            agent_order = list(range(self.n_agents))

        actual_lr = self.curr_lr
        if num_invalids > 0:
            # if we have masked (invalid) data we should reduce the learning rate accordingly
            # this prevents a situation where most of the data in the minibatch is invalid
            # and we end up doing SGD with super noisy gradients
            actual_lr = self.curr_lr * (experience_size - num_invalids) / experience_size
        self._apply_lr(actual_lr)

        # Training loop sequeantial actor
        for agent_id in agent_order:
            agent_mask = (agent_idx == agent_id)

            # snapshot old log probs
            with torch.no_grad():
                old_log_probs = self._evaluate_agent_log_probs(agent_id, gpu_buffer, agent_mask)

            # Train agent_id for K epochs
            policy_loss_val = 0.0
            grad_norm_val = 0.0
            for epoch in range(self.cfg.num_epochs):
                minibatches = self._get_agent_minibatches(batch_size, agent_mask, experience_size)

                for indices in minibatches:
                    mb = AttrDict(self._get_minibatch(gpu_buffer, indices))

                    # Note: forward pass only for this agent
                    policy_loss, exploration_loss = self._calculate_agent_policy_loss(agent_id, mb, M, mb_indices=indices, num_invalids=num_invalids)

                    # We dont add kl_loss like in Learner because lr scheduler dont allow KL div
                    # This is also explained in arguments that KL div doesnt make sense for HAPPO
                    # Note that we dont do loss: Tensor = actor_loss + critic_loss here
                    # We update weights for all actors, loop sequentially everything first
                    # then critic centralized later with value_loss
                    actor_loss = policy_loss + exploration_loss

                    self.agent_optimizers[agent_id].zero_grad()
                    actor_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self._get_agent_params(agent_id), self.cfg.max_grad_norm)

                    with self.param_server.policy_lock:
                        self.agent_optimizers[agent_id].step()

                    self.train_step += 1
                    policy_loss_val = policy_loss.item()
                    grad_norm_val = grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm

            # get post-update
            with torch.no_grad():
                new_log_probs = self._evaluate_agent_log_probs(agent_id, gpu_buffer, agent_mask)
                post_ratio = torch.exp(new_log_probs - old_log_probs)

                # Mask invalid samples
                agent_valids = gpu_buffer["valids"][agent_mask].bool()
                post_ratio[~agent_valids] = 1.0

                factor_clamp = getattr(self.cfg, "happo_factor_clamp", 0.0)
                if factor_clamp > 0:
                    post_ratio = torch.clamp(post_ratio, 1.0 / factor_clamp, factor_clamp)

                # Broadcast agent i's post-ratio to all agents in same transitions
                ratio_per_transition = torch.ones(n_transitions, device=M.device)
                ratio_per_transition[env_group_idx[agent_mask]] = post_ratio

                M = M * ratio_per_transition[env_group_idx]

            stats[f"happo/agent_{agent_id}_policy_loss"] = policy_loss_val
            stats[f"happo/agent_{agent_id}_grad_norm"] = grad_norm_val
            stats[f"happo/agent_{agent_id}_post_ratio_mean"] = post_ratio.mean().item()

            # Sync weights to inference workers after each agent completes
            # If we do this out of the loop agents will block for asynchrous
            synchronize(self.cfg, self.device)
            self.policy_versions_tensor[self.policy_id] = self.train_step

        # training critic
        critic_grad_norm_val = 0.0
        value_loss_val = 0.0
        use_critic_rnn = self.actor_critic.use_critic_rnn
        for epoch in range(self.cfg.num_epochs):
            if use_critic_rnn:
                transition_minibatches = self._get_transition_rnn_minibatches(batch_size, experience_size, env_group_idx, n_transitions, env_idx, agent_idx)
            else:
                transition_minibatches = self._get_transition_minibatches(batch_size, experience_size, env_group_idx, n_transitions)
            for indices in transition_minibatches:
                mb = AttrDict(self._get_minibatch(gpu_buffer, indices))
                mb_agent_idx = agent_idx[indices]
                mb_env_group_idx = env_group_idx[indices]
                # Remap to dense local indices to avoid excessive allocation
                _, mb_env_group_local = torch.unique(mb_env_group_idx, return_inverse=True)
                value_loss = self._calculate_critic_loss(mb, mb_agent_idx, mb_env_group_local, num_invalids)

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._get_critic_params(),
                    self.cfg.max_grad_norm
                )

                with self.param_server.policy_lock:
                    self.critic_optimizer.step()
                self.train_step += 1
                value_loss_val = value_loss.item()
                critic_grad_norm_val = (critic_grad_norm.item() if isinstance(critic_grad_norm, Tensor) else critic_grad_norm)

        stats["happo/factor_m_mean"] = M.abs().mean().item()
        stats["happo/factor_m_max"] = M.abs().max().item()
        stats["happo/agent_order_first"] = agent_order[0]
        stats["happo/critic_grad_norm"] = critic_grad_norm_val
        stats["value_loss"] = value_loss_val
        stats["adv_mean"] = adv_mean.item()
        stats["adv_std"] = adv_std.item()

        # LR scheduling
        # update num_epochs * num_batches_per_epoch times per training iteration to match Learner.LinearDecayScheduler
        num_batches_per_epoch = max(1, experience_size // batch_size)
        total_lr_steps = self.cfg.num_epochs * num_batches_per_epoch
        for _ in range(total_lr_steps):
            self.curr_lr = self.lr_scheduler.update(self.curr_lr, None)
        actual_lr = self.curr_lr
        if num_invalids > 0:
            actual_lr = self.curr_lr * (experience_size - num_invalids) / experience_size
        self._apply_lr(actual_lr)

        stats = self._record_summaries(stats)

        # sync weights to inference workers
        synchronize(self.cfg, self.device)
        self.policy_versions_tensor[self.policy_id] = self.train_step

        return stats


    # =======================================
    # helpers for agents, not from Learner
    # Be careful when modifying

    def _evaluate_agent_log_probs(self, agent_id, gpu_buffer, agent_mask, chunk_size=4096):
        """
        for a single agent across the full buffer, chunk_size must be aligned to recurrence if use_rnn=True
        :param ...: Read yourself
        :returns: [n_agent_samples] tensor of log probs
        """
        agent_indices = torch.where(agent_mask)[0]
        if agent_indices.numel() == 0:
            return torch.tensor([], device=agent_indices.device)

        all_log_probs = []
        recurrence = self.cfg.recurrence if self.cfg.use_rnn else 1
        chunk_size = (chunk_size // recurrence) * recurrence
        chunk_size = max(chunk_size, recurrence)

        for start in range(0, len(agent_indices), chunk_size):
            chunk_idx = agent_indices[start : start + chunk_size]
            chunk_obs = {k: v[chunk_idx] for k, v in gpu_buffer["normalized_obs"].items()}
            chunk_actions = gpu_buffer["actions"][chunk_idx]
            chunk_rnn_states = gpu_buffer["rnn_states"][chunk_idx]
            # Slice actor only RNN states
            if self.actor_critic.use_critic_rnn:
                R = self.actor_critic.critic_rnn_state_size
                chunk_rnn_states = chunk_rnn_states[:, :R]
            agent_idx_sub = chunk_obs["agent_id"].argmax(dim=-1)
            with torch.no_grad():
                head_out = self.actor_critic.forward_head(chunk_obs, agent_idx=agent_idx_sub)
                if self.cfg.use_rnn:
                    # Use CPU indices for dones_cpu
                    chunk_idx_cpu = chunk_idx.cpu() # CPU tensor
                    chunk_dones = gpu_buffer["dones_cpu"][chunk_idx_cpu]
                    chunk_valids = gpu_buffer["valids"][chunk_idx_cpu].cpu()
                    done_or_invalid = torch.logical_or(chunk_dones, ~chunk_valids.bool()).float()
                    seq, rnn_init, inv = build_rnn_inputs(head_out, done_or_invalid, chunk_rnn_states, recurrence)
                    core_seq, _ = self.actor_critic.agent_cores[agent_id](seq, rnn_init)
                    core_out = build_core_out_from_seq(core_seq, inv)
                else:
                    core_out, _ = self.actor_critic.agent_cores[agent_id](head_out, chunk_rnn_states)

                decoder_out = self.actor_critic.agent_decoders[agent_id](core_out)
                logits, _ = self.actor_critic.agent_action_params[agent_id](decoder_out, None)

                dist = get_action_distribution(self.actor_critic.action_space, logits)
                log_probs = dist.log_prob(chunk_actions)

            all_log_probs.append(log_probs)

        return torch.cat(all_log_probs)

    def _calculate_agent_policy_loss(self, agent_id, mb, M_full, mb_indices, num_invalids):
        """
        Calculate PPO policy loss for a single agent with factor M
        :param agent_id: which agent
        :param mb: minibatch AttrDict
        :param M_full: full factor tensor [experience_size]
        :param mb_indices: indices of this minibatch in full buffer
        :param num_invalids: num of invalid samples in full batch
        """
        agent_idx_mb = mb.normalized_obs["agent_id"].argmax(dim=-1)
        assert (agent_idx_mb == agent_id).all(), (f"Mixed agents in single agent minibatch for agent {agent_id}")

        # Forward pass
        head_out = self.actor_critic.forward_head(mb.normalized_obs, agent_idx=agent_idx_mb)

        # Slice actor only RNN states
        actor_rnn = mb.rnn_states
        if self.actor_critic.use_critic_rnn:
            R = self.actor_critic.critic_rnn_state_size
            actor_rnn = actor_rnn[:, :R]

        if self.cfg.use_rnn:
            recurrence = self.cfg.recurrence
            done_or_invalid = torch.logical_or(mb.dones_cpu, ~mb.valids.cpu()).float()
            seq, rnn_init, inv = build_rnn_inputs(head_out, done_or_invalid, actor_rnn, recurrence)
            core_seq, _ = self.actor_critic.agent_cores[agent_id](seq, rnn_init)
            core_out = build_core_out_from_seq(core_seq, inv)
        else:
            core_out, _ = self.actor_critic.agent_cores[agent_id](head_out, actor_rnn)

        decoder_out = self.actor_critic.agent_decoders[agent_id](core_out)
        logits, _ = self.actor_critic.agent_action_params[agent_id](decoder_out, None)

        dist = get_action_distribution(self.actor_critic.action_space, logits)
        log_prob_actions = dist.log_prob(mb.actions)

        ratio = torch.exp(log_prob_actions - mb.log_prob_actions)
        ratio = torch.clamp(ratio, 0.05, 20.0)

        # M values for this minibatch
        M_mb = M_full[mb_indices].detach() # Not yet updated for this agent

        # PPO clipping
        clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
        # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
        clip_ratio_low = 1.0 / clip_ratio_high
        clipped_ratio = torch.clamp(ratio, clip_ratio_low, clip_ratio_high)

        loss_unclipped = ratio * M_mb
        loss_clipped = clipped_ratio * M_mb
        policy_loss = -torch.min(loss_unclipped, loss_clipped)

        # Mask invalid samples and compute mean of valid samples
        if mb.valids is not None:
            policy_loss = policy_loss * mb.valids
            n_valid = mb.valids.sum().clamp(min=1)
        else:
            n_valid = policy_loss.numel()
        policy_loss = policy_loss.sum() / n_valid

        exploration_loss = self.exploration_loss_func(dist, mb.valids, num_invalids)
        return policy_loss, exploration_loss

    def _calculate_critic_loss(self, mb, agent_idx_mb, env_group_idx_mb, num_invalids):
        """Centralized critic, separate critic encoders"""
        obs_no_id = {k: v for k, v in mb.normalized_obs.items() if k != "agent_id"}
        device = next(self.actor_critic.centralized_critic.parameters()).device
        critic_enc_out_size = self.actor_critic.critic_encoders[0].get_out_size()
        use_critic_rnn = self.actor_critic.use_critic_rnn

        if use_critic_rnn:
            critic_feature_size = self.actor_critic.critic_cores[0].get_out_size()
            # get critic half of stored rnn states
            R = self.actor_critic.critic_rnn_state_size
            critic_rnn_all = mb.rnn_states[:, R:]  # [mb_size, R]
        else:
            critic_feature_size = critic_enc_out_size

        critic_features = torch.zeros(agent_idx_mb.shape[0], critic_feature_size, device=device)

        for i in range(self.n_agents):
            mask = agent_idx_mb == i
            if not mask.any():
                continue

            agent_obs = {k: v[mask] for k, v in obs_no_id.items()}
            enc_out = self.actor_critic.critic_encoders[i](agent_obs)

            if self.actor_critic.critic_projection is not None:
                enc_out = self.actor_critic.critic_projection(enc_out)

            if use_critic_rnn:
                recurrence = self.cfg.recurrence
                agent_indices = torch.where(mask)[0]
                agent_indices_cpu = agent_indices.cpu()
                chunk_dones = mb.dones_cpu[agent_indices_cpu]
                chunk_valids = mb.valids[agent_indices_cpu].cpu()
                done_or_invalid = torch.logical_or(chunk_dones, ~chunk_valids.bool()).float()

                # get stored critic rnn states from rollout buff
                agent_critic_rnn = critic_rnn_all[mask]

                seq, rnn_init, inv = build_rnn_inputs(enc_out, done_or_invalid, agent_critic_rnn, recurrence)
                core_seq, _ = self.actor_critic.critic_cores[i](seq, rnn_init)
                core_out = build_core_out_from_seq(core_seq, inv)
                critic_features[mask] = core_out
            else:
                critic_features[mask] = enc_out

        n_transitions = env_group_idx_mb.max().item() + 1
        grouped = _group_by_env(critic_features, agent_idx_mb, env_group_idx_mb, self.n_agents, n_transitions)
        critic_input = grouped.view(n_transitions, -1)
        joint_value = self.actor_critic.centralized_critic(critic_input)
        values = joint_value.squeeze(-1)[env_group_idx_mb]
        value_loss = self._value_loss(values, mb.values, mb.returns, self.cfg.ppo_clip_value, mb.valids, num_invalids)
        return value_loss

    def _get_agent_minibatches(self, batch_size, agent_mask, experience_size):
        """
        For single agent training. Returns indices into full buffer that belong to this agent.
        """
        agent_indices = torch.where(agent_mask)[0].cpu().numpy()
        n_agent_samples = len(agent_indices)

        # Consider recurrence
        recurrence = self.cfg.recurrence
        if recurrence > 1:
            assert self.cfg.rollout % recurrence == 0, (f"rollout ({self.cfg.rollout}) must be divisible by recurrence ({recurrence})") # otherwise recurrence chunks span traj boundaries and corrupt RNN seq
            assert n_agent_samples % recurrence == 0, (f"Agent samples ({n_agent_samples}) not divisible by recurrence ({recurrence})")
            n_chunks = n_agent_samples // recurrence
            chunk_starts = np.arange(n_chunks)
            np.random.shuffle(chunk_starts)

            samples_per_batch = max(1, batch_size // (self.n_agents * recurrence)) * recurrence
            chunks_per_batch = max(1, samples_per_batch // recurrence)
            minibatches = []
            for i in range(0, n_chunks, chunks_per_batch):
                batch_chunks = chunk_starts[i : i + chunks_per_batch]
                indices = np.concatenate([agent_indices[c * recurrence : (c + 1) * recurrence] for c in batch_chunks])
                minibatches.append(indices)
        else:
            np.random.shuffle(agent_indices)
            samples_per_batch = max(1, batch_size // self.n_agents)
            minibatches = []
            for i in range(0, n_agent_samples, samples_per_batch):
                minibatches.append(agent_indices[i : i + samples_per_batch])

        return minibatches

    def _get_transition_minibatches(self, batch_size, experience_size, env_group_idx, n_transitions):
        """
        Make minibatches that contain complete transitions with all N agents
        """
        # Ref: "argsort Returns the indices that sort a tensor along a given dimension in ascending order by value."
        # argsort uses the C++ sort, this uses Introsort (source: I learned from Algo Engineering course:)),
        # and thus argsort is O(nlogn), its runtime dominate bincount and split
        # Vectorized grouping via argsort, O(nlogn) is worse than a Python loop O(n)
        # but faster in practice due to torch's C++/CUDA backend avoid Python interpreter overhead but use a compiled
        # sorting algorithm.
        sorted_idx = torch.argsort(env_group_idx)
        counts = torch.bincount(env_group_idx, minlength=n_transitions)
        transition_groups = torch.split(sorted_idx, counts.tolist())

        # Double check if right agents
        for t, group in enumerate(transition_groups):
            if len(group) != self.n_agents:
                raise RuntimeError(f"Transition {t} has {len(group)} samples, expected {self.n_agents}. Probably buffer layout issue.")

        # Shuffle trans
        transition_order = np.arange(n_transitions)
        np.random.shuffle(transition_order)

        # minibatches
        transitions_per_batch = max(1, batch_size // self.n_agents)
        minibatches = []
        for start in range(0, n_transitions, transitions_per_batch):
            batch_transitions = transition_order[start : start + transitions_per_batch]
            indices = torch.cat([transition_groups[t] for t in batch_transitions])
            minibatches.append(indices.cpu().numpy())
        return minibatches

    def _get_transition_rnn_minibatches(self, batch_size, experience_size, env_group_idx, n_transitions, env_idx, agent_idx):
        """
        Each minibatch contains temporally contiguous chunks of `recurrence` consecutive timesteps, with all N agents present at each timestep. This makes recurrence-aligned minibatches for critic BPTT. Groups by env_window, position (not env_idx) as theres multiple rollout windows from the same env in one training batch.
        """
        recurrence = self.cfg.recurrence
        rollout = self.cfg.rollout
        assert rollout % recurrence == 0, (
            f"rollout ({rollout}) must be divisible by recurrence ({recurrence})"
        )

        # map each trans to its sample indice
        sorted_idx = torch.argsort(env_group_idx)
        counts = torch.bincount(env_group_idx, minlength=n_transitions)
        transition_groups = torch.split(sorted_idx, counts.tolist())

        # Get env_window and timestep from env_group_idx
        # env_group_idx = env_window_group * rollout + timestep_idx
        transition_env_window = torch.zeros(n_transitions, dtype=torch.long, device=env_idx.device)
        transition_timestep = torch.zeros(n_transitions, dtype=torch.long, device=env_idx.device)
        for t, group in enumerate(transition_groups):
            gidx = env_group_idx[group[0]].item()
            transition_env_window[t] = gidx // rollout
            transition_timestep[t] = gidx % rollout

        # Group trans
        unique_windows = torch.unique(transition_env_window)

        all_chunks = []  #each chunk = recurrence consecutive trans
        for win_id in unique_windows:
            win_mask = (transition_env_window == win_id)
            win_transitions = torch.where(win_mask)[0]
            win_timesteps = transition_timestep[win_transitions]
            # sort temporally
            sorted_order = torch.argsort(win_timesteps)
            win_transitions_sorted = win_transitions[sorted_order]

            assert len(win_transitions_sorted) == rollout, (f"Window {win_id.item()} has {len(win_transitions_sorted)} transitions, expected {rollout}")
            for chunk_start in range(0, rollout, recurrence):
                chunk_trans = win_transitions_sorted[chunk_start : chunk_start + recurrence]
                all_chunks.append(chunk_trans)

        # suffer
        chunk_order = np.arange(len(all_chunks))
        np.random.shuffle(chunk_order)

        samples_per_chunk = recurrence * self.n_agents
        chunks_per_batch = max(1, batch_size // samples_per_chunk)
        minibatches = []
        for i in range(0, len(all_chunks), chunks_per_batch):
            batch_chunk_ids = chunk_order[i : i + chunks_per_batch]
            indices = []
            for c_id in batch_chunk_ids:
                chunk_trans = all_chunks[c_id]
                for t_id in chunk_trans:
                    indices.append(transition_groups[t_id.item()])
            indices = torch.cat(indices)
            minibatches.append(indices.cpu().numpy())
        return minibatches

    def _get_agent_params(self, agent_id):
        return (
            list(self.actor_critic.agent_encoders[agent_id].parameters())
            + list(self.actor_critic.agent_cores[agent_id].parameters())
            + list(self.actor_critic.agent_decoders[agent_id].parameters())
            + list(self.actor_critic.agent_action_params[agent_id].parameters())
        )

    def _get_critic_params(self):
        params = (list(self.actor_critic.centralized_critic.parameters()) + list(self.actor_critic.critic_encoders.parameters()))
        if self.actor_critic.critic_cores is not None:
            params += list(self.actor_critic.critic_cores.parameters())
        if self.actor_critic.critic_projection is not None:
            params += list(self.actor_critic.critic_projection.parameters())
        return params

    def _compute_env_group_idx(self, agent_idx, env_idx, dataset_size):
        """
        Compute group index from by position, n_agents consecutive traj share the same env and rollout window at each timestep

        :returns: [dataset_size] tensor mapping each sample to its transition group
        """
        rollout = self.cfg.rollout
        n_agents = self.n_agents

        arange = torch.arange(dataset_size, device=env_idx.device)
        traj_idx = arange // rollout
        timestep_idx = arange % rollout
        # n_agents consecutive traj form one (env, window) group
        env_window_group = traj_idx // n_agents

        group_idx = env_window_group * rollout + timestep_idx
        n_transitions = group_idx.max().item() + 1

        counts = torch.bincount(group_idx, minlength=n_transitions)
        if not (counts == n_agents).all():
            bad = (counts != n_agents).nonzero(as_tuple=True)[0]
            raise RuntimeError(
                f"env_group_idx: {bad.numel()} transitions have wrong agent count (expected {n_agents}, got counts {counts[bad[:5]].tolist()}). "
                f"dataset_size={dataset_size} must be divisible by n_agents*rollout={n_agents * rollout}."
            )

        # consecutive traj groups have same env_idx
        traj_env = env_idx.view(-1, rollout)[:, 0]
        n_traj = traj_env.shape[0]
        if n_traj >= n_agents and n_traj % n_agents == 0:
            grouped_env = traj_env.view(-1, n_agents)
            if not (grouped_env == grouped_env[:, 0:1]).all():
                bad_groups = (grouped_env != grouped_env[:, 0:1]).any(dim=1).nonzero(as_tuple=True)[0]
                raise RuntimeError(f"Consecutive trajectory groups have mismatched env_idx at group(s) {bad_groups[:5].tolist()}.")

        return group_idx
