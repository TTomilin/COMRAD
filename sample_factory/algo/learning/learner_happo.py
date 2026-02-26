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

from sf.doom.happo_model import _group_by_env


class HAPPOLearner(Learner):
    """
    Loop:
    1. Prepare batch, GAE advantages
    2. Init M = advantages
    3. For each agent in random order (can be set to fixed order as well):
        1. snapshot old log probs
        2. train agent for K epochs
        3. eval new log probs 
        4. M *= unclipped post-update ratio
    4. Train centralized critic separately
    """

    def __init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server):
        super().__init__(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        self.n_agents = cfg.num_agents

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

        critic_params = (list(self.actor_critic.centralized_critic.parameters()) + list(self.actor_critic.critic_encoders.parameters()))
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
