from __future__ import annotations

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
