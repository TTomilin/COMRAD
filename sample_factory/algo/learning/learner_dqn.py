from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

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
from sample_factory.algo.utils.replay_buffer import ReplayBuffer
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_factory.algo.utils.torch_utils import synchronize, to_scalar
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import Config, InitModelData, PolicyID
from sample_factory.utils.utils import log


class DQNLearner(Learner):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        super().__init__(cfg, env_info, policy_versions_tensor, policy_id, param_server)

        self.target_network = None
        self.replay_buffer: Optional[ReplayBuffer] = None

        self.total_env_steps_for_training = 0
        self.last_target_update_step = 0

    def init(self) -> InitModelData:
        init_data = super().init()

        self.target_network = copy.deepcopy(self.actor_critic)
        self.target_network.eval()
        # No requires_grad=False for target network bcus torch.no_grad() always used when computing target Q-values

        self.replay_buffer = ReplayBuffer(
            capacity=self.cfg.replay_buffer_size,
            obs_space=self.env_info.obs_space,
            action_space=self.env_info.action_space,
            device="cpu",
            share_memory=not self.cfg.serial_mode,
        )
        return init_data

    def _update_target_network(self, tau: float = 1.0) -> None:
        if (self.train_step - self.last_target_update_step >= self.cfg.target_update_interval):
            if self.target_network is None or self.actor_critic is None:
                return

            if tau == 1.0:
                self.target_network.load_state_dict(self.actor_critic.state_dict())
            else:
                # Interpolate the weights
                for target_param, param in zip(self.target_network.parameters(), self.actor_critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            self.last_target_update_step = self.train_step
            log.debug(f"Updated target network at step {self.train_step}")

    def _prepare_batch_for_buffer(self, batch: TensorDict) -> TensorDict:
        """Batch shape: [num_trajectories, rollout_length, ...]"""
        with torch.no_grad():
            buff = shallow_recursive_copy(batch)

            num_traj = buff["rewards"].shape[0]
            rollout_len = buff["rewards"].shape[1]

            # obs[:, :-1] current states, obs[:, 1:] next states
            obs = buff["obs"]
            transitions = TensorDict()
            transitions["obs"] = TensorDict()
            for key, value in obs.items():
                current_obs = value[:, :-1]
                transitions["obs"][key] = current_obs.reshape((num_traj * rollout_len,) + current_obs.shape[2:])

            transitions["next_obs"] = TensorDict()
            for key, value in obs.items():
                next_obs = value[:, 1:]
                transitions["next_obs"][key] = next_obs.reshape((num_traj * rollout_len,) + next_obs.shape[2:])

            transitions["actions"] = buff["actions"].reshape(-1, *buff["actions"].shape[2:])
            transitions["rewards"] = buff["rewards"].reshape(-1)
            transitions["dones"] = buff["dones"].reshape(-1).float()

            return transitions

    def _calculate_dqn_loss(self, batch: TensorDict) -> Tensor:
        """
        Can use Double DQN
        """
        if self.actor_critic is None or self.target_network is None:
            raise RuntimeError("Networks not initialized")

        with self.param_server.policy_lock:
            normalized_obs = prepare_and_normalize_obs(self.actor_critic, batch["obs"])
            normalized_next_obs = prepare_and_normalize_obs(self.actor_critic, batch["next_obs"])

        actions = batch["actions"].long()
        rewards = batch["rewards"]
        dones = batch["dones"]

        # Current q
        batch_size = actions.shape[0]
        rnn_states = torch.zeros(
            batch_size,
            self.actor_critic.core.get_out_size()
            if hasattr(self.actor_critic, "core")
            else 1,
            device=self.device,
        )
        result = self.actor_critic(normalized_obs, rnn_states, values_only=False)
        q_values = result["action_logits"] # [batch, total_num_actions]

        action_space = self.env_info.action_space
        if hasattr(action_space, "spaces"):
            action_sizes = [s.n for s in action_space.spaces]
            q_splits = torch.split(q_values, action_sizes, dim=1)

            if actions.dim() == 1:
                actions = actions.unsqueeze(1)

            current_q_list = []
            for i, (q_head, a_head_size) in enumerate(zip(q_splits, action_sizes)):
                a_idx = actions[:, i : i + 1] # [batch, 1]
                current_q_list.append(q_head.gather(1, a_idx).squeeze(1))

            current_q = torch.stack(current_q_list, dim=1).sum(dim=1)

            with torch.no_grad():
                # Get next Q-values from target network
                target_result = self.target_network(normalized_next_obs, rnn_states, values_only=False)
                next_q_target = target_result["action_logits"]
                next_q_target_splits = torch.split(next_q_target, action_sizes, dim=1)

                next_q_list = [q.max(dim=1)[0] for q in next_q_target_splits]
                next_q = torch.stack(next_q_list, dim=1).sum(dim=1)

                # r + gamma * Q_target(s', a') * (1 - done)
                # https://stackoverflow.com/questions/58559415/setting-up-target-values-for-deep-q-learning
                target_q = rewards + self.cfg.gamma * next_q * (1.0 - dones)
        else:
            # Single action space
            if actions.dim() > 1:
                actions = actions.squeeze(-1)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                target_result = self.target_network(normalized_next_obs, rnn_states, values_only=False)
                next_q_target = target_result["action_logits"]

                if self.cfg.double_dqn:
                    online_result = self.actor_critic(normalized_next_obs, rnn_states, values_only=False)
                    next_q_online = online_result["action_logits"]
                    next_actions = next_q_online.argmax(dim=1, keepdim=True)
                    next_q = next_q_target.gather(1, next_actions).squeeze(1)
                else:
                    next_q = next_q_target.max(dim=1)[0]

                target_q = rewards + self.cfg.gamma * next_q * (1.0 - dones)

        # Huber loss (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py)
        loss = F.smooth_l1_loss(current_q, target_q)
        return loss

    def _train_on_batch(self, batch: TensorDict) -> Optional[AttrDict]:
        if self.actor_critic is None or self.optimizer is None:
            return None

        self.actor_critic.train()

        loss = self._calculate_dqn_loss(batch)

        for p in self.actor_critic.parameters():
            p.grad = None

        loss.backward()

        if self.cfg.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)

        with self.param_server.policy_lock:
            self.optimizer.step()

        self._after_optimizer_step()

        self._update_target_network(self.cfg.target_update_tau)

        # Sync weights
        synchronize(self.cfg, self.device)
        self.policy_versions_tensor[self.policy_id] = self.train_step

        stats = AttrDict()
        stats.loss = to_scalar(loss)
        stats.lr = self.curr_lr
        return stats

    def train(self, batch: TensorDict) -> Optional[Dict]:
        with self.timing.add_time("misc"):
            self._maybe_update_cfg()
            self._maybe_load_policy()

        with self.timing.add_time("prepare_batch"):
            transitions = self._prepare_batch_for_buffer(batch)

            num_transitions = transitions["rewards"].shape[0]
            if self.cfg.summaries_use_frameskip:
                self.env_steps += num_transitions * self.env_info.frameskip
            else:
                self.env_steps += num_transitions

        with self.timing.add_time("add_to_buffer"):
            if self.replay_buffer is not None:
                self.replay_buffer.add(transitions)

        if (self.replay_buffer is None or len(self.replay_buffer) < self.cfg.learning_starts):
            return {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}

        train_stats = None
        with self.timing.add_time("train"):
            sampled_batch = self.replay_buffer.sample(self.cfg.batch_size, str(self.device))
            if sampled_batch is not None:
                train_stats = self._train_on_batch(sampled_batch)

        stats = {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
        if train_stats is not None:
            stats[TRAIN_STATS] = train_stats
            stats[STATS_KEY] = memory_stats("learner", self.device)

        return stats

    def _get_checkpoint_dict(self):
        checkpoint = super()._get_checkpoint_dict()
        if self.target_network is not None:
            checkpoint["target_network"] = self.target_network.state_dict()
        if self.replay_buffer is not None:
            checkpoint["replay_buffer_size"] = len(self.replay_buffer)
        return checkpoint

    def _load_state(self, checkpoint_dict, load_progress=True):
        super()._load_state(checkpoint_dict, load_progress)
        if "target_network" in checkpoint_dict and self.target_network is not None:
            self.target_network.load_state_dict(checkpoint_dict["target_network"])
            log.info("Loaded target network from checkpoint")
