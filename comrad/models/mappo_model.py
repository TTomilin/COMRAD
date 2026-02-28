import torch
import torch.nn as nn
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.algo.utils.context import global_model_factory
from typing import Optional
from sample_factory.algo.utils.tensor_dict import TensorDict
# from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.actor_critic import obs_space_without_action_mask
from torch.nn import Linear


class MAPPOActorCritic(ActorCriticSharedWeights):
    """
    Multi-Agent PPO (MAPPO) Actor-Critic model with centralized critic.
    """

    def __init__(self, model_factory, obs_space, action_space, cfg):
        # parent constructor sets up encoder, core, decoder, and action_parameterization
        super().__init__(model_factory, obs_space, action_space, cfg)

        self.cfg = cfg
        self.n_agents: int = getattr(cfg, 'num_agents', 1)
        self.use_centralized_critic: bool = str(getattr(cfg, 'algo', 'APPO')).upper() == 'MAPPO'

        # Replace the standard critic with centralized one if MAPPO
        if self.use_centralized_critic:
            decoder_out_size = self.decoder.get_out_size()
            # takes concatenated features from all agents
            # one linear layer may potentially be enough, but we use a small MLP here just in case
            self.centralized_critic = nn.Sequential(
                nn.Linear(decoder_out_size * self.n_agents, 512),
                nn.Tanh(),
                nn.Linear(512, 256),
                nn.Tanh(),
                nn.Linear(256, self.n_agents),
            )
            # Don't use the standard critic
            self.critic_linear: Linear = None

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool,
                     action_mask: Optional[torch.Tensor] = None) -> TensorDict:
        """
        Modified forward_tail to handle centralized critic.
        """
        decoder_output = self.decoder(core_output)

        if self.use_centralized_critic:
            # decoder_output.shape [batch_size * n_agents, decoder_out_size]
            batch_size = decoder_output.shape[0] // self.n_agents

            # [batch_size, n_agents, decoder_out_size]
            decoder_reshaped = decoder_output.view(batch_size, self.n_agents, -1)

            # [batch_size, n_agents * decoder_out_size]
            centralized_input = decoder_reshaped.view(batch_size, -1)

            values_all = self.centralized_critic(centralized_input)

            # back to [batch_size * n_agents]
            values = values_all.view(-1)
        else:
            # IPPO with decentralized critic
            values = self.critic_linear(decoder_output).squeeze()

        result = TensorDict(values=values)

        if values_only:
            return result

        # decentralized actions
        action_distribution_params, self.last_action_distribution = self.action_parameterization(decoder_output, action_mask)

        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result


def make_mappo_actor_critic(cfg, obs_space, action_space):
    """Factory function for MAPPO/IPPO models."""

    model_factory = global_model_factory()
    obs_space = obs_space_without_action_mask(obs_space)

    return MAPPOActorCritic(model_factory, obs_space, action_space, cfg)
