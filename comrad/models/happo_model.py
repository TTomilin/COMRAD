from __future__ import annotations

import torch
import torch.nn as nn

from sample_factory.model.actor_critic import ActorCritic, obs_space_without_action_mask
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.model_utils import model_device
from sample_factory.utils.typing import ActionSpace, ObsSpace, Config

import gymnasium as gym


def remove_agentid(obs_space):
    """Remove agent_id key from obs space for encoder input"""
    return gym.spaces.Dict({k: v for k, v in obs_space.spaces.items() if k != 'agent_id'})


class HAPPOActorCritic(ActorCritic):
    """
    Ref: https://github.com/PKU-MARL/HARL/blob/main/harl/algorithms/actors/happo.py

    HAPPO actor critic
    Separate policy networks (encoder + core + decoder + action_param) each agents
    Independent critic encoder processes all agents' obs, MLP outputs scalar value
    Default: MLP-only critic like HARL paper. Actor still supports GRU/LSTM.
    Optional: --happo_critic_rnn adds per agent RNN cores to critic but training-only BPTT

    I dont use ActorCriticSharedWeights because the sharedweights code doesn't have N separate policy netowrks
    """

    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        cfg: Config
    ):
        super().__init__(obs_space, action_space, cfg)
        self.cfg = cfg
        self.n_agents = cfg.num_agents

        obs_space_no_id = remove_agentid(obs_space)

        # Create N separate policy networks
        self.agent_encoders = nn.ModuleList()
        self.agent_cores = nn.ModuleList()
        self.agent_decoders = nn.ModuleList()
        self.agent_action_params = nn.ModuleList()
        for i in range(self.n_agents):
            enc = model_factory.make_model_encoder_func(cfg, obs_space_no_id)
            core = model_factory.make_model_core_func(cfg, enc.get_out_size())
            dec = model_factory.make_model_decoder_func(cfg, core.get_out_size())
            act_param = self.get_action_parameterization(dec.get_out_size())
            self.agent_encoders.append(enc)
            self.agent_cores.append(core)
            self.agent_decoders.append(dec)
            self.agent_action_params.append(act_param)

        # (device_for_input_tensor() uses self.encoders[0]
        self.encoders = list(self.agent_encoders)


        # ===========================================================

        # https://github.com/PKU-MARL/HARL/blob/main/harl/algorithms/critics/v_critic.py
        # Separate critic, with V Critic
        # concat-ed obs -> MLP -> single joint V(s)
        # Note: Im not using VNet and VCritic, it's coded into this actor critic

        self.critic_encoders = nn.ModuleList()
        for i in range(self.n_agents):
            critic_enc = model_factory.make_model_encoder_func(cfg, obs_space_no_id)
            self.critic_encoders.append(critic_enc)

        critic_enc_out = self.critic_encoders[0].get_out_size()

        # per agent critic RNN cores
        self.critic_cores = None
        self.critic_projection = None
        use_critic_rnn = getattr(cfg, 'happo_critic_rnn', False) and cfg.use_rnn
        if use_critic_rnn:
            # get out size rnn
            _probe_core = model_factory.make_model_core_func(cfg, critic_enc_out)
            critic_feature_size = _probe_core.get_out_size() # probing
            del _probe_core

            # Project encoder output to RNN output size
            # Note: both training and inference: encoder -> projection
            if critic_enc_out != critic_feature_size:
                self.critic_projection = nn.Linear(critic_enc_out, critic_feature_size) # projection gets grad during training
                rnn_input_size = critic_feature_size
            else:
                rnn_input_size = critic_enc_out

            self.critic_cores = nn.ModuleList()
            for i in range(self.n_agents):
                critic_core = model_factory.make_model_core_func(cfg, rnn_input_size)
                self.critic_cores.append(critic_core)
        else:
            critic_feature_size = critic_enc_out

        critic_input_dim = critic_feature_size * self.n_agents

        # Build critic MLP, the critic sees all agents' encoded
        hidden_sizes = getattr(cfg, 'happo_critic_hidden_sizes', [512, 256])
        critic_layers = []
        prev_dim = critic_input_dim
        for h in hidden_sizes:
            critic_layers.append(nn.Linear(prev_dim, h))
            critic_layers.append(nn.Tanh())
            prev_dim = h
        critic_layers.append(nn.Linear(prev_dim, 1)) # Single joint value V(s)
        self.centralized_critic = nn.Sequential(*critic_layers)

        # ===========================================================


        # Note: base __init__ already creates self.obs_normalizer and centralized_critic replaces self.critic_linear


        # Cache action_logit_dim so it's not recomputed every call
        decoder_out = self.agent_decoders[0].get_out_size()
        with torch.no_grad():
            dummy = torch.zeros(1, decoder_out)
            sample_logits, _ = self.agent_action_params[0](dummy, None)
        self.action_logit_dim = sample_logits.shape[-1]

        self.apply(self.initialize_weights)

    def forward(self, normalized_obs_dict, rnn_states, values_only=False, action_mask=None, sample_actions=True):
        """Returns tensordict"""
        agent_idx = normalized_obs_dict['agent_id'].argmax(dim=-1)
        head_out = self.forward_head(normalized_obs_dict, agent_idx=agent_idx)
        core_out, new_rnn = self.forward_core(head_out, rnn_states, agent_idx=agent_idx)

        # Group bases on position. Should be safe for batched_sampling=True unless race condition, but idk
        # In learner, after agent i trains, env_group_idx will be used to multiply all agents in the same transition
        env_group_idx = torch.arange(agent_idx.shape[0], device=agent_idx.device) // self.n_agents
        result = self.forward_tail(
            core_out,
            agent_idx=agent_idx,
            env_group_idx=env_group_idx,
            normalized_obs_dict=normalized_obs_dict,
            values_only=values_only,
            sample_actions=sample_actions,
            action_mask=action_mask,
        )
        result['new_rnn_states'] = new_rnn
        return result

    def forward_head(self, obs_dict, *, agent_idx=None):
        """
        Route obs to each agents' encoder
        """
        if agent_idx is None:
            agent_idx = obs_dict['agent_id'].argmax(dim=-1)
        obs_no_id = {k: v for k, v in obs_dict.items() if k != 'agent_id'}
        device = model_device(self)
        B = next(iter(obs_no_id.values())).shape[0]
        out = torch.zeros(B, self.agent_encoders[0].get_out_size(), device=device)

        for i in range(self.n_agents):
            mask = (agent_idx == i)
            if mask.any():
                agent_obs = {k: v[mask] for k, v in obs_no_id.items()}
                out[mask] = self.agent_encoders[i](agent_obs)
        return out

    def forward_core(self, head_output, rnn_states, *, agent_idx=None):
        """
        same but RNN cores
        """
        if agent_idx is None:
            raise ValueError("agent_idx is required for HAPPOActorCritic.forward_core")
        device = model_device(self)
        B = head_output.shape[0]
        out = torch.zeros(B, self.agent_cores[0].get_out_size(), device=device)
        new_rnn = torch.zeros_like(rnn_states)

        for i in range(self.n_agents):
            mask = (agent_idx == i)
            if mask.any():
                core_out_i, rnn_i = self.agent_cores[i](head_output[mask], rnn_states[mask])
                out[mask] = core_out_i
                new_rnn[mask] = rnn_i
        return out, new_rnn

    def forward_tail(self, core_output, *, agent_idx=None, env_group_idx=None, normalized_obs_dict=None,
                     values_only=False, sample_actions=True, action_mask=None):
        """
        separate decoders + centralized critic + critic encoder

        All args after core_output are keyword-only
        """
        if agent_idx is None:
            raise ValueError("agent_idx is required for HAPPOActorCritic.forward_tail")
        device = model_device(self)
        B = core_output.shape[0]

        # V Critic centralized critic with separate encoders
        # Should also look at HARL's implementation for this, this is afapted to work with SF
        obs_no_id = {k: v for k, v in normalized_obs_dict.items() if k != 'agent_id'}
        critic_enc_out_size = self.critic_encoders[0].get_out_size()
        if self.critic_cores is not None:
            # project MLP input dim
            critic_feature_size = self.critic_cores[0].get_out_size()
        else:
            critic_feature_size = critic_enc_out_size
        critic_features = torch.zeros(B, critic_feature_size, device=device)
        for i in range(self.n_agents):
            mask = (agent_idx == i)
            if mask.any():
                agent_obs = {k: v[mask] for k, v in obs_no_id.items()}
                enc_out = self.critic_encoders[i](agent_obs)
                if self.critic_projection is not None:
                    enc_out = self.critic_projection(enc_out)
                critic_features[mask] = enc_out

        n_transitions = env_group_idx.max().item() + 1
        grouped = _group_by_env(critic_features, agent_idx, env_group_idx, self.n_agents)
        critic_input = grouped.view(n_transitions, -1) # [n_transitions, n_agents * critic_feature_size]
        joint_value = self.centralized_critic(critic_input) # [n_transitions, 1]
        values = joint_value.squeeze(-1)[env_group_idx] # Broadcast to all agents [B]

        result = TensorDict(values=values)
        if values_only: return result

        # decoders each agetns
        # This is only needed for action logits, skipped for values_only
        decoder_out_size = self.agent_decoders[0].get_out_size()
        decoder_outputs = torch.zeros(B, decoder_out_size, device=device)
        for i in range(self.n_agents):
            mask = (agent_idx == i)
            if mask.any():
                decoder_outputs[mask] = self.agent_decoders[i](core_output[mask])

        # action parameterization
        all_action_logits = torch.zeros(B, self.action_logit_dim, device=device)
        for i in range(self.n_agents):
            mask = (agent_idx == i)
            if mask.any():
                mask_action_mask = action_mask[mask] if action_mask is not None else None
                logits_i, _ = self.agent_action_params[i](decoder_outputs[mask], mask_action_mask)
                all_action_logits[mask] = logits_i

        result["action_logits"] = all_action_logits

        # Store dist for when learner call action_distribution()
        from sample_factory.algo.utils.action_distributions import get_action_distribution
        self.last_action_distribution = get_action_distribution(self.action_space, all_action_logits, action_mask=action_mask)
        self._maybe_sample_actions(sample_actions, result)
        return result


def _group_by_env(features, agent_idx, env_group_idx, n_agents):
    """
    [B, F] -> [n_transitions, n_agents, F] by env
    Concat all agents features in a transition to get joint obs
    """
    n_transitions = env_group_idx.max().item() + 1
    # (env_group_idx, agent_idx) pairs shouldnt have dupe
    pairs = env_group_idx * n_agents + agent_idx
    assert pairs.unique().shape[0] == pairs.shape[0], ("Duplicate (env_group_idx, agent_idx) pairs detected in _group_by_env")
    grouped = torch.zeros(n_transitions, n_agents, features.shape[1], device=features.device)
    grouped[env_group_idx, agent_idx] = features
    return grouped


def make_happo_actor_critic(cfg, obs_space, action_space):
    model_factory = global_model_factory()
    # This strips action_mask so it's not learned, but this still flows through `obs.pop("action_mask")` at inference
    # and pass through forward() call. It's logit masking in CategoricalActionDistribution
    # It's correct, so dont change it.
    # Note: To get actual heterogeneous action masking, just have a wrapper that adds action_mask to obs space and fills it per agent at each step
    obs_space = obs_space_without_action_mask(obs_space)
    return HAPPOActorCritic(model_factory, obs_space, action_space, cfg)
