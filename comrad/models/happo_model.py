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
    Optional: --happo_critic_rnn adds per agent RNN cores to critic, runs at both inference and training
    Critic RNN states are stored in the rollout buffer (rnn_states doubles in size)

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

        self.use_critic_rnn = use_critic_rnn
        if use_critic_rnn:
            # Size of one agent's critic RNN state vector
            single_agent_rnn = cfg.rnn_size * cfg.rnn_num_layers
            if cfg.rnn_type == "lstm":
                single_agent_rnn *= 2
            self.critic_rnn_state_size = single_agent_rnn
        else:
            self.critic_rnn_state_size = 0

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

    def _compute_agent_indices(self, agent_idx):
        """
        Precompute per agent idx tensors to avoid repeated boolean masking and GPU syncs
        Used when forward_head/forward_core/forward_tail are called directly (from learner) where batch layout not interleaved
        """
        indices = []
        for i in range(self.n_agents):
            idx = (agent_idx == i).nonzero(as_tuple=True)[0]
            indices.append(idx)
        return indices

    def _compute_agent_indices_strided(self, B, device):
        """Uses deterministic stride pattern - agents always interleaved [0,1,...,N-1,0,1,...] - to avoid GPU-CPU syncs. Only valid when called from forward() where inference batch layout is interleaved"""
        return [torch.arange(i, B, self.n_agents, device=device) for i in range(self.n_agents)]

    def forward(self, normalized_obs_dict, rnn_states, values_only=False, action_mask=None, sample_actions=True):
        """Returns tensordict"""
        agent_idx = normalized_obs_dict['agent_id'].argmax(dim=-1)
        B = agent_idx.shape[0]
        # Precompute integer indices once for all
        agent_indices = self._compute_agent_indices_strided(B, agent_idx.device)
        head_out = self.forward_head(normalized_obs_dict, agent_idx=agent_idx, agent_indices=agent_indices)

        # Split actor critic RNN states
        if self.use_critic_rnn:
            R = self.critic_rnn_state_size
            actor_rnn = rnn_states[:, :R]
            critic_rnn = rnn_states[:, R:]
        else:
            actor_rnn = rnn_states
            critic_rnn = None

        # pass to forward core
        core_out, new_actor_rnn = self.forward_core(head_out, actor_rnn, agent_idx=agent_idx, agent_indices=agent_indices)

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
            critic_rnn_states=critic_rnn,
            head_output=head_out,
            agent_indices=agent_indices,
        )

        # updated critic half
        if self.use_critic_rnn and "new_critic_rnn_states" in result:
            new_critic_rnn = result.pop("new_critic_rnn_states")
            result['new_rnn_states'] = torch.cat([new_actor_rnn, new_critic_rnn], dim=1)
        else:
            result['new_rnn_states'] = new_actor_rnn

        return result

    def forward_head(self, obs_dict, *, agent_idx=None, agent_indices=None):
        """
        Route obs to each agents' encoder
        """
        if agent_idx is None:
            agent_idx = obs_dict['agent_id'].argmax(dim=-1)
        obs_no_id = {k: v for k, v in obs_dict.items() if k != 'agent_id'}
        device = model_device(self)
        B = next(iter(obs_no_id.values())).shape[0]
        out = torch.zeros(B, self.agent_encoders[0].get_out_size(), device=device)

        if agent_indices is None:
            agent_indices = self._compute_agent_indices(agent_idx)

        for i in range(self.n_agents):
            idx = agent_indices[i]
            if idx.numel() > 0:
                agent_obs = {k: v[idx] for k, v in obs_no_id.items()}
                out[idx] = self.agent_encoders[i](agent_obs)
        return out

    def forward_core(self, head_output, rnn_states, *, agent_idx=None, agent_indices=None):
        """
        same but RNN cores
        """
        if agent_idx is None:
            raise ValueError("agent_idx is required for HAPPOActorCritic.forward_core")
        device = model_device(self)
        B = head_output.shape[0]
        out = torch.zeros(B, self.agent_cores[0].get_out_size(), device=device)
        new_rnn = torch.zeros_like(rnn_states)

        if agent_indices is None:
            agent_indices = self._compute_agent_indices(agent_idx)

        for i in range(self.n_agents):
            idx = agent_indices[i]
            if idx.numel() > 0:
                core_out_i, rnn_i = self.agent_cores[i](head_output[idx], rnn_states[idx])
                out[idx] = core_out_i
                new_rnn[idx] = rnn_i
        return out, new_rnn

    def forward_tail(self, core_output, *, agent_idx=None, env_group_idx=None, normalized_obs_dict=None,
                     values_only=False, sample_actions=True, action_mask=None, critic_rnn_states=None,
                     head_output=None, agent_indices=None):
        """
        separate decoders + centralized critic + critic encoder

        All args after core_output are keyword-only.
        head_output: if provided and not training, reuse actor encoder features for critic to skip the expensive critic CNN encoders at inference time.
        agent_indices: pre-computed integer index tensors per agent (avoids repeated masking or GPU syncs)
        """
        if agent_idx is None:
            raise ValueError("agent_idx is required for HAPPOActorCritic.forward_tail")
        device = model_device(self)
        B = core_output.shape[0]

        if agent_indices is None:
            agent_indices = self._compute_agent_indices(agent_idx)

        # V Critic centralized critic with separate encoders
        obs_no_id = {k: v for k, v in normalized_obs_dict.items() if k != 'agent_id'}
        critic_enc_out_size = self.critic_encoders[0].get_out_size()
        if self.use_critic_rnn:
            critic_feature_size = self.critic_cores[0].get_out_size()
        elif self.critic_cores is not None:
            critic_feature_size = self.critic_cores[0].get_out_size()
        else:
            critic_feature_size = critic_enc_out_size

        if self.use_critic_rnn:
            if critic_rnn_states is None:
                raise ValueError("forward_tail: use_critic_rnn=True but critic_rnn_states is None, must extract and pass critic states.")

        # At inference (not training), reuse actor encoder features for the critic to avoid running 2 extra CNN passe
        # The critic encoders have diff weights, but the actor features are a good approx for GAE bootstrapping. The learner always runs the real critic encoders
        skip_critic_encoder = (head_output is not None and not self.training)

        critic_features = torch.zeros(B, critic_feature_size, device=device)
        new_critic_rnn = torch.zeros_like(critic_rnn_states) if critic_rnn_states is not None else None

        if skip_critic_encoder:
            # Reuse actor encoder output as critic features
            enc_out = head_output
            if self.critic_projection is not None:
                enc_out = self.critic_projection(enc_out)
            if self.use_critic_rnn:
                for i in range(self.n_agents):
                    idx = agent_indices[i]
                    if idx.numel() > 0:
                        core_out_i, rnn_i = self.critic_cores[i](enc_out[idx], critic_rnn_states[idx])
                        critic_features[idx] = core_out_i
                        new_critic_rnn[idx] = rnn_i
            else:
                critic_features = enc_out
        else:
            for i in range(self.n_agents):
                idx = agent_indices[i]
                if idx.numel() > 0:
                    agent_obs = {k: v[idx] for k, v in obs_no_id.items()}
                    enc_out = self.critic_encoders[i](agent_obs)
                    if self.critic_projection is not None:
                        enc_out = self.critic_projection(enc_out)
                    if self.use_critic_rnn:
                        core_out_i, rnn_i = self.critic_cores[i](enc_out, critic_rnn_states[idx])
                        critic_features[idx] = core_out_i
                        new_critic_rnn[idx] = rnn_i
                    else:
                        critic_features[idx] = enc_out

        # Group critic features by env transition for centralized critic
        n_transitions = B // self.n_agents # no GPU sync
        grouped = _group_by_env(critic_features, agent_idx, env_group_idx, self.n_agents, n_transitions)
        critic_input = grouped.view(n_transitions, -1) # [n_transitions, n_agents * critic_feature_size]
        joint_value = self.centralized_critic(critic_input) # [n_transitions, 1]
        values = joint_value.squeeze(-1)[env_group_idx] # Broadcast to all agents [B]

        result = TensorDict(values=values)
        if new_critic_rnn is not None:
            result["new_critic_rnn_states"] = new_critic_rnn
        if values_only: return result

        # decoders each agetns
        # This is only needed for action logits, skipped for values_only
        decoder_out_size = self.agent_decoders[0].get_out_size()
        decoder_outputs = torch.zeros(B, decoder_out_size, device=device)
        for i in range(self.n_agents):
            idx = agent_indices[i]
            if idx.numel() > 0:
                decoder_outputs[idx] = self.agent_decoders[i](core_output[idx])

        # action parameterization
        all_action_logits = torch.zeros(B, self.action_logit_dim, device=device)
        for i in range(self.n_agents):
            idx = agent_indices[i]
            if idx.numel() > 0:
                mask_action_mask = action_mask[idx] if action_mask is not None else None
                logits_i, _ = self.agent_action_params[i](decoder_outputs[idx], mask_action_mask)
                all_action_logits[idx] = logits_i

        result["action_logits"] = all_action_logits

        # Store dist for when learner call action_distribution()
        from sample_factory.algo.utils.action_distributions import get_action_distribution
        self.last_action_distribution = get_action_distribution(self.action_space, all_action_logits, action_mask=action_mask)
        self._maybe_sample_actions(sample_actions, result)
        return result


def _group_by_env(features, agent_idx, env_group_idx, n_agents, n_transitions=None):
    """
    [B, F] -> [n_transitions, n_agents, F] by env
    Concat all agents features in a transition to get joint obs
    When n_transitions is provided (inference path), uses zero-copy reshape since agents are interleaved [a0,a1,...,aN-1,a0,a1,...].
    """
    if n_transitions is not None:
        # Agents interleaved so just reshape
        return features.view(n_transitions, n_agents, features.shape[1])
    # Scatter for non-standard orderings
    n_transitions = env_group_idx.max().item() + 1
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
