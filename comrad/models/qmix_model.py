"""
Note:
I tried using model factory encoder path but checkpoint fails on large buffer loads
and kind of breaks training. You may try implementing it instead of for example
hardcoding encoder like Im doing. I just find this more flexible
(Core functionalities are still from sample factory).
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.encoder import make_img_encoder
from sample_factory.model.core import ModelCoreRNN, ModelCoreIdentity
from sample_factory.model.decoder import MlpDecoder
from sample_factory.model.model_utils import nonlinearity
from sample_factory.model.actor_critic import obs_space_without_action_mask
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from sample_factory.utils.utils import log


class QMixAgentNet(nn.Module):
    """
    Q net per agent (encoder -> core (optional) -> decoder -> Q heads) (shared weight)
    """

    def __init__(
        self,
        cfg: Config,
        obs_space: ObsSpace,
        action_space: ActionSpace,
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_space = obs_space
        self.action_space = action_space

        # Encoder
        if "obs" in obs_space.spaces:
            self.encoder = make_img_encoder(cfg, obs_space["obs"])
        else:
            from sample_factory.model.encoder import MlpEncoder
            obs_shape = obs_space.shape if hasattr(obs_space, 'shape') else obs_space["obs"].shape
            self.encoder = MlpEncoder(cfg, obs_shape)

        encoder_out = self.encoder.get_out_size()

        # optional, this is to avoid double processing measurements in q network
        self.measurements_head = None
        if "measurements" in obs_space.spaces:
            meas_dim = obs_space["measurements"].shape[0]
            self.measurements_head = nn.Sequential(
                nn.Linear(meas_dim, 64),
                nonlinearity(cfg),
                nn.Linear(64, 64),
                nonlinearity(cfg),
            )
            encoder_out += 64
        self.encoder_out_size = encoder_out

        # Core
        self.use_rnn = getattr(cfg, 'use_rnn', False)
        if self.use_rnn:
            self.core = ModelCoreRNN(cfg, encoder_out)
        else:
            self.core = ModelCoreIdentity(cfg, encoder_out)
        core_out = self.core.get_out_size()

        # Decoder
        self.decoder = MlpDecoder(cfg, core_out)
        decoder_out = self.decoder.get_out_size()

        # Q-value heads
        self._setup_q_heads(decoder_out, action_space)

        log.info(f"QMixAgentNet: encoder={encoder_out}, core={core_out}, decoder={decoder_out}, q_heads={self.total_actions}")

    def _setup_q_heads(self, decoder_out: int, action_space: ActionSpace):
        import gymnasium as gym

        # Calculate action dimensions for each head
        if isinstance(action_space, gym.spaces.Discrete): # single head
            self.action_sizes = [action_space.n]
        elif isinstance(action_space, gym.spaces.Tuple): # multi discreet
            self.action_sizes = [s.n for s in action_space.spaces if hasattr(s, 'n')]

        self.total_actions = sum(self.action_sizes)
        self.num_heads = len(self.action_sizes)

        if self.num_heads == 1:
            # We dont use this, this is from sample factory, so no harm keeping it here
            log.warning("QMixAgentNet: this is single-head action space, this benchmark doesn't use it")
            self.q_head = nn.Linear(decoder_out, self.action_sizes[0])
            self.q_heads = None
        else:
            self.q_head = None
            self.q_heads = nn.ModuleList([nn.Linear(decoder_out, size) for size in self.action_sizes])

    def get_rnn_size(self) -> int:
        if self.use_rnn:
            if getattr(self.cfg, "rnn_type", "gru") != "gru":
                raise ValueError(f"QMixAgentNet only supports GRU rnn_type, got {self.cfg.rnn_type}")
            return self.cfg.rnn_size * self.cfg.rnn_num_layers
        return 0

    def encode(self, obs: TensorDict) -> Tensor:
        """
        :param obs: obs dct
        :returns: [batch, encoder_out_size]
        """
        if isinstance(obs, dict) and "obs" in obs:
            x = self.encoder(obs["obs"])
        else:
            x = self.encoder(obs)

        if self.measurements_head is not None and "measurements" in obs:
            meas = self.measurements_head(obs["measurements"].float())
            x = torch.cat([x, meas], dim=-1)

        return x

    def forward_decomposed(self, obs: TensorDict, rnn_states = None):
        """Modularize this to use in learner_qmix"""
        x = self.encode(obs)
        encoder_out = x

        x, new_rnn = self.core(x, rnn_states)

        x = self.decoder(x)

        if self.q_head is not None:
            q_values = self.q_head(x)
        else:
            q_values = torch.cat([head(x) for head in self.q_heads], dim=-1)

        return q_values, new_rnn, encoder_out

    def forward(self, obs: TensorDict, rnn_states: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param obs: obs dct
        :param rnn_states: [batch, rnn_size]
        :returns: (q_values, new_rnn)
        """
        q_values, new_rnn, _ = self.forward_decomposed(obs, rnn_states)
        return q_values, new_rnn

    def get_q_for_actions(self, q_values: Tensor, actions: Tensor) -> Tensor:
        """
        :param q_values: [batch, total_actions] from forward()
        :param actions: [batch]/[batch, num_heads] selected actions
        """
        actions = actions.long()

        if self.q_head is not None:
            if actions.dim() == 1: # [batch]
                return q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            else: # [batch, num_heads]
                return q_values.gather(1, actions.view(-1, 1)).squeeze(-1)
        else:
            if actions.dim() == 1:
                raise ValueError(
                    f"Multi-head Q-network received 1D actions (shape {actions.shape}), "
                    f"expected [batch, {self.num_heads}]"
                )
            total_q = torch.zeros(q_values.shape[0], device=q_values.device)
            offset = 0
            for head_idx, size in enumerate(self.action_sizes):
                head_q = q_values[:, offset:offset + size]
                head_action = actions[:, head_idx]
                total_q = total_q + head_q.gather(1, head_action.unsqueeze(-1)).squeeze(-1)
                offset += size
            return total_q


class VDNMixer(nn.Module):
    """
    VDN sum mixer.
    """

    def __init__(self, num_agents: int):
        super().__init__()
        self.num_agents = num_agents

    def forward(self, agent_qs: Tensor, state: Optional[Tensor] = None) -> Tensor:
        """
        :param agent_qs: [batch, num_agents]
        :param state: Ignored
        """
        return agent_qs.sum(dim=-1)


class QMixMixer(nn.Module):
    """
    For all dQ_tot/dQ_i >= 0 enforced by abs() on mixing weights: Q_tot = f(Q_1, ..., Q_n; s)
    https://arxiv.org/pdf/1803.11485 Figure 2

    https://github.com/pytorch/rl/blob/74fcb21404d89c2930292eefa42a4882e4117de9/torchrl/modules/models/multiagent.py#L951
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hypernet_hidden: int = 64, # Hypernetwork (HN) hidden size
    ):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # HN layer 1 weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, num_agents * embed_dim),
        )

        # HN layer 1 bias
        # No monotonicity constraint
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # HN layer 2 weights
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, embed_dim),
        )

        # HN layer 2 bias
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Init with small weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, agent_qs: Tensor, state: Tensor) -> Tensor:
        """
        :param agent_qs: [batch, num_agents]
        :param state: [batch, state_dim] (global)
        """
        batch_size = agent_qs.size(0)

        # First layer weights with monotonicity (abs + small epsilon)
        w1 = torch.abs(self.hyper_w1(state)) + 1e-8
        w1 = w1.view(batch_size, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)

        # Second layer weights
        w2 = torch.abs(self.hyper_w2(state)) + 1e-8
        w2 = w2.view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # Forward
        # [B, N] -> [B, 1, N]
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)

        # First layer: [B, 1, N] @ [B, N, E] + [B, 1, E] = [B, 1, E] + [B, 1, E] = [B, 1, E]
        hidden = torch.bmm(agent_qs, w1) + b1
        hidden = F.elu(hidden)

        # Second layer: [B, 1, E] @ [B, E, 1] + [B, 1, 1] -> [B, 1, 1]
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.squeeze(-1).squeeze(-1)


def make_mixer(cfg: Config, num_agents: int, state_dim: int, n_actions: int = 0, unit_dim: int = 0) -> nn.Module:
    mixer_type = getattr(cfg, 'mixer', 'qmix').lower()

    if mixer_type == 'vdn':
        log.info(f"Using VDN mixer for {num_agents} agents")
        return VDNMixer(num_agents)
    elif mixer_type == 'qmix':
        embed_dim = getattr(cfg, 'qmix_embed_dim', 32)
        hypernet_hidden = getattr(cfg, 'qmix_hypernet_hidden', 64)
        log.info(f"Using QMIX mixer for {num_agents} agents: state_dim={state_dim}, embed={embed_dim}, hypernet={hypernet_hidden}")
        return QMixMixer(num_agents, state_dim, embed_dim, hypernet_hidden)
    elif mixer_type == 'dmaq':
        from comrad.models.qplex_mixer import DMAQer
        log.info(f"Using QPLEX DMAQer mixer for {num_agents} agents: state_dim={state_dim}, n_actions={n_actions}")
        return DMAQer(cfg, num_agents, state_dim, n_actions)
    elif mixer_type == 'dmaq_qatten':
        from comrad.models.qplex_mixer import DMAQ_QattenMixer
        log.info(f"Using QPLEX DMAQ_QattenMixer for {num_agents} agents: state_dim={state_dim}, n_actions={n_actions}, unit_dim={unit_dim}")
        return DMAQ_QattenMixer(cfg, num_agents, state_dim, n_actions, unit_dim)
    else:
        raise ValueError(f"Wrong mixer type: {mixer_type}")


class QMixActorCritic(nn.Module):
    """
    Wrapper for QMIX/VDN for Sample Factory InferenceWorker (and ParameterClient)
    """

    def __init__(
        self,
        cfg: Config,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        num_agents: int = 2,
    ):
        super().__init__()
        self.cfg = cfg
        self.action_space = action_space
        self.num_agents = num_agents
        self.encoders = []

        from sample_factory.utils.normalize import ObservationNormalizer
        # Sample Factory inference pops action_mask before calling normalize_obs() and passes it separately
        # so the normalizer shouldnt require the key (dont use `self.obs_normalizer = ObservationNormalizer(obs_space, cfg)`)
        self.obs_normalizer = ObservationNormalizer(obs_space_without_action_mask(obs_space), cfg)

        # For compatibility, not used in Q learning
        from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace
        self.returns_normalizer = None
        if getattr(cfg, 'normalize_returns', True):
            self.returns_normalizer = RunningMeanStdInPlace((1,))
            self.returns_normalizer = torch.jit.script(self.returns_normalizer)

        # Q-value net
        self.agent_net = QMixAgentNet(cfg, obs_space, action_space)

        # Mixer net
        # Use encoder output size for state_dim instead of raw pixels to avoid too big HN dimensions
        # as we have visual input
        state_dim = self.agent_net.encoder_out_size * num_agents
        n_actions = self.agent_net.total_actions
        unit_dim = self.agent_net.encoder_out_size
        self.mixer = make_mixer(cfg, num_agents, state_dim, n_actions=n_actions, unit_dim=unit_dim)

        self._device = torch.device('cpu')
        self.last_action_distribution = None # Not used

    def forward(
        self,
        obs: TensorDict,
        rnn_states: Optional[Tensor] = None,
        *,
        action_mask: Optional[Tensor] = None,
        sample_actions: bool = True, # Q-values for argmax if false
        **kwargs,
    ) -> TensorDict:
        q_values, new_rnn = self.agent_net(obs, rnn_states)
        batch_size = q_values.shape[0]

        # Set masked actions to -inf so argmax avoids them and
        # QMIXlearner recomputes Q with agent_net.forward() directly instead of reading action_logits from the traj buffer
        if action_mask is not None:
            q_values = q_values.clone()
            q_values[action_mask == 0] = float('-inf')

        # This is not used in QMIX, here for the sake of sample factory
        # Previously `values = q_values.max(dim=-1)[0]` but I switched to this nicer version for prettier log
        values = torch.zeros(batch_size, device=q_values.device)
        offset = 0
        for s in self.agent_net.action_sizes:
            values += q_values[:, offset:offset + s].max(dim=-1)[0]
            offset += s

        num_heads = self.agent_net.num_heads
        # actions/log_prob will be overwritten by inference_worker (epsilon-greedy)
        outputs = TensorDict({
            'actions': q_values.new_zeros((batch_size, num_heads), dtype=torch.long),
            'action_logits': q_values,
            'log_prob_actions': q_values.new_zeros(batch_size),
            'values': values,
            'new_rnn_states': new_rnn if new_rnn is not None else rnn_states,
        })

        return outputs

    def model_to_device(self, device):
        self._device = torch.device(device)
        self.to(device)

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        return self._device

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        if 'obs' in input_tensor_name.lower():
            return torch.float32
        return torch.float32

    def normalize_obs(self, obs: TensorDict) -> TensorDict:
        return self.obs_normalizer(obs)

    def summaries(self):
        return {}

    def get_rnn_size(self) -> int:
        return self.agent_net.get_rnn_size()


def make_qmix_actor_critic(
    cfg: Config,
    obs_space: ObsSpace,
    action_space: ActionSpace,
) -> QMixActorCritic:
    num_agents = getattr(cfg, 'num_agents', 2)
    return QMixActorCritic(cfg, obs_space, action_space, num_agents)
