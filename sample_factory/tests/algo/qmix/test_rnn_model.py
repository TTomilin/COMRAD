from __future__ import annotations

import copy

import gymnasium as gym
import torch

from sample_factory.algo.learning.learner_qmix import QMixLearner
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict
from sf.doom.qmix_model import QMixAgentNet, make_mixer

from .conftest import (
    SumMixer,
    compound_action_space,
    make_real_obs_batch,
    real_model_cfg,
    single_action_space,
    small_obs_space,
)


class TestRealModelSequentialForward:

    def test_single_layer_gru_shapes(self):
        cfg = real_model_cfg(rnn_size=32, rnn_num_layers=1)
        obs_space, act_space = small_obs_space(), compound_action_space()
        agent_net = QMixAgentNet(cfg, obs_space, act_space)

        B, T, N = 2, 4, 2
        obs = make_real_obs_batch(B, T + 1, N)
        dones = torch.zeros(B, T, N)
        rnn_states = torch.zeros(B, N, agent_net.get_rnn_size())

        learner = object.__new__(QMixLearner)
        learner.num_agents = N

        q_values, enc_outs = QMixLearner._sequential_agent_forward(learner, obs, dones, rnn_states, agent_net)

        assert q_values.shape == (B, T + 1, N, sum(agent_net.action_sizes))
        assert enc_outs.shape == (B, T + 1, N, agent_net.encoder_out_size)

    def test_multi_layer_gru_shapes(self):
        cfg = real_model_cfg(rnn_size=16, rnn_num_layers=3)
        obs_space, act_space = small_obs_space(), compound_action_space()
        agent_net = QMixAgentNet(cfg, obs_space, act_space)

        assert agent_net.get_rnn_size() == 16 * 3  # multi-layer

        B, T, N = 2, 3, 2
        obs = make_real_obs_batch(B, T + 1, N)
        dones = torch.zeros(B, T, N)
        rnn_states = torch.zeros(B, N, agent_net.get_rnn_size())

        learner = object.__new__(QMixLearner)
        learner.num_agents = N

        q_values, enc_outs = QMixLearner._sequential_agent_forward(learner, obs, dones, rnn_states, agent_net)

        assert q_values.shape == (B, T + 1, N, sum(agent_net.action_sizes))
        assert enc_outs.shape == (B, T + 1, N, agent_net.encoder_out_size)

    def test_done_resets_hidden_state_real_gru(self):
        """With a real GRU, verify that a done at t=0 resets hidden state,
        producing different outputs than without the done."""
        cfg = real_model_cfg(rnn_size=16, rnn_num_layers=1)
        obs_space, act_space = small_obs_space(), compound_action_space()
        agent_net = QMixAgentNet(cfg, obs_space, act_space)
        agent_net.eval()

        B, T, N = 1, 2, 2
        obs = make_real_obs_batch(B, T + 1, N)
        rnn_states = torch.randn(B, N, agent_net.get_rnn_size())  # non-zero

        learner = object.__new__(QMixLearner)
        learner.num_agents = N

        # No dones
        dones_none = torch.zeros(B, T, N)
        q_no_done, _ = QMixLearner._sequential_agent_forward(learner, obs, dones_none, rnn_states, agent_net)

        # Agent 0 done at t=0 → hidden reset before t=1
        dones_reset = torch.zeros(B, T, N)
        dones_reset[0, 0, 0] = 1.0
        q_with_done, _ = QMixLearner._sequential_agent_forward(learner, obs, dones_reset, rnn_states, agent_net)

        # t=0 outputs should be identical (done not yet applied)
        assert torch.allclose(q_no_done[:, 0], q_with_done[:, 0], atol=1e-6)
        # t=1 agent 0 should differ (hidden was reset)
        assert not torch.allclose(q_no_done[:, 1, 0], q_with_done[:, 1, 0], atol=1e-4)
        # t=1 agent 1 should be the same (not reset)
        assert torch.allclose(q_no_done[:, 1, 1], q_with_done[:, 1, 1], atol=1e-6)


class TestRealModelLossComputation:

    def _make_learner(self, cfg, obs_space, act_space, num_agents=2):
        agent_net = QMixAgentNet(cfg, obs_space, act_space)
        target_net = copy.deepcopy(agent_net)
        state_dim = agent_net.encoder_out_size * num_agents
        mixer = make_mixer(cfg, num_agents, state_dim)
        target_mixer = copy.deepcopy(mixer)

        learner = object.__new__(QMixLearner)
        learner.num_agents = num_agents
        learner.cfg = cfg
        learner.obs_normalizer = None
        learner.agent_net = agent_net
        learner.target_agent_net = target_net
        learner.mixer = mixer
        learner.target_mixer = target_mixer
        return learner

    def test_qmix_loss_shapes_compound_actions(self):
        cfg = real_model_cfg(rnn_size=32, rnn_num_layers=1, mixer='qmix')
        obs_space, act_space = small_obs_space(), compound_action_space()
        N = 2
        learner = self._make_learner(cfg, obs_space, act_space, N)

        B, T = 3, 4
        batch = TensorDict({
            'obs': make_real_obs_batch(B, T + 1, N),
            # compound actions: [B, T, N, num_heads]
            'actions': torch.stack([
                torch.randint(0, 3, (B, T, N)),
                torch.randint(0, 2, (B, T, N)),
            ], dim=-1),
            'rewards': torch.randn(B, T, N),
            'dones': torch.zeros(B, T, N),
            'time_outs': torch.zeros(B, T, N),
            'rnn_states': torch.zeros(B, N, learner.agent_net.get_rnn_size()),
        })

        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert td_summary.shape == (B,)

    def test_vdn_loss_shapes(self):
        cfg = real_model_cfg(rnn_size=32, rnn_num_layers=1, mixer='vdn')
        obs_space, act_space = small_obs_space(), compound_action_space()
        N = 2
        learner = self._make_learner(cfg, obs_space, act_space, N)

        B, T = 2, 3
        batch = TensorDict({
            'obs': make_real_obs_batch(B, T + 1, N),
            'actions': torch.stack([
                torch.randint(0, 3, (B, T, N)),
                torch.randint(0, 2, (B, T, N)),
            ], dim=-1),
            'rewards': torch.randn(B, T, N),
            'dones': torch.zeros(B, T, N),
            'time_outs': torch.zeros(B, T, N),
            'rnn_states': torch.zeros(B, N, learner.agent_net.get_rnn_size()),
        })

        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        assert loss.dim() == 0
        assert td_summary.shape == (B,)

    def test_backward_flows_to_agent_net_and_mixer(self):
        cfg = real_model_cfg(rnn_size=16, rnn_num_layers=1, mixer='qmix')
        obs_space, act_space = small_obs_space(), compound_action_space()
        N = 2
        learner = self._make_learner(cfg, obs_space, act_space, N)

        B, T = 2, 3
        batch = TensorDict({
            'obs': make_real_obs_batch(B, T + 1, N),
            'actions': torch.stack([
                torch.randint(0, 3, (B, T, N)),
                torch.randint(0, 2, (B, T, N)),
            ], dim=-1),
            'rewards': torch.randn(B, T, N),
            'dones': torch.zeros(B, T, N),
            'time_outs': torch.zeros(B, T, N),
            'rnn_states': torch.zeros(B, N, learner.agent_net.get_rnn_size()),
        })

        loss, _ = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        loss.backward()

        # Mixer should receive gradients
        mixer_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in learner.mixer.parameters())
        assert mixer_has_grad, "Mixer parameters should receive gradients"

        # Agent net Q-head should receive gradients (not detached)
        q_heads = learner.agent_net.q_heads if learner.agent_net.q_heads is not None else [learner.agent_net.q_head]
        q_head_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                              for head in q_heads for p in head.parameters())
        assert q_head_has_grad, "Agent net Q-heads should receive gradients"

        # GRU core should receive gradients (backprop through time)
        core_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in learner.agent_net.core.parameters())
        assert core_has_grad, "GRU core should receive gradients via BPTT"

        # Target net should NOT receive gradients
        target_has_grad = any(p.grad is not None for p in learner.target_agent_net.parameters())
        assert not target_has_grad, "Target agent net must not receive gradients"

    def test_multi_layer_gru_full_loss(self):
        cfg = real_model_cfg(rnn_size=16, rnn_num_layers=2, mixer='qmix')
        obs_space, act_space = small_obs_space(), compound_action_space()
        N = 2
        learner = self._make_learner(cfg, obs_space, act_space, N)

        B, T = 2, 3
        batch = TensorDict({
            'obs': make_real_obs_batch(B, T + 1, N),
            'actions': torch.stack([
                torch.randint(0, 3, (B, T, N)),
                torch.randint(0, 2, (B, T, N)),
            ], dim=-1),
            'rewards': torch.randn(B, T, N),
            'dones': torch.zeros(B, T, N),
            'time_outs': torch.zeros(B, T, N),
            'rnn_states': torch.zeros(B, N, learner.agent_net.get_rnn_size()),
        })

        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        assert loss.dim() == 0
        assert td_summary.shape == (B,)
        # Should not error on backward with multi-layer GRU
        loss.backward()

    def test_single_action_space_loss(self):
        cfg = real_model_cfg(rnn_size=16, rnn_num_layers=1, mixer='qmix')
        obs_space = small_obs_space()
        act_space = single_action_space()
        N = 2
        learner = self._make_learner(cfg, obs_space, act_space, N)

        B, T = 2, 3
        batch = TensorDict({
            'obs': make_real_obs_batch(B, T + 1, N),
            # Single action: [B, T, N] (no head dim)
            'actions': torch.randint(0, 4, (B, T, N)),
            'rewards': torch.randn(B, T, N),
            'dones': torch.zeros(B, T, N),
            'time_outs': torch.zeros(B, T, N),
            'rnn_states': torch.zeros(B, N, learner.agent_net.get_rnn_size()),
        })

        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        assert loss.dim() == 0
        assert td_summary.shape == (B,)

    def test_with_dones_and_timeouts(self):
        cfg = real_model_cfg(rnn_size=16, rnn_num_layers=1, mixer='qmix')
        obs_space, act_space = small_obs_space(), compound_action_space()
        N = 2
        learner = self._make_learner(cfg, obs_space, act_space, N)

        B, T = 2, 4
        dones = torch.zeros(B, T, N)
        time_outs = torch.zeros(B, T, N)
        # Agent 0 terminates at t=1
        dones[0, 1, 0] = 1.0
        # Agent 1 times out at t=2
        dones[0, 2, 1] = 1.0
        time_outs[0, 2, 1] = 1.0

        batch = TensorDict({
            'obs': make_real_obs_batch(B, T + 1, N),
            'actions': torch.stack([
                torch.randint(0, 3, (B, T, N)),
                torch.randint(0, 2, (B, T, N)),
            ], dim=-1),
            'rewards': torch.randn(B, T, N),
            'dones': dones,
            'time_outs': time_outs,
            'rnn_states': torch.randn(B, N, learner.agent_net.get_rnn_size()),
        })

        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        assert loss.dim() == 0
        assert td_summary.shape == (B,)


class TestObsNormalizationRNN:

    def test_normalization_preserves_shape(self):
        from sample_factory.utils.normalize import ObservationNormalizer

        obs_space = small_obs_space()
        cfg = real_model_cfg(rnn_size=16, rnn_num_layers=1)
        cfg['normalize_input'] = True
        cfg['normalize_input_keys'] = ['obs']
        cfg['obs_subtract_mean'] = 0.0
        cfg['obs_scale'] = 1.0

        normalizer = ObservationNormalizer(obs_space, cfg)

        learner = object.__new__(QMixLearner)
        learner.num_agents = 2
        learner.cfg = cfg
        learner.obs_normalizer = normalizer
        learner.agent_net = QMixAgentNet(cfg, obs_space, compound_action_space())
        learner.target_agent_net = copy.deepcopy(learner.agent_net)
        state_dim = learner.agent_net.encoder_out_size * 2
        learner.mixer = make_mixer(cfg, 2, state_dim)
        learner.target_mixer = copy.deepcopy(learner.mixer)

        B, T, N = 2, 3, 2
        batch = TensorDict({
            'obs': make_real_obs_batch(B, T + 1, N),
            'actions': torch.stack([
                torch.randint(0, 3, (B, T, N)),
                torch.randint(0, 2, (B, T, N)),
            ], dim=-1),
            'rewards': torch.randn(B, T, N),
            'dones': torch.zeros(B, T, N),
            'time_outs': torch.zeros(B, T, N),
            'rnn_states': torch.zeros(B, N, learner.agent_net.get_rnn_size()),
        })

        # Should not raise any shape errors
        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        assert loss.dim() == 0
        assert td_summary.shape == (B,)


class TestMinimalRollout:

    def test_rollout_2_loss(self):
        cfg = real_model_cfg(rnn_size=16, rnn_num_layers=1)
        obs_space, act_space = small_obs_space(), compound_action_space()
        N = 2

        agent_net = QMixAgentNet(cfg, obs_space, act_space)
        target_net = copy.deepcopy(agent_net)
        state_dim = agent_net.encoder_out_size * N
        mixer = make_mixer(cfg, N, state_dim)
        target_mixer = copy.deepcopy(mixer)

        learner = object.__new__(QMixLearner)
        learner.num_agents = N
        learner.cfg = cfg
        learner.obs_normalizer = None
        learner.agent_net = agent_net
        learner.target_agent_net = target_net
        learner.mixer = mixer
        learner.target_mixer = target_mixer

        B, T = 2, 2  # minimum rollout
        batch = TensorDict({
            'obs': make_real_obs_batch(B, T + 1, N),
            'actions': torch.stack([
                torch.randint(0, 3, (B, T, N)),
                torch.randint(0, 2, (B, T, N)),
            ], dim=-1),
            'rewards': torch.randn(B, T, N),
            'dones': torch.zeros(B, T, N),
            'time_outs': torch.zeros(B, T, N),
            'rnn_states': torch.zeros(B, N, agent_net.get_rnn_size()),
        })

        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        assert loss.dim() == 0
        assert td_summary.shape == (B,)
        loss.backward()
