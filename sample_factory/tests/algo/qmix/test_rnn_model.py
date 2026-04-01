from __future__ import annotations

import copy

import torch

from sample_factory.algo.learning.learner_qmix import QMixLearner
from sample_factory.algo.utils.tensor_dict import TensorDict
from comrad.models.qmix_model import QMixAgentNet, make_mixer

from .conftest import (
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
        action_dim = agent_net.total_actions
        unit_dim = agent_net.encoder_out_size
        mixer = make_mixer(cfg, num_agents, state_dim, action_dim, unit_dim)
        target_mixer = copy.deepcopy(mixer)

        learner = object.__new__(QMixLearner)
        learner.num_agents = num_agents
        learner.cfg = cfg
        learner.obs_normalizer = None
        learner.agent_net = agent_net
        learner.target_agent_net = target_net
        learner.mixer = mixer
        learner.target_mixer = target_mixer
        learner._is_qplex = getattr(cfg, 'mixer', 'qmix') in ('dmaq', 'dmaq_qatten')
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

    def test_qplex_qatten_sequential_backward(self):
        cfg = real_model_cfg(rnn_size=16, rnn_num_layers=1, mixer='dmaq_qatten')
        obs_space, act_space = small_obs_space(), compound_action_space()
        N = 2
        learner = self._make_learner(cfg, obs_space, act_space, N)

        B, T = 1, 2
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
        assert torch.isfinite(loss)
        assert td_summary.shape == (B,)

        loss.backward()

        attention_has_grad = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for name, param in learner.mixer.named_parameters()
            if name.startswith('attention_weight')
        )
        si_has_grad = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for name, param in learner.mixer.named_parameters()
            if name.startswith('si_weight')
        )
        assert attention_has_grad
        assert si_has_grad

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


class TestBatchedEncoderEquivalence:
    """Verify batched encoder produces identical results to per-timestep sequential encoding"""

    @staticmethod
    def _sequential_reference(obs, dones, rnn_states, agent_net, num_agents):
        """Reference implementation of per-timestep forward_decomposed (the old approach)"""
        sample_obs = obs["obs"] if isinstance(obs, dict) and "obs" in obs else obs
        batch_size = sample_obs.shape[0]
        obs_steps = sample_obs.shape[1]
        t_steps = dones.shape[1]

        def flatten_step_obs(step_obs):
            from sample_factory.algo.utils.tensor_dict import TensorDict as TD
            flat = TD()
            for key, val in step_obs.items():
                if isinstance(val, TD):
                    flat[key] = flatten_step_obs(val)
                else:
                    flat[key] = val.reshape(val.shape[0] * val.shape[1], *val.shape[2:])
            return flat

        rnn_flat = rnn_states.reshape(batch_size * num_agents, -1)
        q_values_list = []
        encoder_outs_list = []
        for t in range(obs_steps):
            step_obs = obs[:, t]
            flat_obs = flatten_step_obs(step_obs)
            q_flat, new_rnn_flat, encoder_flat = agent_net.forward_decomposed(flat_obs, rnn_flat)
            num_actions = q_flat.shape[-1]
            encoder_dim = encoder_flat.shape[-1]
            q_values_list.append(q_flat.view(batch_size, num_agents, num_actions))
            encoder_outs_list.append(encoder_flat.view(batch_size, num_agents, encoder_dim))
            if t < t_steps:
                done_mask = dones[:, t, :].reshape(batch_size * num_agents, 1).to(new_rnn_flat.dtype)
                new_rnn_flat = new_rnn_flat * (1.0 - done_mask)
            rnn_flat = new_rnn_flat
        return torch.stack(q_values_list, dim=1), torch.stack(encoder_outs_list, dim=1)

    def test_batched_matches_sequential_no_dones(self):
        cfg = real_model_cfg(rnn_size=32, rnn_num_layers=1)
        obs_space, act_space = small_obs_space(), compound_action_space()
        agent_net = QMixAgentNet(cfg, obs_space, act_space)
        agent_net.eval()

        B, T, N = 2, 4, 2
        obs = make_real_obs_batch(B, T + 1, N)
        dones = torch.zeros(B, T, N)
        rnn_states = torch.randn(B, N, agent_net.get_rnn_size())

        learner = object.__new__(QMixLearner)
        learner.num_agents = N

        q_batched, enc_batched = QMixLearner._sequential_agent_forward(
            learner, obs, dones, rnn_states, agent_net
        )
        q_ref, enc_ref = self._sequential_reference(obs, dones, rnn_states, agent_net, N)

        assert torch.allclose(q_batched, q_ref, atol=1e-5), \
            f"Q-value max diff: {(q_batched - q_ref).abs().max()}"
        assert torch.allclose(enc_batched, enc_ref, atol=1e-5), \
            f"Encoder output max diff: {(enc_batched - enc_ref).abs().max()}"

    def test_batched_matches_sequential_with_dones(self):
        cfg = real_model_cfg(rnn_size=16, rnn_num_layers=1)
        obs_space, act_space = small_obs_space(), compound_action_space()
        agent_net = QMixAgentNet(cfg, obs_space, act_space)
        agent_net.eval()

        B, T, N = 2, 5, 2
        obs = make_real_obs_batch(B, T + 1, N)
        dones = torch.zeros(B, T, N)
        dones[0, 1, 0] = 1.0  # agent 0 dies at t=1
        dones[1, 3, 1] = 1.0  # agent 1 dies at t=3
        rnn_states = torch.randn(B, N, agent_net.get_rnn_size())

        learner = object.__new__(QMixLearner)
        learner.num_agents = N

        q_batched, enc_batched = QMixLearner._sequential_agent_forward(
            learner, obs, dones, rnn_states, agent_net
        )
        q_ref, enc_ref = self._sequential_reference(obs, dones, rnn_states, agent_net, N)

        assert torch.allclose(q_batched, q_ref, atol=1e-5), \
            f"Q-value max diff: {(q_batched - q_ref).abs().max()}"
        assert torch.allclose(enc_batched, enc_ref, atol=1e-5), \
            f"Encoder output max diff: {(enc_batched - enc_ref).abs().max()}"

    def test_batched_matches_sequential_multi_layer_gru(self):
        cfg = real_model_cfg(rnn_size=16, rnn_num_layers=2)
        obs_space, act_space = small_obs_space(), compound_action_space()
        agent_net = QMixAgentNet(cfg, obs_space, act_space)
        agent_net.eval()

        B, T, N = 1, 3, 2
        obs = make_real_obs_batch(B, T + 1, N)
        dones = torch.zeros(B, T, N)
        dones[0, 0, 1] = 1.0
        rnn_states = torch.randn(B, N, agent_net.get_rnn_size())

        learner = object.__new__(QMixLearner)
        learner.num_agents = N

        q_batched, enc_batched = QMixLearner._sequential_agent_forward(
            learner, obs, dones, rnn_states, agent_net
        )
        q_ref, enc_ref = self._sequential_reference(obs, dones, rnn_states, agent_net, N)

        assert torch.allclose(q_batched, q_ref, atol=1e-5), \
            f"Q-value max diff: {(q_batched - q_ref).abs().max()}"
        assert torch.allclose(enc_batched, enc_ref, atol=1e-5), \
            f"Encoder output max diff: {(enc_batched - enc_ref).abs().max()}"

    def test_batched_gradients_flow_correctly(self):
        """Verify that backward through batched encoder produces valid gradients."""
        cfg = real_model_cfg(rnn_size=16, rnn_num_layers=1)
        obs_space, act_space = small_obs_space(), compound_action_space()
        agent_net = QMixAgentNet(cfg, obs_space, act_space)

        B, T, N = 1, 3, 2
        obs = make_real_obs_batch(B, T + 1, N)
        dones = torch.zeros(B, T, N)
        rnn_states = torch.zeros(B, N, agent_net.get_rnn_size())

        learner = object.__new__(QMixLearner)
        learner.num_agents = N

        q_values, enc_outs = QMixLearner._sequential_agent_forward(
            learner, obs, dones, rnn_states, agent_net
        )
        loss = q_values.sum()
        loss.backward()

        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in agent_net.encoder.parameters()
        )
        core_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in agent_net.core.parameters()
        )
        assert encoder_has_grad, "Encoder should receive gradients through batched path"
        assert core_has_grad, "RNN core should receive gradients through sequential path"
