"""Test encode()+forward_head() vs forward_decomposed() equivalence and flatten_rnn_parameters() effects."""
import copy

import pytest
import torch
import torch.nn as nn

from sample_factory.utils.typing import Config


def _make_minimal_config(**overrides):
    defaults = dict(
        rnn_size=64,
        rnn_num_layers=1,
        rnn_type="gru",
        use_rnn=True,
        encoder_type="conv",
        encoder_subtype="convnet_simple",
        encoder_extra_fc_layers=0,
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        nonlinearity="relu",
        policy_init_gain=1.0,
        use_spectral_norm=False,
        adaptive_stddev=False,
    )
    defaults.update(overrides)
    return Config(defaults)


class SimpleQNet(nn.Module):
    def __init__(self, obs_dim=16, hidden_dim=64, num_actions=14):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.core = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self.decoder = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, num_actions)
        self.encoder_out_size = hidden_dim
        self.use_rnn = True
        self._rnn_size = hidden_dim

    def get_rnn_size(self):
        return self._rnn_size

    def flatten_rnn_parameters(self):
        self.core.flatten_parameters()

    def encode(self, obs):
        return self.encoder(obs)

    def forward_head(self, encoded, rnn_states=None):
        x = encoded.unsqueeze(0)
        h = rnn_states.unsqueeze(0) if rnn_states is not None else None
        x, new_h = self.core(x, h)
        x = x.squeeze(0)
        new_h = new_h.squeeze(0)
        x = self.decoder(x)
        q = self.q_head(x)
        return q, new_h

    def forward_decomposed(self, obs, rnn_states=None):
        enc = self.encode(obs)
        encoder_out = enc
        x = enc.unsqueeze(0)
        h = rnn_states.unsqueeze(0) if rnn_states is not None else None
        x, new_h = self.core(x, h)
        x = x.squeeze(0)
        new_h = new_h.squeeze(0)
        x = self.decoder(x)
        q = self.q_head(x)
        return q, new_h, encoder_out


class TestForwardPathEquivalence:
    @pytest.fixture
    def model_and_data(self):
        torch.manual_seed(42)
        model = SimpleQNet(obs_dim=16, hidden_dim=64, num_actions=14)
        model.eval()
        batch_size = 8
        obs = torch.randn(batch_size, 16)
        rnn_states = torch.randn(batch_size, 64)
        return model, obs, rnn_states

    def test_forward_values_identical(self, model_and_data):
        model, obs, rnn = model_and_data
        q1, rnn1, enc1 = model.forward_decomposed(obs, rnn)
        enc2 = model.encode(obs)
        q2, rnn2 = model.forward_head(enc2, rnn)

        assert torch.equal(q1, q2)
        assert torch.equal(rnn1, rnn2)
        assert torch.equal(enc1, enc2)

    def test_gradients_identical(self, model_and_data):
        model, obs, rnn = model_and_data

        model.zero_grad()
        q1, _, _ = model.forward_decomposed(obs, rnn)
        q1.sum().backward()
        grads1 = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        model.zero_grad()
        enc2 = model.encode(obs)
        q2, _ = model.forward_head(enc2, rnn)
        q2.sum().backward()
        grads2 = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        assert set(grads1.keys()) == set(grads2.keys())
        for name in grads1:
            assert torch.equal(grads1[name], grads2[name])

    def test_sequential_loop_identical(self, model_and_data):
        model, obs, rnn = model_and_data
        T = 5
        batch_size = obs.shape[0]
        obs_seq = torch.randn(T, batch_size, 16)

        # Path 1: forward_decomposed loop
        rnn_state = rnn.clone()
        qs1, encs1 = [], []
        for t in range(T):
            q, rnn_state, enc = model.forward_decomposed(obs_seq[t], rnn_state)
            qs1.append(q)
            encs1.append(enc)

        # Path 2: encode + forward_head loop
        rnn_state = rnn.clone()
        qs2, encs2 = [], []
        for t in range(T):
            enc = model.encode(obs_seq[t])
            q, rnn_state = model.forward_head(enc, rnn_state)
            qs2.append(q)
            encs2.append(enc)

        for t in range(T):
            assert torch.equal(qs1[t], qs2[t])
            assert torch.equal(encs1[t], encs2[t])


class TestFlattenRNNParameters:
    @pytest.fixture
    def model_and_data(self):
        torch.manual_seed(42)
        model = SimpleQNet(obs_dim=16, hidden_dim=64, num_actions=14)
        batch_size = 8
        obs = torch.randn(batch_size, 16)
        rnn_states = torch.randn(batch_size, 64)
        return model, obs, rnn_states

    def test_flatten_does_not_change_forward(self, model_and_data):
        model, obs, rnn = model_and_data
        q1, rnn1, _ = model.forward_decomposed(obs, rnn)
        model.flatten_rnn_parameters()
        q2, rnn2, _ = model.forward_decomposed(obs, rnn)

        assert torch.equal(q1, q2)
        assert torch.equal(rnn1, rnn2)

    def test_flatten_does_not_change_gradients(self, model_and_data):
        model, obs, rnn = model_and_data

        model.zero_grad()
        q1, _, _ = model.forward_decomposed(obs, rnn)
        q1.sum().backward()
        grads_no_flatten = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        model.zero_grad()
        model.flatten_rnn_parameters()
        q2, _, _ = model.forward_decomposed(obs, rnn)
        q2.sum().backward()
        grads_with_flatten = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        for name in grads_no_flatten:
            assert torch.equal(grads_no_flatten[name], grads_with_flatten[name])

    def test_flatten_after_optimizer_step(self, model_and_data):
        model, obs, rnn = model_and_data
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        q, _, _ = model.forward_decomposed(obs, rnn)
        q.sum().backward()
        optimizer.step()

        with torch.no_grad():
            q_no_flatten, _, _ = model.forward_decomposed(obs, rnn)
        model.flatten_rnn_parameters()
        with torch.no_grad():
            q_with_flatten, _, _ = model.forward_decomposed(obs, rnn)

        assert torch.equal(q_no_flatten, q_with_flatten)

    def test_multi_step_training_divergence(self, model_and_data):
        """On CPU flatten is no-op. On GPU cuDNN may differ."""
        model_a, obs, rnn = model_and_data
        model_b = copy.deepcopy(model_a)

        opt_a = torch.optim.Adam(model_a.parameters(), lr=0.001)
        opt_b = torch.optim.Adam(model_b.parameters(), lr=0.001)

        T = 5
        obs_seq = torch.randn(T, obs.shape[0], 16)
        actions = torch.randint(0, 14, (obs.shape[0], T))

        for _ in range(50):
            opt_a.zero_grad()
            rnn_a = rnn.clone()
            total_q_a = torch.tensor(0.0)
            for t in range(T):
                q, rnn_a, _ = model_a.forward_decomposed(obs_seq[t], rnn_a)
                total_q_a = total_q_a + q.gather(1, actions[:, t:t+1]).squeeze(1).sum()
            total_q_a.backward()
            opt_a.step()

            opt_b.zero_grad()
            model_b.flatten_rnn_parameters()
            rnn_b = rnn.clone()
            total_q_b = torch.tensor(0.0)
            for t in range(T):
                enc = model_b.encode(obs_seq[t])
                q, rnn_b = model_b.forward_head(enc, rnn_b)
                total_q_b = total_q_b + q.gather(1, actions[:, t:t+1]).squeeze(1).sum()
            total_q_b.backward()
            opt_b.step()

        max_diff = max(
            torch.max(torch.abs(pa.data - pb.data)).item()
            for (_, pa), (_, pb) in zip(model_a.named_parameters(), model_b.named_parameters())
        )
        assert max_diff == 0.0, f"Models diverged: max_param_diff={max_diff:.2e}"


class TestFlattenRNNGPU:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_gpu_flatten_gradient_difference(self):
        torch.manual_seed(42)
        device = torch.device("cuda")
        model = SimpleQNet(obs_dim=16, hidden_dim=64, num_actions=14).to(device)
        obs = torch.randn(8, 16, device=device)
        rnn = torch.randn(8, 64, device=device)

        model.zero_grad()
        q1, _, _ = model.forward_decomposed(obs, rnn)
        q1.sum().backward()
        grads_no_flatten = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        model.zero_grad()
        model.flatten_rnn_parameters()
        q2, _, _ = model.forward_decomposed(obs, rnn)
        q2.sum().backward()
        grads_with_flatten = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        for name in grads_no_flatten:
            diff = torch.max(torch.abs(grads_no_flatten[name] - grads_with_flatten[name])).item()
            if diff > 0:
                print(f"GPU gradient diff for {name}: {diff:.2e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_gpu_multi_step_divergence(self):
        torch.manual_seed(42)
        device = torch.device("cuda")
        model_a = SimpleQNet().to(device)
        model_b = copy.deepcopy(model_a)
        opt_a = torch.optim.Adam(model_a.parameters(), lr=0.001)
        opt_b = torch.optim.Adam(model_b.parameters(), lr=0.001)

        rnn = torch.randn(8, 64, device=device)
        T = 5
        obs_seq = torch.randn(T, 8, 16, device=device)
        actions = torch.randint(0, 14, (8, T), device=device)

        for _ in range(50):
            opt_a.zero_grad()
            rnn_a = rnn.clone()
            total_q = torch.tensor(0.0, device=device)
            for t in range(T):
                q, rnn_a, _ = model_a.forward_decomposed(obs_seq[t], rnn_a)
                total_q = total_q + q.gather(1, actions[:, t:t+1]).squeeze(1).sum()
            total_q.backward()
            opt_a.step()

            opt_b.zero_grad()
            model_b.flatten_rnn_parameters()
            rnn_b = rnn.clone()
            total_q = torch.tensor(0.0, device=device)
            for t in range(T):
                enc = model_b.encode(obs_seq[t])
                q, rnn_b = model_b.forward_head(enc, rnn_b)
                total_q = total_q + q.gather(1, actions[:, t:t+1]).squeeze(1).sum()
            total_q.backward()
            opt_b.step()

        max_diff = max(
            torch.max(torch.abs(pa.data - pb.data)).item()
            for (_, pa), (_, pb) in zip(model_a.named_parameters(), model_b.named_parameters())
        )
        print(f"GPU multi-step max param divergence: {max_diff:.2e}")


class TestFlattenParametersWeightSharing:
    """flatten_parameters() breaks ParameterServer weight sharing by replacing .data with views into a new buffer."""

    def test_flatten_breaks_state_dict_sharing(self):
        gru = nn.GRU(32, 64, 1)
        state_dict = gru.state_dict()
        assert gru.weight_ih_l0.data_ptr() == state_dict['weight_ih_l0'].data_ptr()

        # Simulate GPU flatten_parameters
        all_weights = [gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0]
        flat = torch.zeros(sum(w.numel() for w in all_weights))
        offset = 0
        for w in all_weights:
            n = w.numel()
            flat[offset:offset+n].copy_(w.data.view(-1))
            w.data = flat[offset:offset+n].view_as(w)
            offset += n

        assert gru.weight_ih_l0.data_ptr() != state_dict['weight_ih_l0'].data_ptr()

        with torch.no_grad():
            gru.weight_ih_l0.data.add_(torch.ones_like(gru.weight_ih_l0))

        diff = (gru.weight_ih_l0.data - state_dict['weight_ih_l0']).abs().max().item()
        assert diff > 0.5

    def test_no_flatten_preserves_state_dict_sharing(self):
        gru = nn.GRU(32, 64, 1)
        state_dict = gru.state_dict()

        with torch.no_grad():
            gru.weight_ih_l0.data.add_(torch.ones_like(gru.weight_ih_l0))

        diff = (gru.weight_ih_l0.data - state_dict['weight_ih_l0']).abs().max().item()
        assert diff < 1e-7

    def test_full_model_weight_sharing_broken(self):
        model = SimpleQNet()
        state_dict = model.state_dict()
        gru_key = 'core.weight_ih_l0'
        assert model.core.weight_ih_l0.data_ptr() == state_dict[gru_key].data_ptr()

        all_weights = [model.core.weight_ih_l0, model.core.weight_hh_l0,
                       model.core.bias_ih_l0, model.core.bias_hh_l0]
        flat = torch.zeros(sum(w.numel() for w in all_weights))
        offset = 0
        for w in all_weights:
            n = w.numel()
            flat[offset:offset+n].copy_(w.data.view(-1))
            w.data = flat[offset:offset+n].view_as(w)
            offset += n

        assert model.core.weight_ih_l0.data_ptr() != state_dict[gru_key].data_ptr()
        enc_key = 'encoder.0.weight'
        assert model.encoder[0].weight.data_ptr() == state_dict[enc_key].data_ptr()

        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(4, 16)
        h = torch.zeros(1, 4, 64)
        q, _, _ = model.forward_decomposed(x, h.squeeze(0))
        q.sum().backward()
        opt.step()

        gru_diff = (model.core.weight_ih_l0.data - state_dict[gru_key]).abs().max().item()
        enc_diff = (model.encoder[0].weight.data - state_dict[enc_key]).abs().max().item()
        assert gru_diff > 0.001
        assert enc_diff < 1e-7

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_gpu_flatten_parameters_breaks_sharing(self):
        device = torch.device("cuda")
        gru = nn.GRU(32, 64, 1).to(device)
        state_dict = gru.state_dict()
        sd_ptr = state_dict['weight_ih_l0'].data_ptr()

        gru.flatten_parameters()
        new_ptr = gru.weight_ih_l0.data_ptr()
        assert new_ptr != sd_ptr

        with torch.no_grad():
            gru.weight_ih_l0.data.add_(torch.ones_like(gru.weight_ih_l0))
        diff = (gru.weight_ih_l0.data - state_dict['weight_ih_l0']).abs().max().item()
        assert diff > 0.5
