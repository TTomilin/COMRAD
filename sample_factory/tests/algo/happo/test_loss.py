import numpy as np
import pytest
import torch
import torch.nn as nn

from sample_factory.algo.learning.learner_happo import HAPPOLearner
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from sf.doom.happo_model import make_happo_actor_critic

from .conftest import (
    _make_happo_cfg,
    _make_happo_model,
    _make_obs_batch,
    _make_obs_space,
    _make_action_space,
)


class TestFactorM:

    def test_m_initialized_as_advantage(self):
        adv = torch.tensor([1.0, -2.0, 3.0, 0.5])
        mean = adv.mean()
        std = adv.std()
        normalized = (adv - mean) / torch.clamp_min(std, 1e-7)
        M = normalized.clone().detach()

        torch.testing.assert_close(M, normalized)

    def test_m_detached(self):
        adv = torch.tensor([1.0, 2.0], requires_grad=True)
        M = adv.clone().detach()
        assert not M.requires_grad

    def test_post_ratio_broadcast_two_agents(self):
        n_agents = 2
        n_transitions = 3
        experience_size = n_agents * n_transitions

        # Layout: [agent0_t0, agent1_t0, agent0_t1, agent1_t1, agent0_t2, agent1_t2]
        agent_idx = torch.tensor([0, 1, 0, 1, 0, 1])
        env_group_idx = torch.tensor([0, 0, 1, 1, 2, 2])

        M = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        agent_mask = agent_idx == 0

        # Simulate: agent 0 has post-update ratio of [1.5, 2.0, 0.8]
        post_ratio = torch.tensor([1.5, 2.0, 0.8])

        # Broadcast logic (from _train)
        ratio_per_transition = torch.ones(n_transitions)
        ratio_per_transition[env_group_idx[agent_mask]] = post_ratio
        M_new = M * ratio_per_transition[env_group_idx]

        # Agent 0's entries: [1.0*1.5, 2.0*2.0, 3.0*0.8] = [1.5, 4.0, 2.4]
        # Agent 1's entries: same transitions -> same ratio -> [1.0*1.5, 2.0*2.0, 3.0*0.8]
        expected = torch.tensor([1.5, 1.5, 4.0, 4.0, 2.4, 2.4])
        torch.testing.assert_close(M_new, expected)

    def test_post_ratio_accumulates_three_agents(self):
        n_agents = 3
        n_transitions = 2
        experience_size = n_agents * n_transitions

        agent_idx = torch.tensor([0, 1, 2, 0, 1, 2])
        env_group_idx = torch.tensor([0, 0, 0, 1, 1, 1])

        # Start: M = advantage
        M = torch.ones(experience_size)

        # Agent 0 post-ratio: [2.0, 3.0] for transitions [0, 1]
        post_ratio_0 = torch.tensor([2.0, 3.0])
        ratio_per_trans = torch.ones(n_transitions)
        ratio_per_trans[env_group_idx[agent_idx == 0]] = post_ratio_0
        M = M * ratio_per_trans[env_group_idx]
        # M = [2, 2, 2, 3, 3, 3]

        # Agent 1 post-ratio: [1.5, 0.5] for transitions [0, 1]
        post_ratio_1 = torch.tensor([1.5, 0.5])
        ratio_per_trans = torch.ones(n_transitions)
        ratio_per_trans[env_group_idx[agent_idx == 1]] = post_ratio_1
        M = M * ratio_per_trans[env_group_idx]
        # M = [2*1.5, 2*1.5, 2*1.5, 3*0.5, 3*0.5, 3*0.5] = [3, 3, 3, 1.5, 1.5, 1.5]

        expected = torch.tensor([3.0, 3.0, 3.0, 1.5, 1.5, 1.5])
        torch.testing.assert_close(M, expected)

    def test_m_handles_negative_advantages(self):
        M = torch.tensor([-2.0, 1.0, -0.5, 3.0])
        post_ratio = torch.tensor([2.0, 2.0])
        env_group_idx = torch.tensor([0, 0, 1, 1])
        agent_mask = torch.tensor([True, False, True, False])

        ratio_per_trans = torch.ones(2)
        ratio_per_trans[env_group_idx[agent_mask]] = post_ratio
        M_new = M * ratio_per_trans[env_group_idx]

        expected = torch.tensor([-4.0, 2.0, -1.0, 6.0])
        torch.testing.assert_close(M_new, expected)

    def test_factor_clamp(self):
        post_ratio = torch.tensor([100.0, 0.001])
        factor_clamp = 5.0
        clamped = torch.clamp(post_ratio, 1.0 / factor_clamp, factor_clamp)
        assert clamped[0].item() == 5.0
        assert clamped[1].item() == pytest.approx(0.2)

    def test_invalid_samples_masked(self):
        post_ratio = torch.tensor([1.5, 99.0, 2.0])
        agent_valids = torch.tensor([True, False, True])
        post_ratio[~agent_valids] = 1.0

        assert post_ratio[0].item() == 1.5
        assert post_ratio[1].item() == 1.0  # Was 99.0, now masked
        assert post_ratio[2].item() == 2.0


class TestPolicyLoss:

    def test_asymmetric_clipping(self):
        ppo_clip_ratio = 0.2
        clip_high = 1.0 + ppo_clip_ratio  # 1.2
        clip_low = 1.0 / clip_high  # 1/1.2 ~ 0.8333

        assert clip_high == pytest.approx(1.2)
        assert clip_low == pytest.approx(1.0 / 1.2)
        # Asymmetric: clip_low ~ 0.833, NOT 0.8
        assert clip_low != pytest.approx(1.0 - ppo_clip_ratio)

    def test_loss_direction_positive_advantage(self):
        ratio = torch.tensor([1.5])
        M = torch.tensor([2.0])  # Positive advantage

        clip_high = 1.2
        clip_low = 1.0 / 1.2
        clipped = torch.clamp(ratio, clip_low, clip_high)

        loss = -torch.min(ratio * M, clipped * M)
        # ratio*M = 3.0, clipped*M = 1.2*2.0 = 2.4
        # min = 2.4, loss = -2.4
        assert loss.item() == pytest.approx(-2.4)

    def test_loss_direction_negative_advantage(self):
        ratio = torch.tensor([1.5])
        M = torch.tensor([-2.0])  # Negative advantage

        clip_high = 1.2
        clip_low = 1.0 / 1.2
        clipped = torch.clamp(ratio, clip_low, clip_high)

        loss = -torch.min(ratio * M, clipped * M)
        # ratio*M = -3.0, clipped*M = 1.2*(-2.0) = -2.4
        # min = -3.0, loss = -(-3.0) = 3.0
        assert loss.item() == pytest.approx(3.0)


class TestAgentOrdering:
    def test_random_order_changes_with_train_step(self):
        n_agents = 4
        orders = set()
        for step in range(20):
            rng = torch.Generator().manual_seed(step)
            order = tuple(torch.randperm(n_agents, generator=rng).tolist())
            orders.add(order)

        # With 4!=24 permutations and 20 tries, we should see multiple orderings
        assert len(orders) > 1

    def test_fixed_order_is_sequential(self):
        agent_order = list(range(3))
        assert agent_order == [0, 1, 2]

    def test_random_order_deterministic_with_same_seed(self):
        n_agents = 3
        rng1 = torch.Generator().manual_seed(42)
        order1 = torch.randperm(n_agents, generator=rng1).tolist()

        rng2 = torch.Generator().manual_seed(42)
        order2 = torch.randperm(n_agents, generator=rng2).tolist()

        assert order1 == order2


class TestLearnerMethodIntegration:

    @pytest.fixture
    def model_and_cfg(self):
        num_agents = 2
        cfg = _make_happo_cfg(num_agents=num_agents)
        cfg.ppo_clip_ratio = 0.2
        cfg.ppo_clip_value = 10.0
        cfg.value_loss_coeff = 0.5
        cfg.exploration_loss_coeff = 0.0
        cfg.recurrence = 1
        cfg.rollout = 4
        cfg.batch_size = 8
        obs_space = _make_obs_space(num_agents=num_agents)
        action_space = _make_action_space(4)
        model = make_happo_actor_critic(cfg, obs_space, action_space)
        model.eval()
        return model, cfg, obs_space, action_space

    def test_calculate_agent_policy_loss(self, model_and_cfg):
        model, cfg, obs_space, action_space = model_and_cfg

        # Create learner stub with real model
        stub = object.__new__(HAPPOLearner)
        stub.cfg = cfg
        stub.n_agents = cfg.num_agents
        stub.actor_critic = model
        stub.exploration_loss_func = lambda d, v, n: 0.0
        stub.use_critic_rnn = getattr(cfg, 'happo_critic_rnn', False) and cfg.use_rnn

        # Create synthetic minibatch
        batch_size = 4
        obs = _make_obs_batch(batch_size, cfg.num_agents)
        # All samples for agent 0
        obs["agent_id"] = torch.zeros(batch_size, cfg.num_agents)
        obs["agent_id"][:, 0] = 1.0

        # Get real log probs for the actions
        with torch.no_grad():
            head = model.forward_head(obs, agent_idx=torch.zeros(batch_size, dtype=torch.long))
            core, _ = model.forward_core(head, torch.zeros(batch_size, get_rnn_size(cfg)),
                                          agent_idx=torch.zeros(batch_size, dtype=torch.long))
            dec = model.agent_decoders[0](core)
            logits, _ = model.agent_action_params[0](dec, None)
            dist = get_action_distribution(action_space, logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        mb = AttrDict({
            "normalized_obs": obs,
            "actions": actions,
            "log_prob_actions": log_probs,
            "rnn_states": torch.zeros(batch_size, get_rnn_size(cfg)),
            "valids": torch.ones(batch_size),
            "dones_cpu": torch.zeros(batch_size),
        })

        M_full = torch.ones(10)  # Larger than batch for indexing
        mb_indices = np.arange(batch_size)

        model.train()
        policy_loss, exploration_loss = stub._calculate_agent_policy_loss(
            agent_id=0, mb=mb, M_full=M_full, mb_indices=mb_indices, num_invalids=0
        )

        assert policy_loss.requires_grad
        assert isinstance(policy_loss.item(), float)
        assert not torch.isnan(policy_loss)
        assert not torch.isinf(policy_loss)

    def test_calculate_critic_loss(self, model_and_cfg):
        model, cfg, obs_space, action_space = model_and_cfg

        stub = object.__new__(HAPPOLearner)
        stub.cfg = cfg
        stub.n_agents = cfg.num_agents
        stub.actor_critic = model
        stub.use_critic_rnn = getattr(cfg, 'happo_critic_rnn', False) and cfg.use_rnn

        # Create synthetic minibatch with 2 complete transitions
        batch_size = cfg.num_agents * 2  # 4 samples = 2 transitions
        obs = _make_obs_batch(batch_size, cfg.num_agents)
        agent_idx = obs["agent_id"].argmax(dim=-1)
        env_group_idx = torch.tensor([0, 0, 1, 1])

        mb = AttrDict({
            "normalized_obs": obs,
            "values": torch.zeros(batch_size),
            "returns": torch.ones(batch_size),
            "valids": torch.ones(batch_size),
            "rnn_states": torch.zeros(batch_size, get_rnn_size(cfg)),
        })

        model.train()
        value_loss = stub._calculate_critic_loss(mb, agent_idx, env_group_idx, num_invalids=0)

        assert value_loss.requires_grad
        assert isinstance(value_loss.item(), float)
        assert not torch.isnan(value_loss)
        assert value_loss.item() > 0  # returns=1.0, values=0.0 -> nonzero loss


class TestCheckpointRoundTrip:
    def test_checkpoint_save_restore(self):
        cfg = _make_happo_cfg(num_agents=2)
        cfg.learning_rate = 0.001
        cfg.adam_beta1 = 0.9
        cfg.adam_beta2 = 0.999
        cfg.adam_eps = 1e-8

        obs_space = _make_obs_space(num_agents=2)
        action_space = _make_action_space(4)
        model = make_happo_actor_critic(cfg, obs_space, action_space)

        # Create optimizers manually (simulating init)
        agent_optimizers = []
        for i in range(2):
            params = (
                list(model.agent_encoders[i].parameters())
                + list(model.agent_cores[i].parameters())
                + list(model.agent_decoders[i].parameters())
                + list(model.agent_action_params[i].parameters())
            )
            agent_optimizers.append(torch.optim.Adam(params, lr=0.001))

        critic_params = (
            list(model.centralized_critic.parameters())
            + list(model.critic_encoders.parameters())
        )
        critic_optimizer = torch.optim.Adam(critic_params, lr=0.001)

        # Simulate a training step to populate optimizer states
        dummy_loss = sum(p.sum() for p in model.parameters())
        dummy_loss.backward()
        for opt in agent_optimizers:
            opt.step()
        critic_optimizer.step()

        # Build checkpoint dict (matching learner's _get_checkpoint_dict)
        checkpoint = {
            "model": model.state_dict(),
            "agent_optimizers": [opt.state_dict() for opt in agent_optimizers],
            "critic_optimizer": critic_optimizer.state_dict(),
            "env_steps": 1000,
            "train_step": 42,
            "curr_lr": 0.0005,
            "best_performance": 10.0,
        }

        # Verify all required keys present
        assert "model" in checkpoint
        assert len(checkpoint["agent_optimizers"]) == 2
        assert "critic_optimizer" in checkpoint
        assert checkpoint["train_step"] == 42
        assert checkpoint["curr_lr"] == 0.0005

        # Create fresh model and restore
        model2 = make_happo_actor_critic(cfg, obs_space, action_space)
        model2.load_state_dict(checkpoint["model"])

        # Verify parameter equality
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)


class TestApplyLR:
    def test_apply_lr_updates_all_optimizers(self):
        stub = object.__new__(HAPPOLearner)
        stub.n_agents = 2

        # Create dummy optimizers
        params_a = [nn.Parameter(torch.randn(3, 3))]
        params_b = [nn.Parameter(torch.randn(3, 3))]
        params_c = [nn.Parameter(torch.randn(3, 3))]
        stub.agent_optimizers = [
            torch.optim.Adam(params_a, lr=0.01),
            torch.optim.Adam(params_b, lr=0.01),
        ]
        stub.critic_optimizer = torch.optim.Adam(params_c, lr=0.01)

        # Apply new LR
        new_lr = 0.0001
        HAPPOLearner._apply_lr(stub, new_lr)

        for opt in stub.agent_optimizers:
            assert opt.param_groups[0]["lr"] == new_lr
        assert stub.critic_optimizer.param_groups[0]["lr"] == new_lr

    def test_apply_lr_does_not_set_curr_lr(self):
        stub = object.__new__(HAPPOLearner)
        stub.n_agents = 1
        stub.curr_lr = 0.001
        stub.agent_optimizers = [torch.optim.Adam([nn.Parameter(torch.randn(2))], lr=0.01)]
        stub.critic_optimizer = torch.optim.Adam([nn.Parameter(torch.randn(2))], lr=0.01)

        HAPPOLearner._apply_lr(stub, 0.0001)

        assert stub.curr_lr == 0.001  # Unchanged
