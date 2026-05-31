from __future__ import annotations

import pytest
import torch
import gymnasium as gym
from types import SimpleNamespace

import sample_factory.algo.utils.shared_buffers as shared_buffers
from sample_factory.algo.sampling.inference_worker import InferenceWorker
from sample_factory.algo.learning.learner_qmix import QMixLearner
from sample_factory.algo.utils.joint_replay_buffer import JointReplayBuffer
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict
from comrad.models.qmix_model import QMixAgentNet, QMixActorCritic
from comrad.utils.doom_utils import DOOM_ENVS


def _make_agent_net_multi_head():
    import gymnasium as gym
    from sample_factory.utils.attr_dict import AttrDict
    cfg = AttrDict({
        'encoder_conv_architecture': 'convnet_simple',
        'encoder_conv_mlp_layers': [],
        'encoder_extra_fc_layers': 0,
        'hidden_size': 32,
        'nonlinearity': 'relu',
        'use_rnn': False,
        'rnn_type': 'gru',
        'rnn_num_layers': 1,
        'decoder_mlp_layers': [],
    })
    obs_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 1, shape=(3, 64, 64))})
    action_space = gym.spaces.Tuple((
        gym.spaces.Discrete(3),
        gym.spaces.Discrete(2),
        gym.spaces.Discrete(2),
    ))
    return QMixAgentNet(cfg, obs_space, action_space)


def _make_actor_critic_multi_head():
    import gymnasium as gym
    from sample_factory.utils.attr_dict import AttrDict
    cfg = AttrDict({
        'encoder_conv_architecture': 'convnet_simple',
        'encoder_conv_mlp_layers': [],
        'encoder_extra_fc_layers': 0,
        'hidden_size': 32,
        'nonlinearity': 'relu',
        'use_rnn': False,
        'rnn_type': 'gru',
        'rnn_num_layers': 1,
        'decoder_mlp_layers': [],
        'normalize_input': False,
        'normalize_returns': False,
        'obs_subtract_mean': 0.0,
        'obs_scale': 1.0,
        'num_agents': 2,
        'mixer': 'qmix',
        'qmix_embed_dim': 32,
        'qmix_hypernet_hidden': 64,
    })
    obs_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 1, shape=(3, 64, 64))})
    action_space = gym.spaces.Tuple((
        gym.spaces.Discrete(3),
        gym.spaces.Discrete(2),
        gym.spaces.Discrete(2),
    ))
    return QMixActorCritic(cfg, obs_space, action_space, num_agents=2)


def _make_actor_critic_with_action_mask_and_input_norm():
    cfg = AttrDict({
        'encoder_conv_architecture': 'convnet_simple',
        'encoder_conv_mlp_layers': [],
        'encoder_extra_fc_layers': 0,
        'hidden_size': 32,
        'nonlinearity': 'relu',
        'use_rnn': False,
        'rnn_type': 'gru',
        'rnn_num_layers': 1,
        'decoder_mlp_layers': [],
        'normalize_input': True,
        'normalize_input_keys': None,
        'normalize_returns': False,
        'obs_subtract_mean': 0.0,
        'obs_scale': 255.0,
        'num_agents': 2,
        'mixer': 'qmix',
        'qmix_embed_dim': 32,
        'qmix_hypernet_hidden': 64,
    })

    obs_space = gym.spaces.Dict({
        'action_mask': gym.spaces.Box(0.0, 1.0, shape=(7,), dtype=float),
        'obs': gym.spaces.Box(0, 255, shape=(3, 64, 64), dtype=int),
    })
    action_space = gym.spaces.Tuple((
        gym.spaces.Discrete(3),
        gym.spaces.Discrete(2),
        gym.spaces.Discrete(2),
    ))
    return QMixActorCritic(cfg, obs_space, action_space, num_agents=2)


class TestValuesGlobalMaxVsSumOfHeadMax:

    def test_values_should_equal_sum_of_per_head_max(self):
        ac = _make_actor_critic_multi_head()
        ac.eval()
        obs = TensorDict({"obs": torch.randn(4, 3, 64, 64)})
        rnn = torch.zeros(4, ac.agent_net.core.get_out_size())

        with torch.no_grad():
            outputs = ac(obs, rnn)

        q_values = outputs['action_logits'] # [4, 3+2+2=7]
        values = outputs['values'] # [4]

        # Correct: sum of per-head maxima
        action_sizes = ac.agent_net.action_sizes # [3, 2, 2]
        correct_values = torch.zeros(4)
        offset = 0
        for size in action_sizes:
            head_q = q_values[:, offset:offset + size]
            correct_values += head_q.max(dim=-1)[0]
            offset += size

        assert torch.allclose(values, correct_values, atol=1e-5), (
            f"values (global max) = {values.tolist()}, "
            f"correct (sum-of-head-max) = {correct_values.tolist()}"
        )


class TestPlaceholderActionsShape:

    def test_actions_placeholder_should_match_num_heads(self):
        ac = _make_actor_critic_multi_head()
        ac.eval()
        obs = TensorDict({"obs": torch.randn(2, 3, 64, 64)})
        rnn = torch.zeros(2, ac.agent_net.core.get_out_size())

        with torch.no_grad():
            outputs = ac(obs, rnn)

        actions = outputs['actions']
        num_heads = ac.agent_net.num_heads  # 3

        assert actions.shape == (2, num_heads), (
            f"actions shape = {actions.shape}, expected (2, {num_heads})"
        )


class TestActionMaskNowApplied:

    def test_action_mask_should_affect_q_values(self):
        ac = _make_actor_critic_multi_head()
        ac.eval()
        obs = TensorDict({"obs": torch.randn(1, 3, 64, 64)})
        rnn = torch.zeros(1, ac.agent_net.core.get_out_size())

        with torch.no_grad():
            # No mask
            out_no_mask = ac(obs, rnn, action_mask=None)
            q_no_mask = out_no_mask['action_logits']

            # Create a mask that blocks the best action in the first head
            # Mask shape: [B, total_actions] = [1, 7]
            mask = torch.ones(1, 7)
            best_idx = q_no_mask[0, :3].argmax().item() # Best action in head 0
            mask[0, best_idx] = 0.0  # Block it

            # With mask — masked action's Q should be -inf
            out_with_mask = ac(obs, rnn, action_mask=mask)
            q_with_mask = out_with_mask['action_logits']

        # Masked action should have -inf Q-value
        assert q_with_mask[0, best_idx] == float('-inf'), (
            f"Masked action Q should be -inf, got {q_with_mask[0, best_idx]}"
        )
        # Other Q-values should be unchanged
        unmasked_idx = [i for i in range(7) if i != best_idx]
        assert torch.allclose(q_no_mask[0, unmasked_idx], q_with_mask[0, unmasked_idx], atol=1e-5), (
            "Unmasked Q-values should be identical"
        )


class TestActionMaskNotRequiredForNormalization:

    def test_normalize_obs_should_not_require_action_mask_key(self):
        # InferenceWorker pops 'action_mask' before calling normalize_obs() and passes it separately.
        # QMIX/QPLEX should not crash when 'action_mask' is absent from obs dict.
        from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

        ac = _make_actor_critic_with_action_mask_and_input_norm()
        ac.eval()

        obs_without_mask = TensorDict({
            'obs': torch.randint(0, 255, (4, 3, 64, 64), dtype=torch.uint8),
        })

        normalized = prepare_and_normalize_obs(ac, obs_without_mask)
        assert 'obs' in normalized
        assert 'action_mask' not in normalized


class TestEpsilonFallbackRemoved:

    def test_fallback_code_raises_on_missing_global(self):
        """Verify that the inference_worker code path raises RuntimeError
        when global_env_steps_tensor is None (instead of silently falling back)."""
        import importlib
        import inspect
        src = inspect.getsource(
            importlib.import_module('sample_factory.algo.sampling.inference_worker')
        )
        # The old fallback `epsilon_schedule.step(num_samples)` should be gone
        assert 'epsilon_schedule.step(num_samples)' not in src, (
            "Local epsilon fallback `epsilon_schedule.step(num_samples)` still present in inference_worker"
        )
        # The specific error message for this case should be present
        assert 'global_env_steps_tensor is None' in src, (
            "Expected RuntimeError mentioning 'global_env_steps_tensor is None' in inference_worker"
        )

    def test_epsilon_random_actions_respect_flat_tuple_mask(self):
        worker = SimpleNamespace(action_space_d=[3, 2, 4])
        mask = torch.tensor(
            [
                [1, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )

        for _ in range(50):
            actions = InferenceWorker._sample_random_actions(worker, 2, torch.device("cpu"), mask)

            assert actions.shape == (2, 3)
            assert actions[0, 0].item() == 0
            assert actions[0, 2].item() == 0
            assert actions[1, 0].item() == 1
            assert actions[1, 1].item() == 0
            assert actions[1, 2].item() == 3


class TestSampleLock:

    def test_sample_should_acquire_lock(self):
        rb = JointReplayBuffer(
            capacity=8, num_agents=2, obs_space=None, action_space=None,
            device="cpu", share_memory=False, use_per=False,
        )
        # Add a transition
        joint = TensorDict({
            "obs": TensorDict({"obs": torch.zeros(2, 1)}),
            "next_obs": TensorDict({"obs": torch.zeros(2, 1)}),
            "actions": torch.zeros(2, 3, dtype=torch.long),
            "rewards": torch.tensor([0.5, 0.5]),
            "dones": torch.tensor([0.0, 0.0]),
            "time_outs": torch.tensor([0.0, 0.0]),
        })
        rb.add_joint(joint)

        # Wrap _lock with a tracking lock that records whether acquire() is called
        acquire_count = [0]
        original_lock = rb._lock

        class TrackingLock:
            def __enter__(self):
                acquire_count[0] += 1
                return original_lock.__enter__()
            def __exit__(self, *a):
                return original_lock.__exit__(*a)
            def acquire(self, *a, **kw):
                acquire_count[0] += 1
                return original_lock.acquire(*a, **kw)
            def release(self):
                return original_lock.release()
            def locked(self):
                return original_lock.locked()

        rb._lock = TrackingLock()
        acquire_before = acquire_count[0]
        rb.sample(1, "cpu")
        acquire_during_sample = acquire_count[0] - acquire_before

        assert acquire_during_sample > 0, (
            "sample() should acquire _lock for thread safety, but it doesn't"
        )

class TestRewardShapingMissesTeamTermination:

    @pytest.mark.xfail(
        reason="Per-agent reward shaping runs before MultiAgentEnv team termination",
        strict=True,
    )
    def test_surviving_agent_should_get_true_objective_on_team_death(self):
        """When agent A dies and wipe_when_one_die terminates agent B,
        agent B's wrapper should still set true_objective."""
        from comrad.wrappers.scenario_wrappers.pitfall_reward_shaping import DoomPitfallRewardShaping

        class FakeEnv:
            def __init__(self):
                self.observation_space = None
                self.action_space = None
                self._step_result = None

            def step(self, action):
                return self._step_result

            def reset(self, **kwargs):
                return {}, {"POSITION_X": 32.0, "DEAD": 0}

        # Agent B's wrapper (agent B is alive, agent A already died)
        fake_env = FakeEnv()
        wrapper = DoomPitfallRewardShaping(
            fake_env, goal_x=100.0, goal_reward=10.0,
        )
        wrapper.reset()

        fake_env._step_result = (
            {}, # obs
            0.0, # reward
            False, # terminated — wrapper sees False (agent B is alive)
            False, # truncated
            {"POSITION_X": 100.0, "DEAD": 0},  # At goal_x
        )
        _, _, _, _, infos = wrapper.step(0)
        # Goal flag is now set in the wrapper

        # Now agent A dies → wipe_when_one_die will terminate agent B.
        # But per-agent step runs BEFORE wipe_when_one_die.
        # Agent B's wrapper sees done=False (not yet team-terminated).
        fake_env._step_result = (
            {}, # obs
            0.0, # reward
            False, # terminated — wrapper sees False (team termination hasn't happened yet)
            False, # truncated
            {"POSITION_X": 100.0, "DEAD": 0},
        )
        _, _, _, _, infos = wrapper.step(0)

        # After MultiAgentEnv.wipe_when_one_die runs, agent B IS terminated.
        # But the wrapper already ran with done=False, so true_objective is NOT set
        # on this step. The wrapper would only set it on the NEXT reset or if
        # done=True were passed.
        assert "true_objective" in infos, (
            "Surviving agent should have true_objective set when team is terminated, "
            "but per-agent wrapper ran before team termination was applied"
        )


class TestSingleHeadBranchInactive:

    @pytest.mark.xfail(
        reason="All current Doom envs use multi-head Tuple action spaces",
        strict=True,
    )
    def test_any_doom_env_uses_single_head(self):
        single_head_envs = []
        for spec in DOOM_ENVS:
            action_space = spec.action_space
            if isinstance(action_space, gym.spaces.Discrete):
                single_head_envs.append(spec.name)
            elif isinstance(action_space, gym.spaces.Tuple) and len(action_space.spaces) == 1:
                single_head_envs.append(spec.name)

        assert len(single_head_envs) > 0, (
            f"No Doom env uses single-head action space. "
            f"All {len(DOOM_ENVS)} envs use multi-head Tuple spaces, "
            f"making the single-head branch in get_q_for_actions dead code."
        )


class _DummyParamServer:
    def init(self, *args, **kwargs):
        return None


class _DummyReplayBuffer:
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 0


class _TinyAgentNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.encoder_out_size = 4

    @staticmethod
    def get_rnn_size():
        return 1


class _TinyMixer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))


class _TinyQMixActorCritic(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.agent_net = _TinyAgentNet()
        self.mixer = _TinyMixer()
        self.obs_normalizer = None

    def model_to_device(self, device):
        self.to(device)


class TestQMixCheckpointSaving:
    def test_qmix_init_enables_checkpoint_saves(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shared_buffers, "policy_device", lambda cfg, policy_id: torch.device("cpu"))
        monkeypatch.setattr("sample_factory.algo.learning.learner_qmix.QMixActorCritic", _TinyQMixActorCritic)
        monkeypatch.setattr("sample_factory.algo.learning.learner_qmix.JointReplayBuffer", _DummyReplayBuffer)
        monkeypatch.setattr("sample_factory.algo.learning.learner_qmix.JointSequenceReplayBuffer", _DummyReplayBuffer)
        monkeypatch.setattr("sample_factory.algo.learning.learner_qmix.flatten_rnn_parameters", lambda module: None)

        cfg = AttrDict(
            seed=123,
            num_agents=2,
            use_rnn=False,
            mixer="qmix",
            replay_buffer_size=64,
            learning_rate=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.999,
            serial_mode=True,
            train_dir=str(tmp_path),
            experiment="qmix_save_smoke",
            load_checkpoint_kind="latest",
            keep_checkpoints=2,
        )
        env_info = SimpleNamespace(obs_space=None, action_space=gym.spaces.Discrete(3))
        learner = QMixLearner(
            cfg,
            env_info,
            policy_versions_tensor=torch.zeros(1, dtype=torch.int64),
            policy_id=0,
            param_server=_DummyParamServer(),
            global_env_steps_tensor=torch.zeros(1, dtype=torch.int64),
        )

        learner.init()
        checkpoint_dir = tmp_path / "qmix_save_smoke" / "checkpoint_p0"

        assert learner.is_initialized is True
        assert learner.save() is True
        assert list(checkpoint_dir.glob("checkpoint_*.pth"))

        assert learner.save_best(0, "true_objective", 123.0) is True
        assert list(checkpoint_dir.glob("best_*.pth"))
