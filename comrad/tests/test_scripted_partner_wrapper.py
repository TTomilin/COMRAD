import gymnasium as gym
import numpy as np
import pytest

from comrad.envs.doom_params import default_doom_cfg
from comrad.utils import doom_utils
from comrad.utils.doom_utils import DoomSpec
from comrad.wrappers.scripted_partner import NoopPartnerPolicy, ScriptedPartnerWrapper, zero_action
from comrad.wrappers.shared_reward import SharedRewardWrapper


ACTION_SPACE = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(2), gym.spaces.Discrete(5)))


class FakeMultiAgentEnv(gym.Env):
    """Minimal stand-in for MultiAgentEnv: list I/O, auto-reset on all(dones)."""

    def __init__(self, done_at=None, learner_only_done_at=None, **kwargs):
        self.num_agents = 2
        self.is_multiagent = True
        self.action_space = ACTION_SPACE
        self.observation_space = gym.spaces.Box(0, 255, shape=(2, 2, 3), dtype=np.uint8)
        self.done_at = done_at
        self.learner_only_done_at = learner_only_done_at
        self.t = 0
        self.episode = 0
        self.received_actions = []
        self.reset_calls = 0

    def _obs(self, agent):
        return np.full((2, 2, 3), 10 * agent + self.episode, dtype=np.uint8)

    def reset(self, **kwargs):
        self.reset_calls += 1
        self.t = 0
        return [self._obs(0), self._obs(1)], [{"agent": 0, "t": 0}, {"agent": 1, "t": 0}]

    def step(self, actions):
        assert len(actions) == 2
        self.received_actions.append(list(actions))
        self.t += 1
        rewards = [1.0, 2.0]
        infos = [{"agent": 0, "t": self.t}, {"agent": 1, "t": self.t}]
        terminated = [False, False]
        if self.done_at is not None and self.t == self.done_at:
            terminated = [True, True]
        if self.learner_only_done_at is not None and self.t == self.learner_only_done_at:
            terminated = [True, False]
        obs = [self._obs(0), self._obs(1)]
        if all(terminated):
            self.episode += 1
            obs, reset_infos = self.reset()
            for i, reset_info in enumerate(reset_infos):
                infos[i]["reset_info"] = reset_info
        return obs, rewards, terminated, [False, False], infos


class RecordingPolicy:
    instances = []

    def __init__(self, action_space, marker=4):
        self.action_space = action_space
        self.marker = marker
        self.reset_calls = 0
        self.seen = []
        RecordingPolicy.instances.append(self)

    def reset(self):
        self.reset_calls += 1

    def act(self, obs, info):
        self.seen.append((obs, info))
        return (0, 0, self.marker)


def test_zero_action_for_tuple_space():
    assert zero_action(ACTION_SPACE) == (0, 0, 0)
    assert NoopPartnerPolicy(ACTION_SPACE).act(None, None) == (0, 0, 0)


@pytest.mark.parametrize("learner_index", [0, 1])
def test_wrapper_exposes_single_learner(learner_index):
    inner = FakeMultiAgentEnv()
    env = ScriptedPartnerWrapper(inner, learner_index=learner_index, partner_policy=RecordingPolicy)
    partner_index = 1 - learner_index

    assert env.num_agents == 1
    assert env.is_multiagent is True
    assert env.action_space == inner.action_space
    assert env.observation_space == inner.observation_space

    obs, infos = env.reset()
    assert len(obs) == 1 and len(infos) == 1
    assert infos[0]["agent"] == learner_index
    assert np.array_equal(obs[0], inner._obs(learner_index))
    assert env.partner.reset_calls == 1

    obs, rewards, terminated, truncated, infos = env.step([(2, 1, 3)])
    assert rewards == [[1.0, 2.0][learner_index]]
    assert infos[0]["agent"] == learner_index
    assert inner.received_actions[-1][learner_index] == (2, 1, 3)
    assert inner.received_actions[-1][partner_index] == (0, 0, 4)
    # partner saw its own obs/info from reset
    seen_obs, seen_info = env.partner.seen[-1]
    assert seen_info["agent"] == partner_index
    assert np.array_equal(seen_obs, inner._obs(partner_index))


def test_partner_kwargs_and_independent_instances():
    RecordingPolicy.instances.clear()
    a = ScriptedPartnerWrapper(FakeMultiAgentEnv(), learner_index=0, partner_policy=RecordingPolicy, partner_kwargs={"marker": 2})
    b = ScriptedPartnerWrapper(FakeMultiAgentEnv(), learner_index=0, partner_policy=RecordingPolicy, partner_kwargs={"marker": 2})
    assert a.partner is not b.partner
    assert a.partner.marker == 2


def test_joint_done_resets_partner_and_uses_reset_info():
    inner = FakeMultiAgentEnv(done_at=2)
    env = ScriptedPartnerWrapper(inner, learner_index=0, partner_policy=RecordingPolicy)
    env.reset()
    env.step([(0, 0, 0)])
    obs, _, terminated, _, infos = env.step([(0, 0, 0)])

    assert terminated == [True]
    assert "reset_info" in infos[0]
    assert env.partner.reset_calls == 2
    # partner's next decision is based on the new episode's reset info, not the terminal info
    env.step([(0, 0, 0)])
    _, seen_info = env.partner.seen[-1]
    assert seen_info == {"agent": 1, "t": 0}


def test_learner_only_done_forces_joint_reset():
    inner = FakeMultiAgentEnv(learner_only_done_at=1)
    env = ScriptedPartnerWrapper(inner, learner_index=0, partner_policy=RecordingPolicy)
    env.reset()
    _, _, terminated, _, infos = env.step([(0, 0, 0)])
    assert terminated == [True]
    assert inner.reset_calls == 2
    assert "reset_info" in infos[0]
    assert env.partner.reset_calls == 2


def test_rejects_non_two_agent_env():
    inner = FakeMultiAgentEnv()
    inner.num_agents = 3
    with pytest.raises(ValueError):
        ScriptedPartnerWrapper(inner, learner_index=0, partner_policy=NoopPartnerPolicy)


class _Marker(gym.Wrapper):
    def __init__(self, env, tag):
        super().__init__(env)
        self.tag = tag
        self.num_agents = getattr(env, "num_agents", 1)
        self.is_multiagent = True


def _make_multiplayer(monkeypatch, joint_wrappers, algo="MAPPO", shared_reward_alpha=1.0):
    import comrad.envs.multiagent.doom_multiagent_wrapper as mae_module

    monkeypatch.setattr(mae_module, "MultiAgentEnv", lambda **kwargs: FakeMultiAgentEnv())
    spec = DoomSpec(
        "fake_joint",
        "fake.cfg",
        ACTION_SPACE,
        num_agents=2,
        shared_reward_alpha=shared_reward_alpha,
        joint_wrappers=joint_wrappers,
    )
    cfg = default_doom_cfg(env="lava_maze")
    cfg.algo = algo
    cfg.num_bots = 0
    cfg.num_agents = -1
    cfg.num_humans = 0
    cfg.wandb_record_every = 0
    return doom_utils.make_doom_multiplayer_env(spec, cfg=cfg, env_config=None)


def test_joint_wrappers_applied_in_order_before_shared_reward(monkeypatch):
    env = _make_multiplayer(
        monkeypatch,
        [(_Marker, {"tag": "first"}), (_Marker, {"tag": "second"})],
    )
    assert isinstance(env, SharedRewardWrapper)
    assert isinstance(env.env, _Marker) and env.env.tag == "second"
    assert isinstance(env.env.env, _Marker) and env.env.env.tag == "first"
    assert isinstance(env.env.env.env, FakeMultiAgentEnv)


def test_on_policy_single_learner_gets_benchmark_team_reward(monkeypatch):
    env = _make_multiplayer(
        monkeypatch,
        [(ScriptedPartnerWrapper, {"learner_index": 0, "partner_policy": NoopPartnerPolicy})],
        algo="IPPO",
    )
    # shared reward is blended while both players are visible, then the partner is hidden
    assert isinstance(env, ScriptedPartnerWrapper)
    assert isinstance(env.env, SharedRewardWrapper)
    assert isinstance(env.env.env, FakeMultiAgentEnv)
    assert env.num_agents == 1
    env.reset()
    _, rewards, *_ = env.step([(0, 0, 0)])
    assert rewards == [3.0]  # alpha=1, sum: learner reward = 1.0 + partner 2.0, as for benchmark IPPO agents


def test_off_policy_single_learner_keeps_individual_reward(monkeypatch):
    env = _make_multiplayer(
        monkeypatch,
        [(ScriptedPartnerWrapper, {"learner_index": 1, "partner_policy": NoopPartnerPolicy})],
        algo="IDQN",
    )
    assert isinstance(env, ScriptedPartnerWrapper)
    assert isinstance(env.env, FakeMultiAgentEnv)
    env.reset()
    _, rewards, *_ = env.step([(0, 0, 0)])
    assert rewards == [2.0]


def test_no_joint_wrappers_keeps_previous_stack(monkeypatch):
    env = _make_multiplayer(monkeypatch, None)
    assert isinstance(env, SharedRewardWrapper)
    assert isinstance(env.env, FakeMultiAgentEnv)


def test_happo_agent_id_patch_applied_after_joint_wrapper(monkeypatch):
    env = _make_multiplayer(monkeypatch, [(_Marker, {"tag": "x"})], algo="HAPPO")
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert env.observation_space["agent_id"].shape == (2,)


SCRIPTED_SCENARIOS = [spec.name for spec in doom_utils.DOOM_ENVS if spec.num_scripted_players > 0]


def _make_scenario(monkeypatch, scenario, algo="MAPPO", num_agents=-1):
    """Real config + env factory path for a registered scenario; only the ViZDoom MultiAgentEnv is faked."""
    import comrad.envs.multiagent.doom_multiagent_wrapper as mae_module
    from comrad.train import configure_batch_env_and_agents

    spawned = {}

    def fake_multi_agent_env(**kwargs):
        spawned["num_players"] = kwargs["num_agents"]
        inner = FakeMultiAgentEnv()
        inner.num_agents = kwargs["num_agents"]
        return inner

    monkeypatch.setattr(mae_module, "MultiAgentEnv", fake_multi_agent_env)
    cfg = default_doom_cfg(algo=algo, env=scenario)
    cfg.num_agents = num_agents
    cfg.num_bots = 0
    cfg.num_humans = 0
    cfg.wandb_record_every = 0
    configure_batch_env_and_agents(cfg)
    env = doom_utils.make_doom_multiplayer_env(doom_utils.doom_env_by_name(scenario), cfg=cfg, env_config=None)
    return cfg, env, spawned["num_players"]


def test_scripted_variants_declare_their_scripted_player():
    assert set(SCRIPTED_SCENARIOS) == {
        "lava_maze_scripted_observer",
        "lava_maze_scripted_navigator",
        "stealth_labyrinth_scripted_scout",
        "stealth_labyrinth_scripted_gunner",
    }
    for name in SCRIPTED_SCENARIOS:
        spec = doom_utils.doom_env_by_name(name)
        assert (spec.num_players, spec.num_scripted_players, spec.num_agents) == (2, 1, 1)


@pytest.mark.parametrize("scenario", SCRIPTED_SCENARIOS)
def test_scripted_variant_spawns_both_players_but_exposes_one_learner(monkeypatch, scenario):
    cfg, env, num_players = _make_scenario(monkeypatch, scenario)
    assert num_players == 2
    assert cfg.num_agents == env.num_agents == 1


@pytest.mark.parametrize("scenario", ["lava_maze", "stealth_labyrinth", "stag_hunt_arena"])
def test_benchmark_scenarios_keep_two_learners(monkeypatch, scenario):
    cfg, env, num_players = _make_scenario(monkeypatch, scenario)
    assert num_players == cfg.num_agents == env.num_agents == 2


def test_explicit_learner_count_mismatch_raises(monkeypatch):
    with pytest.raises(ValueError, match="num_agents"):
        _make_scenario(monkeypatch, "lava_maze_scripted_observer", num_agents=2)


@pytest.mark.parametrize("scenario", SCRIPTED_SCENARIOS)
def test_mappo_centralized_critic_matches_wrapped_learner_count(monkeypatch, scenario):
    import torch
    from comrad.models.mappo_model import MAPPOActorCritic
    from comrad.train import register_model_factory, register_vizdoom_models
    from sample_factory.algo.utils.context import global_model_factory
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.model.model_utils import get_rnn_size

    factory = global_model_factory()
    monkeypatch.setattr(factory, "make_actor_critic_func", factory.make_actor_critic_func)
    monkeypatch.setattr(factory, "make_model_encoder_func", factory.make_model_encoder_func)

    cfg, env, _ = _make_scenario(monkeypatch, scenario, algo="MAPPO")
    register_vizdoom_models()
    register_model_factory(cfg)
    obs_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 255, (3, cfg.res_h, cfg.res_w), np.uint8)})
    model = create_actor_critic(cfg, obs_space, doom_utils.doom_env_by_name(scenario).action_space)

    assert isinstance(model, MAPPOActorCritic) and model.use_centralized_critic
    assert model.n_agents == env.num_agents == 1
    for batch in (1, 3):
        obs = {"obs": torch.zeros(batch, 3, cfg.res_h, cfg.res_w)}
        result = model(obs, torch.zeros(batch, get_rnn_size(cfg)))
        assert result["values"].shape == (batch,)


def test_mappo_centralized_critic_rejects_partial_agent_groups(monkeypatch):
    import torch
    from comrad.train import register_model_factory, register_vizdoom_models
    from sample_factory.algo.utils.context import global_model_factory
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.model.model_utils import get_rnn_size

    factory = global_model_factory()
    monkeypatch.setattr(factory, "make_actor_critic_func", factory.make_actor_critic_func)
    monkeypatch.setattr(factory, "make_model_encoder_func", factory.make_model_encoder_func)

    cfg = default_doom_cfg(algo="MAPPO", env="lava_maze")
    cfg.num_agents = 2
    register_vizdoom_models()
    register_model_factory(cfg)
    obs_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 255, (3, cfg.res_h, cfg.res_w), np.uint8)})
    model = create_actor_critic(cfg, obs_space, doom_utils.doom_env_by_name("lava_maze").action_space)
    with pytest.raises(ValueError, match="not a multiple"):
        model({"obs": torch.zeros(3, 3, cfg.res_h, cfg.res_w)}, torch.zeros(3, get_rnn_size(cfg)))
