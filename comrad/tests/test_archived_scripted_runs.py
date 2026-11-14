import importlib.util
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from sample_factory.algo.utils.context import global_model_factory, reset_global_context
from sample_factory.algo.utils.make_env import NonBatchedDictObservationsWrapper
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic

from comrad.models.mappo_model import make_mappo_actor_critic
from comrad.models.qmix_model import make_qmix_actor_critic
from comrad.train import parse_args, prepare_cfg_for_training, register_model_factory, register_vizdoom_components
from comrad.utils import doom_utils

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = REPO_ROOT / "train_dir" / "very_difficult_scripted"
ARCHIVED_RUNS = [
    ("lava_maze_scripted_observer", "IPPO", "04_IPPO_env_lava_maze_scripted_observer_s.r.alp_1.0_see_154"),
    ("stealth_labyrinth_scripted_scout", "IDQN", "04_IDQN_env_stealth_labyrinth_scripted_scout_see_154"),
]
SCRIPTED_SCENARIOS = [spec.name for spec in doom_utils.DOOM_ENVS if spec.num_scripted_players > 0]
ALGOS = ["APPO", "IPPO", "MAPPO", "HAPPO", "IDQN", "QMIX", "VDN", "QPLEX"]


@pytest.fixture(autouse=True)
def comrad_components():
    register_vizdoom_components()
    yield
    reset_global_context()


def _heldout_module():
    spec = importlib.util.spec_from_file_location("heldout_wad_evaluation", REPO_ROOT / "scripts" / "heldout_wad_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _archive_run(tmp_path: Path, scenario: str, algo: str, num_agents: int = 2) -> Path:
    """config.json as written by a pre-split run, which stored the physical player count."""
    run_dir = tmp_path / f"00_{algo}_env_{scenario}_see_154"
    run_dir.mkdir(parents=True)
    cfg = parse_args(
        argv=[f"--algo={algo}", f"--env={scenario}", f"--experiment={run_dir.name}", f"--train_dir={tmp_path}"],
        evaluation=True,
    )
    payload = {key: value for key, value in vars(cfg).items() if key != "cli_args"}
    payload["num_agents"] = num_agents
    (run_dir / "config.json").write_text(json.dumps(payload, default=str), encoding="utf-8")
    return run_dir


def _load_for_eval(run_dir: Path, scenario: str, algo: str):
    cfg = parse_args(
        argv=[f"--algo={algo}", f"--env={scenario}", f"--experiment={run_dir.name}", f"--train_dir={run_dir.parent}"],
        evaluation=True,
    )
    return load_from_checkpoint(cfg)


class _FakeDoomPlayer(gym.Env):
    """Stands in for VizdoomEnvMultiplayer so the real per-player wrapper stack defines the observation space."""

    def __init__(self, action_space, env_spec_file, player_id=None, **kwargs):
        self.action_space = action_space
        self.player_id = player_id
        self.screen_w, self.screen_h = 160, 120
        self.calc_observation_space()

    def calc_observation_space(self):
        self.observation_space = gym.spaces.Box(0, 255, (self.screen_h, self.screen_w, 3), dtype=np.uint8)


class _FakeMultiAgentEnv(gym.Env):
    def __init__(self, num_agents, make_env_func, **kwargs):
        self.num_agents = num_agents
        self.is_multiagent = True
        self.envs = [make_env_func(player_id=i) for i in range(num_agents)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space


def _make_env(monkeypatch, cfg, scenario):
    import comrad.envs.multiagent.doom_multiagent as player_module
    import comrad.envs.multiagent.doom_multiagent_wrapper as mae_module

    monkeypatch.setattr(player_module, "VizdoomEnvMultiplayer", _FakeDoomPlayer)
    monkeypatch.setattr(mae_module, "MultiAgentEnv", _FakeMultiAgentEnv)
    cfg.num_bots = 0
    cfg.num_humans = 0
    cfg.wandb_record_every = 0
    env = doom_utils.make_doom_multiplayer_env(doom_utils.doom_env_by_name(scenario), cfg=cfg, env_config=None)
    return env if isinstance(env.observation_space, gym.spaces.Dict) else NonBatchedDictObservationsWrapper(env)


def _pre_split_factory(algo: str, num_players: int):
    """Factory register_model_factory chose when cfg.num_agents held the physical player count."""
    if num_players <= 1:
        return None
    if algo in ("QMIX", "VDN", "QPLEX"):
        return make_qmix_actor_critic
    if algo == "HAPPO":
        from comrad.models.happo_model import make_happo_actor_critic

        return make_happo_actor_critic
    return make_mappo_actor_critic


def test_resolver_maps_only_archived_physical_count_on_scripted_specs():
    resolve = doom_utils.resolve_archived_num_agents
    assert resolve({"env": "lava_maze_scripted_observer", "num_agents": 2}) == 1
    assert resolve({"env": "stealth_labyrinth_scripted_scout_batch", "num_agents": 2}) == 1
    assert resolve({"env": "lava_maze_scripted_observer", "num_agents": 1}) == 1
    assert resolve({"env": "lava_maze_scripted_observer", "num_agents": -1}) == -1
    assert resolve({"env": "lava_maze", "num_agents": 2}) == 2
    assert resolve({"env": "not_a_doom_env", "num_agents": 2}) == 2
    assert resolve({"num_agents": 2}) == 2
    assert doom_utils.resolve_archived_cfg({"env": "lava_maze_scripted_observer"}) == {
        "env": "lava_maze_scripted_observer", "observation_clock_version": 1
    }


@pytest.mark.parametrize("scenario, algo", [(s, a) for s, a, _ in ARCHIVED_RUNS])
def test_archived_scripted_run_resolves_on_every_load_path(tmp_path, monkeypatch, scenario, algo):
    run_dir = _archive_run(tmp_path, scenario, algo)

    assert _load_for_eval(run_dir, scenario, algo).num_agents == 1

    train_cfg = parse_args(
        argv=[
            f"--algo={algo}",
            f"--env={scenario}",
            f"--experiment={run_dir.name}",
            f"--train_dir={tmp_path}",
            "--restart_behavior=resume",
        ]
    )
    assert prepare_cfg_for_training(train_cfg).num_agents == 1

    heldout = _heldout_module()
    monkeypatch.setattr(heldout, "configure_batch_env_and_agents", lambda cfg: None)
    policy = heldout.PolicySpec(
        scenario=scenario,
        algorithm=algo,
        run_dir=str(run_dir),
        checkpoint=str(run_dir / "checkpoint_p0" / "checkpoint_000000001_100000000.pth"),
        checkpoint_steps=100_000_000,
        config_sha256="x",
        checkpoint_sha256="y",
        train_seed=154,
    )
    assert heldout.build_eval_cfg(policy, tmp_path / "batch", device="cpu", deterministic=False).num_agents == 1


@pytest.mark.parametrize("scenario, algo", [(s, a) for s, a, _ in ARCHIVED_RUNS])
def test_archived_scripted_run_builds_the_pre_split_model(tmp_path, monkeypatch, scenario, algo):
    cfg = _load_for_eval(_archive_run(tmp_path, scenario, algo), scenario, algo)
    env = _make_env(monkeypatch, cfg, scenario)
    assert env.num_agents == cfg.num_agents == 1

    register_model_factory(cfg)
    assert global_model_factory().make_actor_critic_func is make_mappo_actor_critic
    model = create_actor_critic(cfg, env.observation_space, env.action_space)

    archived_cfg = _load_for_eval(_archive_run(tmp_path / "raw", scenario, algo), scenario, algo)
    archived_cfg.num_agents = 2
    pre_split = make_mappo_actor_critic(archived_cfg, env.observation_space, env.action_space)
    assert type(model) is type(pre_split)
    assert {k: v.shape for k, v in model.state_dict().items()} == {k: v.shape for k, v in pre_split.state_dict().items()}


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("spec", doom_utils.DOOM_ENVS, ids=lambda spec: spec.name)
def test_model_factory_matches_pre_split_selection(spec, algo):
    cfg = parse_args(argv=[f"--algo={algo}", f"--env={spec.name}"], evaluation=True)
    cfg.num_agents = doom_utils.get_num_agents(cfg, spec.name)
    default_factory = global_model_factory().make_actor_critic_func

    register_model_factory(cfg)

    expected = _pre_split_factory(algo, spec.num_players) or default_factory
    assert global_model_factory().make_actor_critic_func is expected


@pytest.mark.parametrize("scenario", SCRIPTED_SCENARIOS)
def test_mappo_critic_on_scripted_variant_stays_sized_by_learners(monkeypatch, scenario):
    cfg = parse_args(argv=["--algo=MAPPO", f"--env={scenario}"], evaluation=True)
    cfg.num_agents = doom_utils.get_num_agents(cfg, scenario)
    env = _make_env(monkeypatch, cfg, scenario)
    register_model_factory(cfg)
    model = create_actor_critic(cfg, env.observation_space, env.action_space)
    assert model.n_agents == env.num_agents == 1


@pytest.mark.parametrize("scenario, algo, run_name", ARCHIVED_RUNS)
def test_local_archived_checkpoint_loads_strictly(monkeypatch, scenario, algo, run_name):
    run_dir = ARCHIVE_ROOT / run_name
    checkpoints = sorted((run_dir / "checkpoint_p0").glob("checkpoint_*.pth"))
    if not (run_dir / "config.json").is_file() or not checkpoints:
        pytest.skip(f"archived run {run_dir} not available")

    cfg = _load_for_eval(run_dir, scenario, algo)
    assert cfg.num_agents == 1
    env = _make_env(monkeypatch, cfg, scenario)
    register_model_factory(cfg)
    model = create_actor_critic(cfg, env.observation_space, env.action_space)
    state = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=True)
