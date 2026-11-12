import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "scripts", REPO_ROOT / "experiments" / "iclr2027_rebuttal"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import eval_frozen_partner as efp  # noqa: E402
import generate_c2k6_frozen_partner_table as gen  # noqa: E402
from comrad.utils.doom_utils import doom_env_by_name  # noqa: E402
from experiments.common import CANONICAL_STEALTH_QMIX_REPLAY_BUFFER  # noqa: E402
from generate_kd6j_tables import SCENARIOS  # noqa: E402
from heldout_wad_evaluation import PolicySpec, WadManifestEntry  # noqa: E402

SCENARIO_NAMES = [scen for scen, _ in SCENARIOS]


def _heads(space):
    return [int(s.n) for s in space.spaces] if isinstance(space, gym.spaces.Tuple) else [int(space.n)]


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_partner_action_tensor_matches_every_action_space(scenario):
    space = doom_env_by_name(scenario).action_space
    heads = _heads(space)
    rng = np.random.default_rng(0)
    noop = efp.build_partner_action_tensor("noop", space, rng, None, torch.int64, torch.device("cpu"))
    assert noop.shape == (len(heads),) and noop.dtype == torch.int64 and torch.all(noop == 0)

    draws = torch.stack(
        [efp.build_partner_action_tensor("random", space, rng, None, torch.int64, torch.device("cpu")) for _ in range(400)]
    )
    assert draws.shape == (400, len(heads))
    for h, n in enumerate(heads):
        assert int(draws[:, h].min()) >= 0 and int(draws[:, h].max()) < n
        if n > 1:
            assert len(set(draws[:, h].tolist())) == n  # every option reachable

    # mask: allow only the last option of every head
    mask = np.concatenate([np.eye(n)[-1] for n in heads]).astype(np.float32)
    masked = efp.build_partner_action_tensor("random", space, rng, mask, torch.int64, torch.device("cpu"))
    assert masked.tolist() == [n - 1 for n in heads]

    with pytest.raises(ValueError):
        efp.build_partner_action_tensor("learned", space, rng, None, torch.int64, torch.device("cpu"))


class FakeTwoAgentEnv:
    """2-agent env: episodes last `ep_len` steps, objective = number of steps on which agent 1 chose head-0 action 0."""

    num_agents = 2

    def __init__(self, ep_len=3, goal_infos=False):
        self.ep_len = ep_len
        self.goal_infos = goal_infos
        self.t = 0
        self.resets = 0
        self.score = 0
        self.actions = []
        self.goal = (0, 0)

    def _obs(self):
        return {"obs": torch.randint(0, 255, (2, 3, 72, 128), dtype=torch.uint8)}

    def _infos(self):
        info = {"USER13": self.goal[0], "USER18": self.goal[1]} if self.goal_infos else {}
        return [dict(info), dict(info)]

    def reset(self):
        self.resets += 1
        self.t, self.score = 0, 0
        self.goal = (100 + self.resets, 0)
        return self._obs(), self._infos()

    def step(self, actions):
        actions = np.asarray(actions)
        assert actions.shape[0] == 2 and actions.dtype == np.int32
        self.actions.append(actions.copy())
        self.t += 1
        first = actions[1] if actions.ndim == 1 else actions[1, 0]
        self.score += int(first == 0)
        if self.goal_infos and self.t == 2:
            self.goal = (self.goal[0], 7)  # one genuine relocation per episode
        done = self.t >= self.ep_len
        infos = self._infos()
        for i in infos:
            i["true_objective"] = float(self.score)
        if done:
            self.t, self.score = 0, 0
            self.goal = (999, 999)  # new episode's goal: must not count as a relocation
            for i in infos:
                i["reset_info"] = {"USER13": 999, "USER18": 999} if self.goal_infos else {}
        flags = torch.tensor([done, done])
        return self._obs(), torch.zeros(2), flags, torch.zeros(2, dtype=torch.bool), infos


class RecordingModel:
    def __init__(self, model):
        self.model = model
        self.rnn_inputs = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, obs, rnn_states, **kwargs):
        assert obs["obs"].shape[0] == 2  # frozen model always sees both agents
        self.rnn_inputs.append(rnn_states.clone())
        return self.model(obs, rnn_states, **kwargs)


def _real_model(algo, scenario):
    from comrad.envs.doom_params import default_doom_cfg
    from comrad.train import register_model_factory, register_vizdoom_components
    from sample_factory.model.actor_critic import create_actor_critic

    register_vizdoom_components()
    cfg = default_doom_cfg(env=scenario)
    cfg.algo, cfg.num_agents = algo, 2
    if algo == "QMIX":
        cfg.mixer = "qmix"
    register_model_factory(cfg)
    obs_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 255, (3, 72, 128), np.uint8)})
    space = doom_env_by_name(scenario).action_space
    model = create_actor_critic(cfg, obs_space, space)
    model.eval()
    env_info = SimpleNamespace(all_discrete=True, action_space=space, gpu_actions=False)
    return cfg, model, env_info


def _policy(scenario, algo):
    return PolicySpec(scenario, algo, "run", "ckpt", 1, "c", "k", 42)


ENTRY = WadManifestEntry("coop_puzzle", "w", 1009, "w.wad", "sha", {})


@pytest.mark.parametrize("algo", ["MAPPO", "IPPO", "QMIX"])
def test_live_passes_override_only_partner_seat_and_never_leak_state(algo):
    cfg, model, env_info = _real_model(algo, "coop_puzzle")
    rec = RecordingModel(model)
    env = FakeTwoAgentEnv(ep_len=3)
    seen = {}
    for idx, (name, partner, seat) in enumerate(efp.PASS_SPECS):
        start_calls, start_steps = len(rec.rnn_inputs), len(env.actions)
        records = efp.run_pass(
            env, rec, env_info, cfg, _policy("coop_puzzle", algo), ENTRY, 2, torch.device("cpu"),
            pass_name=name, partner_policy=partner, partner_idx=seat, rng_seed=efp.pass_rng_seed(1009, idx),
        )
        assert len(records) == 2 and all(r["pass"] == name and r["partner_idx"] == seat for r in records)
        assert env.resets == idx + 1  # explicit reset per pass
        assert torch.count_nonzero(rec.rnn_inputs[start_calls]) == 0  # zero RNN state at pass start
        acts = np.stack(env.actions[start_steps:])
        assert acts.shape == (6, 2, 3)  # 2 episodes x 3 steps, [agents, heads]
        seen[name] = acts
        if partner == "noop":
            assert np.all(acts[:, seat] == 0)
            if seat == 1:  # fake objective counts agent-1 head-0 zeros: 3 per 3-step episode
                assert [r["true_objective"] for r in records] == [3.0, 3.0]
        if partner != "learned":
            assert records[0]["learner_idx"] == 1 - seat
    heads = _heads(env_info.action_space)
    for name in ("random_seat0", "random_seat1"):
        seat = int(name[-1])
        assert all(0 <= a < n for row in seen[name] for a, n in zip(row[seat], heads))


def test_random_partner_is_deterministic_per_wad_and_pass():
    cfg, model, env_info = _real_model("MAPPO", "coop_puzzle")
    out = []
    for _ in range(2):
        env = FakeTwoAgentEnv(ep_len=4)
        efp.run_pass(
            env, model, env_info, cfg, _policy("coop_puzzle", "MAPPO"), ENTRY, 2, torch.device("cpu"),
            pass_name="random_seat1", partner_policy="random", partner_idx=1, rng_seed=efp.pass_rng_seed(1009, 3),
        )
        out.append(np.stack(env.actions)[:, 1])
    assert np.array_equal(out[0], out[1])
    assert efp.pass_rng_seed(1009, 3) != efp.pass_rng_seed(2027, 3) != efp.pass_rng_seed(1009, 4)


def test_goal_counter_resets_between_episodes_and_passes():
    cfg, model, env_info = _real_model("MAPPO", "lava_maze")
    env = FakeTwoAgentEnv(ep_len=3, goal_infos=True)
    for idx, (name, partner, seat) in enumerate(efp.PASS_SPECS[:3]):
        records = efp.run_pass(
            env, model, env_info, cfg, _policy("lava_maze", "MAPPO"), ENTRY, 3, torch.device("cpu"),
            pass_name=name, partner_policy=partner, partner_idx=seat, rng_seed=idx,
        )
        # exactly one relocation per episode: no phantom count from auto-reset or the explicit pass reset
        assert [r["goals_reached"] for r in records] == [1, 1, 1]
        assert all(r["true_objective"] == 1.0 for r in records)


class HiddenResetEnv(FakeTwoAgentEnv):
    def step(self, actions):
        out = list(super().step(actions))
        if self.t == 2:
            for info in out[4]:
                info["_hidden_reset_count"] = 1
        return tuple(out)


def test_hidden_crash_reset_aborts_the_pass():
    cfg, model, env_info = _real_model("MAPPO", "coop_puzzle")
    crashes = []
    with pytest.raises(efp.HiddenResetError):
        efp.run_pass(
            HiddenResetEnv(ep_len=4), model, env_info, cfg, _policy("coop_puzzle", "MAPPO"), ENTRY, 1,
            torch.device("cpu"), pass_name="noop_seat1", partner_policy="noop", partner_idx=1, rng_seed=0,
            crash_log=crashes,
        )
    assert len(crashes) == efp.MAX_CRASHED_EPISODES_PER_PASS


class OneCrashEnv(FakeTwoAgentEnv):
    """Crashes once, mid-way through the first episode; the wrapper's recovery leaves the game at a fresh start."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.crashed = False

    def step(self, actions):
        out = list(super().step(actions))
        if not self.crashed and self.t == 2:
            self.crashed = True
            self.t, self.score = 0, 0
            for info in out[4]:
                info["_hidden_reset_count"] = 1
        return tuple(out)


def test_single_crash_discards_only_the_interrupted_episode():
    cfg, model, env_info = _real_model("MAPPO", "coop_puzzle")
    rec = RecordingModel(model)
    env = OneCrashEnv(ep_len=3)
    crashes = []
    records = efp.run_pass(
        env, rec, env_info, cfg, _policy("coop_puzzle", "MAPPO"), ENTRY, 2, torch.device("cpu"),
        pass_name="noop_seat1", partner_policy="noop", partner_idx=1, rng_seed=0, crash_log=crashes,
    )
    assert crashes == [{"wad_seed": 1009, "pass": "noop_seat1", "episode_index": 0, "episode_step": 2}]
    assert env.resets == 2  # pass-start reset + explicit reset after the crash
    assert torch.count_nonzero(rec.rnn_inputs[2]) == 0  # RNN memory zeroed after the crash
    # 2 partial steps discarded + 2 complete 3-step episodes; each complete episode scores 3 under a no-op seat 1
    assert len(env.actions) == 2 + 6
    assert [r["true_objective"] for r in records] == [3.0, 3.0]
    assert [r["episode_index"] for r in records] == [0, 1]


def test_run_pass_rejects_invalid_seat():
    cfg, model, env_info = _real_model("MAPPO", "coop_puzzle")
    with pytest.raises(ValueError):
        efp.run_pass(
            FakeTwoAgentEnv(), model, env_info, cfg, _policy("coop_puzzle", "MAPPO"), ENTRY, 1, torch.device("cpu"),
            pass_name="noop_seat2", partner_policy="noop", partner_idx=2, rng_seed=0,
        )


def _run(scenario, algo, seed, both, noop0, noop1, rand0, rand1, wads=(1009, 2027)):
    def recs(value):
        return [{"true_objective": value, "wad_seed": w} for w in wads]

    passes = {
        "both_learned": both, "noop_seat0": noop0, "noop_seat1": noop1, "random_seat0": rand0, "random_seat1": rand1,
    }
    return {
        "scenario": scenario, "algorithm": algo, "train_seed": seed, "checkpoint_steps": 100,
        "passes": {k: {"records": recs(v)} for k, v in passes.items()},
    }


def test_table_selects_top_seed_averages_seats_and_clips():
    runs = [
        _run("ammo_carrier", "IPPO", 42, 4200.0, 0.0, 840.0, 420.0, 420.0),
        _run("ammo_carrier", "IPPO", 68, 6300.0, 0.0, 1680.0, 0.0, 840.0),
        _run("ammo_carrier", "QMIX", 81, 8400.0, 0.0, 0.0, 0.0, 0.0),  # not the Table 3 winner
        _run("armory_siege", "QMIX", 68, 20.0, -9.1, -5.0, -3.0, 140.0),
    ]
    chosen = gen.select_top_runs(runs)
    assert chosen["ammo_carrier"]["train_seed"] == 68
    row = gen.scenario_row("ammo_carrier", chosen["ammo_carrier"], 840.0, runs[:2])
    assert row["pct"]["both_learned"] == pytest.approx(75.0)
    assert row["pct"]["partner_noop"] == pytest.approx(10.0)  # (0 + 1680) / 2 / 8400
    assert row["pct"]["partner_random"] == pytest.approx(5.0)
    assert row["category"] == "drops_to_random"
    assert row["retention"] == pytest.approx({"noop": 0.0, "random": -5.0 / 65.0})
    assert [c["train_seed"] for c in row["candidates"]] == [42, 68]

    siege = gen.scenario_row("armory_siege", chosen["armory_siege"], -9.1, [chosen["armory_siege"]])
    assert siege["pct"]["partner_noop"] == 0.0  # negative raw -> 0%
    assert siege["pct"]["partner_random"] == pytest.approx(68.5)  # (−3 + 140) / 2 = 68.5 raw on a 100 ceiling
    assert siege["random_policy_pct"] == 0.0


def test_table_ties_and_classification_bands():
    runs = [_run("lava_maze", "MAPPO", 68, 0.0, 0, 0, 0, 0), _run("lava_maze", "MAPPO", 42, 0.0, 0, 0, 0, 0)]
    assert gen.select_top_runs(runs)["lava_maze"]["train_seed"] == 42
    assert gen.classify(1.0, 0.0, 0.0, 0.0) == ("near_floor", None)
    assert gen.classify(80.0, 10.0, 12.0, 10.0)[0] == "drops_to_random"
    assert gen.classify(80.0, 40.0, 12.0, 10.0)[0] == "partner_dependent"
    # insensitivity needs both replacements to keep the score
    assert gen.classify(80.0, 80.0, 0.0, 0.0)[0] == "partner_dependent"
    assert gen.classify(80.0, 78.0, 79.0, 10.0)[0] == "partner_insensitive"
    cat, ret = gen.classify(80.0, 45.0, 10.0, 10.0)
    assert ret == pytest.approx({"noop": 0.5, "random": 0.0})


def test_student_t_ci_and_paired_drop():
    recs = [{"true_objective": v, "wad_seed": w} for w, v in zip(range(5), (1.0, 2.0, 3.0, 4.0, 5.0))]
    assert gen.layout_ci95(recs) == pytest.approx(2.776 * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5))
    partner = [{"true_objective": r["true_objective"] - 1.0, "wad_seed": r["wad_seed"]} for r in recs]
    assert gen.paired_drop(recs, partner) == pytest.approx((1.0, 0.0))


def test_validation_rejects_incomplete_or_wrong_budget_runs():
    full = _run("ammo_carrier", "IPPO", 42, 1.0, 1, 1, 1, 1, wads=gen.EXPECTED_WADS)
    for name, recs in full["passes"].items():
        n = gen.BOTH_EPISODES_PER_WAD if name == "both_learned" else gen.SEAT_EPISODES_PER_WAD
        recs["records"] = [dict(r) for r in recs["records"] for _ in range(n)]
    full["checkpoint_steps"] = 100_003_840
    full.update(
        protocol_version=efp.PROTOCOL_VERSION, heldout_seeds=list(gen.EXPECTED_WADS), checkpoint_sha256="c",
        config_sha256="k", wad_sha256={str(w): "h" for w in gen.EXPECTED_WADS}, run_dir="/r/run",
    )
    assert gen.validate_run(full) == []
    assert gen.validate_run(full, expect_run_dir="/r/run", expect_steps=100_000_000) == []
    assert gen.validate_run(full, expect_run_dir="/r/other")
    old = dict(full, protocol_version=None)
    assert any("protocol_version" in p for p in gen.validate_run(old))
    no_wads = dict(full, heldout_seeds=[])
    assert gen.validate_run(no_wads)
    short = json.loads(json.dumps(full))
    short["passes"]["noop_seat0"]["records"].pop()
    assert any("noop_seat0" in p for p in gen.validate_run(short))
    early = json.loads(json.dumps(full))
    early["checkpoint_steps"] = 60_000_000
    assert any("checkpoint_steps" in p for p in gen.validate_run(early))
    missing = gen.missing_candidates([full])
    assert "ammo_carrier IPPO s68" in missing and "stealth_labyrinth QMIX s81" in missing
    assert "ammo_carrier IPPO s42" not in missing
    extra = dict(full, train_seed=97)
    assert any("unexpected" in p for p in gen.unexpected_candidates([full, extra]))
    assert any("duplicate" in p for p in gen.unexpected_candidates([full, full]))
    assert gen.unexpected_candidates([full]) == []
    assert gen.PROTOCOL_VERSION == efp.PROTOCOL_VERSION


def test_frozen_partner_winners_profile(monkeypatch):
    monkeypatch.setenv("COMRAD_EXPERIMENT_PROFILE", "frozen_partner_winners")
    import experiments.runs as runs

    runs = importlib.reload(runs)
    assert runs.BENCHMARK_NAME == "comrad_frozen_partner_winners"
    cmds = [c for c, *_ in runs.RUN_DESCRIPTION.generate_experiments("/scratch-shared/x", makedirs=False)]
    assert len(cmds) == runs.TOTAL_RUNS == 9
    combos = sorted(
        (next(t.split("=")[1] for t in c.split() if t.startswith("--env=")),
         next(t.split("=")[1] for t in c.split() if t.startswith("--algo=")),
         next(t.split("=")[1] for t in c.split() if t.startswith("--seed=")))
        for c in cmds
    )
    assert combos == sorted(
        [(e, "IPPO", s) for e in ("coop_health_gathering", "dumb_enemies", "ammo_carrier") for s in ("42", "68")]
        + [("stealth_labyrinth", "QMIX", s) for s in ("42", "68", "81")]
    )
    for c in cmds:
        assert "--train_for_env_steps=100000000" in c and "--with_wandb=False" in c
        assert "--obs_modality=pixel" in c and "--full_observability=False" in c
        assert "--train_dir=/scratch-shared/x/" in c and "frozen_partner_winners" in c
        if "--algo=IPPO" in c:
            assert "--shared_reward_alpha=0.0" in c
        else:
            assert f"--replay_buffer_size={CANONICAL_STEALTH_QMIX_REPLAY_BUFFER}" in c
