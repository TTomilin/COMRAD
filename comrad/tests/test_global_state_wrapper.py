import os

import gymnasium as gym
import numpy as np
import pytest

from comrad.envs.wad_geometry import WadGeometry, load_wad_geometry, scenario_wad_and_map
from comrad.wrappers.global_state import GlobalStateWrapper, rasterize_blocking_lines, task_variable_keys

GAME_VARS = ["POSITION_X", "POSITION_Y", "ANGLE", "HEALTH", "USER1", "USER2", "KILLCOUNT"]
# a 1024 x 1024 square room centred on the origin
ROOM = np.array(
    [
        [-512, -512, 512, -512, 1],
        [512, -512, 512, 512, 1],
        [512, 512, -512, 512, 1],
        [-512, 512, -512, -512, 1],
    ],
    dtype=np.float32,
)
SCENARIOS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "scenarios")


def _agent_info(k, t=0):
    return {
        "POSITION_X": -400.0 + 300.0 * k + t,
        "POSITION_Y": 100.0 * k,
        "POSITION_Z": 0.0,
        "ANGLE": 90.0 * k,
        "HEALTH": 100.0 - 10 * k,
        "AMMO1": 20.0 + k,
        "USER1": float(k + 1),
        "USER2": 3.0,
        "KILLCOUNT": float(t),
        "DEAD": 0.0,
    }


class FakeGlobalJointEnv(gym.Env):
    """Joint env whose agent 0 info carries the engine global-state keys, like VizdoomEnv(enable_global_state_info)."""

    def __init__(self, num_agents=2, objects=None, lines=ROOM, done_at=None, dict_obs=False, meas_dim=0):
        self.num_agents = num_agents
        self.is_multiagent = True
        img = gym.spaces.Box(0, 255, shape=(3, 72, 128), dtype=np.uint8)
        if dict_obs or meas_dim:
            spaces = {"obs": img}
            if meas_dim:
                spaces["measurements"] = gym.spaces.Box(-1.0, 1.0, shape=(meas_dim,), dtype=np.float32)
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            self.observation_space = img
        self.action_space = gym.spaces.Discrete(2)
        self.meas_dim = meas_dim
        self.objects = np.zeros((0, 6), np.float32) if objects is None else np.asarray(objects, np.float32)
        self.lines = lines
        self.done_at = done_at
        self.layout_id = 0
        self.t = 0
        self.episode = 0
        self.send_gs = True

    def _obs(self):
        out = []
        for i in range(self.num_agents):
            img = np.full((3, 72, 128), i, dtype=np.uint8)
            if isinstance(self.observation_space, gym.spaces.Dict):
                o = {"obs": img}
                if self.meas_dim:
                    o["measurements"] = np.full(self.meas_dim, 0.5, dtype=np.float32)
                out.append(o)
            else:
                out.append(img)
        return out

    def _infos(self, with_lines):
        infos = [_agent_info(k, self.t) for k in range(self.num_agents)]
        if self.send_gs:
            infos[0]["_gs_objects"] = self.objects.copy()
            infos[0]["_gs_layout_id"] = self.layout_id
            if with_lines and self.lines is not None:
                infos[0]["_gs_lines"] = np.asarray(self.lines, np.float32).copy()
        return infos

    def reset(self, **kwargs):
        self.t = 0
        self.episode += 1
        self.layout_id += 1
        return self._obs(), self._infos(with_lines=True)

    def set_layout(self, lines):
        self.lines = lines
        self.layout_id += 1
        self._lines_pending = True

    def step(self, actions):
        self.t += 1
        done = self.done_at is not None and self.t >= self.done_at
        if done:
            terminal = self._infos(with_lines=False)
            for info in terminal:
                info.pop("_gs_objects", None)
                info.pop("_gs_layout_id", None)
            obs, reset_infos = self.reset()
            for info, r in zip(terminal, reset_infos):
                info["reset_info"] = r
            return obs, [0.0] * self.num_agents, [True] * self.num_agents, [False] * self.num_agents, terminal
        with_lines = getattr(self, "_lines_pending", False)
        self._lines_pending = False
        infos = self._infos(with_lines=with_lines)
        return self._obs(), [0.0] * self.num_agents, [False] * self.num_agents, [False] * self.num_agents, infos


def _make(num_agents=2, **kwargs):
    inner = FakeGlobalJointEnv(num_agents=num_agents, **{k: v for k, v in kwargs.items() if k in (
        "objects", "lines", "done_at", "dict_obs", "meas_dim")})
    env = GlobalStateWrapper(inner, horizon=100, game_variables=GAME_VARS)
    return inner, env


def _split(env, vec):
    n, a, td, od, g = env.num_agents, env.AGENT_DIM, env.task_dim, env.object_dim, env.grid_size
    i = 0
    agents = vec[i : i + n * a].reshape(n, a); i += n * a
    task = vec[i : i + n * td].reshape(n, td); i += n * td
    objects = vec[i : i + env.max_objects * od].reshape(env.max_objects, od); i += env.max_objects * od
    layout = vec[i : i + g * g].reshape(g, g); i += g * g
    time = vec[i:]
    return agents, task, objects, layout, time


@pytest.mark.parametrize("num_agents", [2, 3])
def test_observation_space_keeps_own_frame_and_adds_state(num_agents):
    _, env = _make(num_agents)
    expected = num_agents * (9 + 24) + 48 * 22 + 16 * 16 + 1
    assert env.observation_space["obs"].shape == (3, 72, 128)
    assert env.observation_space["measurements"].shape == (expected,)
    obs, _ = env.reset()
    assert len(obs) == num_agents
    for i, o in enumerate(obs):
        assert o["obs"].shape == (3, 72, 128) and o["obs"][0, 0, 0] == i
        assert o["measurements"].shape == (expected,)
        assert env.observation_space.contains(o)


def test_task_keys_exclude_pose_variables():
    assert task_variable_keys(GAME_VARS) == ["KILLCOUNT", "USER1", "USER2"]


def test_both_agents_receive_the_same_information():
    objects = [[100, 100, 0, 1, 0, 11], [-300, 50, 0, 0, 0, 12], [400, -400, 0, 0, 2, 13]]
    _, env = _make(2, objects=objects)
    obs, _ = env.reset()
    a0, t0, o0, l0, time0 = _split(env, obs[0]["measurements"])
    a1, t1, o1, l1, time1 = _split(env, obs[1]["measurements"])
    assert a0[0, 8] == 1.0 and a0[1, 8] == 0.0 and a1[0, 8] == 1.0
    np.testing.assert_array_equal(a0[:, :8], a1[::-1, :8])
    np.testing.assert_array_equal(t0, t1[::-1])
    present0 = o0[o0[:, 0] > 0]
    present1 = o1[o1[:, 0] > 0]
    assert len(present0) == len(present1) == 3
    np.testing.assert_array_equal(np.array(sorted(map(tuple, present0))), np.array(sorted(map(tuple, present1))))
    np.testing.assert_array_equal(l0, l1)
    np.testing.assert_array_equal(time0, time1)


def test_agent_and_task_features():
    _, env = _make(2)
    obs, _ = env.reset()
    agents, task, _, _, _ = _split(env, obs[0]["measurements"])
    # room bounds are [-512, 512]^2, so x=-400 maps to -400/512
    np.testing.assert_allclose(agents[0, :2], [-400 / 512, 0.0], atol=1e-6)
    np.testing.assert_allclose(agents[1, 3:5], [0.0, 1.0], atol=1e-6)  # partner yaw 90 deg
    assert agents[0, 7] == 1.0  # alive
    # task keys sorted: KILLCOUNT, USER1, USER2
    np.testing.assert_allclose(task[0, :3], np.log1p([0.0, 1.0, 3.0]), atol=1e-6)
    np.testing.assert_allclose(task[1, :3], np.log1p([0.0, 2.0, 3.0]), atol=1e-6)
    assert np.all(task[:, 3:] == 0)


def test_hidden_objects_are_included_and_sorted_nearest_first():
    # the far object is behind walls for both agents; the global state must still contain it
    objects = [[500, 500, 0, 0, 0, 1], [-390, 0, 0, 0, 0, 2]]
    _, env = _make(2, objects=objects)
    obs, _ = env.reset()
    _, _, o0, _, _ = _split(env, obs[0]["measurements"])
    assert o0[0, 0] == 1.0 and o0[1, 0] == 1.0 and o0[2, 0] == 0.0
    np.testing.assert_allclose(o0[0, 1:3], [-390 / 512, 0.0], atol=1e-6)  # nearest to agent 0 first
    np.testing.assert_allclose(o0[1, 1:3], [500 / 512, 500 / 512], atol=1e-6)
    np.testing.assert_array_equal(o0[0, 6:], [-1.0, 1.0] + [-1.0] * 14)  # type code 2 as +-1 bits
    assert not np.array_equal(o0[0, 6:], o0[1, 6:])


def test_more_objects_than_slots_keeps_nearest_and_zero_objects_is_fine():
    xs = np.linspace(-400, 500, 60)
    objects = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs),
                        np.arange(60)], axis=1)
    _, env = _make(2, objects=objects)
    obs, _ = env.reset()
    _, _, o0, _, _ = _split(env, obs[0]["measurements"])
    assert o0[:, 0].sum() == 48
    assert o0[:, 1].max() < 500 / 512  # the farthest objects from agent 0 were dropped

    _, env = _make(2)
    obs, _ = env.reset()
    _, _, o0, _, _ = _split(env, obs[0]["measurements"])
    assert np.all(o0 == 0)


def test_layout_grid_marks_crossed_cells_and_door_opens():
    bounds = (0.0, 0.0, 160.0, 160.0)  # 16 x 16 cells of 10 units
    wall = np.array([[0, 25, 160, 25, 1]], dtype=np.float32)
    grid = rasterize_blocking_lines(wall, bounds, 16)
    assert np.all(grid[2] == 1.0) and grid.sum() == 16.0
    short = np.array([[30, 5, 50, 5, 1]], dtype=np.float32)
    grid = rasterize_blocking_lines(short, bounds, 16)
    assert set(zip(*np.nonzero(grid))) == {(0, 3), (0, 4)}
    door_open = wall.copy()
    door_open[0, 4] = 0.0
    assert rasterize_blocking_lines(door_open, bounds, 16).sum() == 0.0

    closed = np.concatenate([ROOM, [[-512, 0, 512, 0, 1]]]).astype(np.float32)
    opened = closed.copy()
    opened[-1, 4] = 0.0
    inner, env = _make(2, lines=closed)
    obs, _ = env.reset()
    l_closed = _split(env, obs[0]["measurements"])[3]
    inner.set_layout(opened)
    obs, *_ = env.step([0, 0])
    l_open = _split(env, obs[0]["measurements"])[3]
    # row 8 holds the door line; columns 0 and 15 are the room's side walls; v2 maps one cell of wall to 1 - e^-0.5
    np.testing.assert_allclose(l_closed[8, 1:15], 1.0 - np.exp(-0.5), rtol=1e-5)
    assert np.all(l_open[8, 1:15] == 0.0)
    np.testing.assert_array_equal(l_closed[0], l_open[0])  # outer walls unchanged
    # layout persists on steps that do not resend lines
    obs, *_ = env.step([0, 0])
    np.testing.assert_array_equal(_split(env, obs[0]["measurements"])[3], l_open)


def test_auto_reset_step_uses_reset_info_and_strips_keys():
    inner, env = _make(2, objects=[[100, 0, 0, 0, 0, 1]], done_at=3)
    obs, infos = env.reset()
    assert not any(k.startswith("_gs") for info in infos for k in info)
    for _ in range(2):
        obs, *_ = env.step([0, 0])
    assert _split(env, obs[0]["measurements"])[4][0] == pytest.approx(2 / 100)
    obs, _, term, _, infos = env.step([0, 0])
    assert all(term)
    agents, _, objects, layout, time = _split(env, obs[0]["measurements"])
    assert time[0] == 0.0  # new episode
    np.testing.assert_allclose(agents[0, 0], -400 / 512, atol=1e-6)  # reset pose (t=0), not terminal pose
    assert objects[0, 0] == 1.0 and layout.sum() > 0
    for info in infos:
        assert not any(k.startswith("_gs") for k in info)
        assert not any(k.startswith("_gs") for k in info["reset_info"])


def test_missing_engine_keys_give_zero_objects_and_layout():
    inner, env = _make(2, objects=[[100, 0, 0, 0, 0, 1]])
    inner.send_gs = False
    obs, _ = env.reset()
    agents, _, objects, layout, _ = _split(env, obs[0]["measurements"])
    assert np.all(objects == 0) and np.all(layout == 0)
    assert agents[0, 7] == 1.0  # agents block still filled from game variables
    obs, *_ = env.step([0, 0])
    assert np.all(_split(env, obs[0]["measurements"])[2] == 0)


def test_existing_measurements_are_kept_in_front():
    _, env = _make(2, meas_dim=4)
    base = 2 * (9 + 24) + 48 * 22 + 16 * 16 + 1
    assert env.observation_space["measurements"].shape == (4 + base,)
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs[0]["measurements"][:4], np.full(4, 0.5, np.float32))
    assert obs[0]["measurements"].shape == (4 + base,)


def test_wad_swap_recomputes_bounds():
    inner, env = _make(2)
    env.reset()
    assert env._bounds == (-512.0, -512.0, 512.0, 512.0)
    inner.lines = ROOM * np.array([2, 2, 2, 2, 1], dtype=np.float32)
    env.reset()
    assert env._bounds == (-1024.0, -1024.0, 1024.0, 1024.0)


def test_wad_geometry_blocked_lines_follow_closed_sectors():
    geo = WadGeometry(
        segments=np.array([[0, 0, 1, 0], [1, 0, 1, 1], [2, 0, 2, 1]], dtype=np.float32),
        impassable=np.array([True, False, False]),
        front=np.array([0, 0, 1], dtype=np.int32),
        back=np.array([-1, 1, 2], dtype=np.int32),
        floor=np.zeros(3, dtype=np.float32),
        ceiling=np.array([128, 0, 128], dtype=np.float32),
        max_lines_per_sector=3,
    )
    closed = geo.closed_sectors()
    assert closed.tolist() == [False, True, False]
    np.testing.assert_array_equal(geo.lines_array(closed)[:, 4], [1, 1, 1])
    opened = geo.closed_sectors(floor=np.zeros(3), ceiling=np.full(3, 128.0))
    np.testing.assert_array_equal(geo.lines_array(opened)[:, 4], [1, 0, 0])


@pytest.mark.parametrize(
    "cfg,live",
    [("armory_siege.cfg", True), ("coop_puzzle.cfg", False), ("rhythm_sync.cfg", False)],
)
def test_real_wads_parse_and_live_sectors_only_where_safe(cfg, live):
    wad, map_name = scenario_wad_and_map(os.path.join(SCENARIOS_DIR, cfg))
    geo = load_wad_geometry(wad, map_name)
    assert geo.segments.shape[0] > 100 and geo.num_sectors > 0
    assert geo.supports_live_sectors is live
    lines = geo.lines_array(geo.closed_sectors())
    assert 0 < lines[:, 4].sum() <= len(lines)


# ------------------------------------------------------------------ encoding v2 fixes


class _VarEnv(FakeGlobalJointEnv):
    """Fake env whose agent infos carry extra game variables."""

    def __init__(self, extra, **kwargs):
        super().__init__(**kwargs)
        self.extra = extra

    def _infos(self, with_lines):
        infos = super()._infos(with_lines)
        for info in infos:
            info.update(self.extra)
        return infos


@pytest.mark.parametrize("version,core_distinct", [(1, False), (2, True)])
def test_v2_task_clip_keeps_large_variables_distinct(version, core_distinct):
    feats = []
    for core in (1000.0, 600.0):
        inner = _VarEnv({"USER1": core}, num_agents=2)
        env = GlobalStateWrapper(inner, horizon=100, game_variables=GAME_VARS, version=version)
        obs, _ = env.reset()
        _, task, *_ = _split(env, obs[0]["measurements"])
        feats.append(task[0, env.task_keys.index("USER1")])
    assert bool(feats[0] != feats[1]) == core_distinct
    assert np.all(np.abs(obs[0]["measurements"]) <= env.observation_space["measurements"].high)


@pytest.mark.parametrize(
    "extra,version,expected",
    [
        ({"AMMO1": 0.0, "AMMO2": 40.0}, 1, 0.0),  # v1 read AMMO1 first: shells were invisible in Armory Siege
        ({"AMMO1": 0.0, "AMMO2": 40.0}, 2, 40.0),
        ({"AMMO1": 0.0, "AMMO2": 40.0, "SELECTED_WEAPON_AMMO": 12.0}, 2, 12.0),
    ],
)
def test_v2_ammo_reads_the_held_weapon(extra, version, expected):
    inner = _VarEnv(extra, num_agents=2)
    env = GlobalStateWrapper(inner, horizon=100, game_variables=GAME_VARS, version=version)
    obs, _ = env.reset()
    agents, *_ = _split(env, obs[0]["measurements"])
    assert agents[0, 6] == pytest.approx(expected / 200.0)


def test_v2_hidden_crash_reset_restarts_clock_and_layout():
    inner, env = _make(2)
    env.reset()
    for _ in range(5):
        env.step([0, 0])
    small = ROOM * np.array([0.5, 0.5, 0.5, 0.5, 1], np.float32)
    real_step = inner.step

    def crashed_step(actions):
        obs, rew, term, trunc, infos = real_step(actions)
        _, reset_infos = inner.reset()
        reset_infos[0]["_gs_lines"] = small.copy()
        for info, r in zip(infos, reset_infos):
            info["_hidden_reset_count"] = 1
            info["reset_info"] = r
        return obs, rew, term, trunc, infos

    inner.step = crashed_step
    obs, *_ = env.step([0, 0])
    assert env._step_count == 1
    assert env._bounds == pytest.approx((-256.0, -256.0, 256.0, 256.0))
    *_, time = _split(env, obs[0]["measurements"])
    assert time[0] == pytest.approx(1 / 100)


def test_v2_long_corridor_uses_all_grid_rows_and_per_axis_positions():
    # an 8000 x 400 corridor with a wall every 500 units (like Rhythm Sync): v1 squashes it into one saturated row
    corridor = [[-4000, -200, 4000, -200, 1], [-4000, 200, 4000, 200, 1]]
    corridor += [[x, -200, x, 100, 1] for x in range(-3500, 4000, 500)]
    corridor = np.array(corridor, dtype=np.float32)
    layouts, ys = {}, {}
    for version in (1, 2):
        inner = FakeGlobalJointEnv(num_agents=2, lines=corridor)
        env = GlobalStateWrapper(inner, horizon=100, game_variables=GAME_VARS, version=version)
        obs, _ = env.reset()
        agents, _, _, layout, _ = _split(env, obs[0]["measurements"])
        layouts[version], ys[version] = layout, agents[1, 1]  # agent 1 stands at y = 100
    assert (layouts[1].sum(1) > 0).sum() <= 2
    assert (layouts[2].sum(1) > 0).sum() >= 12
    assert layouts[2].max() < 1.0
    assert ys[1] == pytest.approx(100 / 4000) and ys[2] == pytest.approx(100 / 200)


def test_global_state_v2_training_normalizes_only_the_image():
    from types import SimpleNamespace

    from comrad.utils.doom_utils import restrict_global_state_input_normalization

    def cfg(**kw):
        base = dict(full_observability=True, full_obs_mode="global_state", global_state_version=2,
                    normalize_input=True, normalize_input_keys=None)
        base.update(kw)
        return SimpleNamespace(**base)

    c = cfg()
    restrict_global_state_input_normalization(c)
    assert c.normalize_input_keys == ["obs"]
    for kw in (dict(global_state_version=1), dict(full_obs_mode="shared_vision"), dict(full_observability=False),
               dict(normalize_input_keys=["obs", "measurements"])):
        c = cfg(**kw)
        before = c.normalize_input_keys
        restrict_global_state_input_normalization(c)
        assert c.normalize_input_keys == before
