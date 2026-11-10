import math

import gymnasium as gym
import numpy as np
import pytest

from comrad.envs.doom_gym import actor_name_hash
from comrad.wrappers.entity_state import EntityStateWrapper
from comrad.wrappers.state_observation import ENTITY_STATE_DIM

R = 512.0
EGO, PARTNER, ENT, ENT_DIM, RAYS = 0, 5, 11, 20, 16
RAY0 = ENT + 12 * ENT_DIM
TASK0 = RAY0 + RAYS


def _info(x, y, yaw=0.0, health=100.0, ammo=50.0, **extra):
    d = {"POSITION_X": x, "POSITION_Y": y, "POSITION_Z": 0.0, "ANGLE": yaw, "HEALTH": health, "AMMO1": ammo}
    d.update(extra)
    return d


class FakeEntityJointEnv(gym.Env):
    """Joint env: agent 0's info carries `_gs_objects`, every agent's info a `_depth_row`, like the real engine."""

    def __init__(self, infos, objects=None, depth_rows=None, done_at=None, reset_infos=None):
        self.num_agents = len(infos)
        self.is_multiagent = True
        self.observation_space = gym.spaces.Dict(
            {"obs": gym.spaces.Box(-5.0, 5.0, shape=(ENTITY_STATE_DIM,), dtype=np.float32)}
        )
        self.action_space = gym.spaces.Discrete(2)
        self.base_infos = infos
        self.objects = np.zeros((0, 6), np.float32) if objects is None else np.asarray(objects, np.float32)
        self.depth_rows = depth_rows
        self.done_at = done_at
        self.reset_infos = reset_infos
        self.t = 0

    def _obs(self):
        return [{"obs": np.zeros(ENTITY_STATE_DIM, np.float32)} for _ in range(self.num_agents)]

    def _infos(self, base=None):
        infos = [dict(i) for i in (base or self.base_infos)]
        infos[0]["_gs_objects"] = self.objects.copy()
        if self.depth_rows is not None:
            for k in range(self.num_agents):
                infos[k]["_depth_row"] = np.asarray(self.depth_rows[k], np.uint8)
        return infos

    def reset(self, **kwargs):
        self.t = 0
        return self._obs(), self._infos()

    def step(self, actions):
        self.t += 1
        infos = self._infos()
        done = self.done_at is not None and self.t == self.done_at
        if done:
            new = self._infos(self.reset_infos)
            for i, n in zip(infos, new):
                i["reset_info"] = n
        flags = [done] * self.num_agents
        return self._obs(), [0.0] * self.num_agents, flags, [False] * self.num_agents, infos


def _wrap(env, scenario="armory_siege", **kw):
    return EntityStateWrapper(env, scenario_name=scenario, horizon=100, sight_range=R, **kw)


def test_vector_shape_bounds_and_both_agents_on_reset_and_step():
    env = _wrap(FakeEntityJointEnv([_info(0, 0), _info(100, 0)], objects=[[50, 0, 0, 0, 0, 7]],
                                   depth_rows=[np.full(160, 40), np.full(160, 10)]))
    assert env.observation_space["obs"].shape == (277,)
    obs, infos = env.reset()
    obs2, *_ = env.step([0, 0])
    for o in obs + obs2:
        v = o["obs"]
        assert v.shape == (277,) and v.dtype == np.float32
        assert np.all(v >= -5.0) and np.all(v <= 5.0)
    assert obs2[0]["obs"][4] == pytest.approx(0.01)


def test_partner_inside_radius_rotated_and_outside_is_zero():
    # agent 0 at origin facing +y (90 deg); partner 200 units along +x, i.e. to its right
    env = _wrap(FakeEntityJointEnv([_info(0, 0, yaw=90.0), _info(200, 0, yaw=0.0)]))
    obs, _ = env.reset()
    p = obs[0]["obs"][PARTNER:PARTNER + 6]
    assert p[0] == 1.0
    assert p[1] == pytest.approx(0.0, abs=1e-6)  # forward
    assert p[2] == pytest.approx(-200 / R)  # left (negative = right)
    assert p[3] == pytest.approx(200 / R)
    assert p[4] == pytest.approx(math.cos(math.radians(-90)), abs=1e-6)
    assert p[5] == pytest.approx(math.sin(math.radians(-90)))

    far = _wrap(FakeEntityJointEnv([_info(0, 0), _info(600, 0)]))
    obs, _ = far.reset()
    assert np.all(obs[0]["obs"][PARTNER:PARTNER + 6] == 0.0)
    assert np.all(obs[1]["obs"][PARTNER:PARTNER + 6] == 0.0)


def test_rotating_agent_by_90_degrees_rotates_offsets():
    objs = [[100, 50, 0, 0, 0, 3]]
    a = _wrap(FakeEntityJointEnv([_info(0, 0, yaw=0.0), _info(2000, 0)], objects=objs)).reset()[0][0]["obs"]
    b = _wrap(FakeEntityJointEnv([_info(0, 0, yaw=90.0), _info(2000, 0)], objects=objs)).reset()[0][0]["obs"]
    fa, la = a[ENT + 1], a[ENT + 2]
    fb, lb = b[ENT + 1], b[ENT + 2]
    assert (fa, la) == pytest.approx((100 / R, 50 / R))
    # turning left by 90 deg: what was ahead is now to the right, what was left is now ahead
    assert (fb, lb) == pytest.approx((la, -fa), abs=1e-6)
    assert a[ENT + 3] == pytest.approx(b[ENT + 3])


def test_entities_sorted_by_distance_with_type_code_and_radius_cut():
    h1, h2 = actor_name_hash("Clip"), actor_name_hash("BenchImp")
    objs = [[300, 0, 0, 0, 0, h1], [100, 0, 0, 0, 0, h2], [900, 0, 0, 0, 0, h1]]
    obs, _ = _wrap(FakeEntityJointEnv([_info(0, 0), _info(0, 2000)], objects=objs)).reset()
    v = obs[0]["obs"]
    slots = v[ENT:ENT + 12 * ENT_DIM].reshape(12, ENT_DIM)
    assert slots[:, 0].tolist()[:3] == [1.0, 1.0, 0.0]  # third object is outside the radius
    assert slots[0, 3] == pytest.approx(100 / R) and slots[1, 3] == pytest.approx(300 / R)
    expected = 2.0 * ((h2 >> np.arange(16)) & 1) - 1.0
    assert np.array_equal(slots[0, 4:], expected.astype(np.float32))
    assert np.all(slots[2:] == 0.0)


def test_more_objects_than_slots_keeps_nearest_and_zero_objects_is_zero():
    objs = [[10.0 * (k + 1), 0, 0, 0, 0, 1] for k in range(20)][::-1]
    obs, _ = _wrap(FakeEntityJointEnv([_info(0, 0), _info(0, 2000)], objects=objs)).reset()
    slots = obs[0]["obs"][ENT:ENT + 12 * ENT_DIM].reshape(12, ENT_DIM)
    assert np.all(slots[:, 0] == 1.0)
    assert slots[-1, 3] == pytest.approx(120 / R)
    obs, _ = _wrap(FakeEntityJointEnv([_info(0, 0), _info(0, 2000)])).reset()
    assert np.all(obs[0]["obs"][ENT:ENT + 12 * ENT_DIM] == 0.0)


def test_agent_1_receives_host_objects_relative_to_itself():
    objs = [[500, 0, 0, 0, 0, 5]]
    obs, infos = _wrap(FakeEntityJointEnv([_info(0, 0), _info(400, 0)], objects=objs)).reset()
    assert obs[1]["obs"][ENT] == 1.0
    assert obs[1]["obs"][ENT + 3] == pytest.approx(100 / R)
    assert "_gs_objects" not in infos[0]


def test_rays_are_min_pooled_depth_row():
    row = np.full(160, 60, np.uint8)
    row[:10] = 5
    obs, infos = _wrap(FakeEntityJointEnv([_info(0, 0), _info(0, 50)], depth_rows=[row, np.full(160, 255)])).reset()
    rays = obs[0]["obs"][RAY0:RAY0 + RAYS]
    assert rays[0] == pytest.approx(5 / 255) and rays[1] == pytest.approx(60 / 255)
    assert np.all(obs[1]["obs"][RAY0:RAY0 + RAYS] == pytest.approx(1.0))
    assert all("_depth_row" not in i for i in infos)


def test_missing_engine_keys_give_zero_blocks():
    class Bare(FakeEntityJointEnv):
        def _infos(self, base=None):
            return [dict(i) for i in self.base_infos]

    obs, _ = _wrap(Bare([_info(0, 0), _info(0, 3000)])).reset()
    v = obs[0]["obs"]
    assert np.all(v[ENT:TASK0] == 0.0)


def test_rhythm_sync_target_gated_and_rotated():
    near = _info(0, 0, yaw=90.0, USER33=0.0, USER34=200.0, USER35=5000.0, USER36=0.0)
    other = _info(0, 3000, USER33=0.0, USER34=200.0, USER35=5000.0, USER36=0.0)
    obs, _ = _wrap(FakeEntityJointEnv([near, dict(other, POSITION_X=0.0)]), scenario="rhythm_sync_dense").reset()
    t0 = obs[0]["obs"][TASK0:]
    assert t0[8] == pytest.approx(200 / R) and t0[9] == pytest.approx(0.0, abs=1e-6)
    t1 = obs[1]["obs"][TASK0:]  # agent 1's target (USER35/36) is ~5000 units away
    assert t1[0] == 1.0 and t1[8] == 0.0 and t1[9] == 0.0


def test_ammo_carrier_drops_enemy_count_and_gates_shooter_ammo():
    extra = dict(USER41=100.0, USER44=8.0, USER42=3.0)
    far = _wrap(FakeEntityJointEnv([_info(0, 0, **extra), _info(1000, 0, **extra)]), scenario="ammo_carrier")
    obs, _ = far.reset()
    shooter, runner = obs[0]["obs"][TASK0:], obs[1]["obs"][TASK0:]
    assert shooter[1] == pytest.approx(0.5) and shooter[4] == 0.0
    assert runner[1] == 0.0 and runner[4] == 0.0 and runner[2] == pytest.approx(3 / 20)
    near = _wrap(FakeEntityJointEnv([_info(0, 0, **extra), _info(100, 0, **extra)]), scenario="ammo_carrier")
    obs, _ = near.reset()
    assert obs[1]["obs"][TASK0 + 1] == pytest.approx(0.5)


def test_absolute_position_is_not_in_the_vector():
    a = _wrap(FakeEntityJointEnv([_info(0, 0), _info(100, 0)])).reset()[0][0]["obs"]
    b = _wrap(FakeEntityJointEnv([_info(700, -300), _info(800, -300)])).reset()[0][0]["obs"]
    assert np.array_equal(a, b)


def test_auto_reset_step_uses_reset_info_and_strips_keys():
    env = _wrap(
        FakeEntityJointEnv([_info(0, 0), _info(100, 0)], done_at=2, reset_infos=[_info(0, 0), _info(1000, 0)]),
    )
    env.reset()
    obs, *_ = env.step([0, 0])
    assert obs[0]["obs"][PARTNER] == 1.0
    obs, _, _, _, infos = env.step([0, 0])
    assert obs[0]["obs"][PARTNER] == 0.0  # new episode: partner 1000 units away
    assert obs[0]["obs"][4] == 0.0
    assert "_gs_objects" not in infos[0] and "_gs_objects" not in infos[0]["reset_info"]


def test_requires_multi_agent_and_entity_placeholder_space():
    with pytest.raises(ValueError):
        _wrap(FakeEntityJointEnv([_info(0, 0)]))
    env = FakeEntityJointEnv([_info(0, 0), _info(1, 0)])
    env.observation_space = gym.spaces.Dict({"obs": gym.spaces.Box(-5, 5, shape=(18,), dtype=np.float32)})
    with pytest.raises(ValueError):
        _wrap(env)
