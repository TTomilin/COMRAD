import gymnasium as gym
import numpy as np
import pytest

from comrad.wrappers.state_observation import BlindObservationWrapper, StateObservationWrapper
from comrad.wrappers.visibility_diagnostics import VisibilityDiagnosticsWrapper, is_in_frustum


class _FakeSinglePlayerEnv(gym.Env):
    def __init__(self, with_measurements=True):
        self.player_id = 0
        img_box = gym.spaces.Box(0, 255, shape=(3, 72, 128), dtype=np.uint8)
        if with_measurements:
            meas_box = gym.spaces.Box(0.0, 100.0, shape=(4,), dtype=np.float32)
            self.observation_space = gym.spaces.Dict(
                {
                    "obs": img_box,
                    "measurements": meas_box,
                    "action_mask": gym.spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32),
                }
            )
        else:
            self.observation_space = img_box
        self.action_space = gym.spaces.Discrete(5)
        self.with_measurements = with_measurements

    def _make_obs(self):
        img = np.full((3, 72, 128), 120, dtype=np.uint8)
        if self.with_measurements:
            return {
                "obs": img,
                "measurements": np.array([50.0, 25.0, 10.0, 1.0], dtype=np.float32),
                "action_mask": np.ones(5, dtype=np.float32),
            }
        return img

    def _make_info(self):
        return {
            "POSITION_X": 512.0,
            "POSITION_Y": -256.0,
            "POSITION_Z": 64.0,
            "ANGLE": 90.0,
            "HEALTH": 100.0,
            "AMMO1": 40.0,
            "USER1": 250.0,
            "KILLCOUNT": 5.0,
        }

    def reset(self, **kwargs):
        return self._make_obs(), self._make_info()

    def step(self, action):
        return self._make_obs(), 1.0, False, False, self._make_info()


@pytest.mark.parametrize("with_measurements", [True, False])
def test_state_observation_wrapper_replaces_obs_with_1d_vector(with_measurements):
    base = _FakeSinglePlayerEnv(with_measurements=with_measurements)
    env = StateObservationWrapper(base, scenario_name="armory_siege", horizon=100)
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert "measurements" not in env.observation_space.spaces
    assert len(env.observation_space["obs"].shape) == 1
    expected_dim = 8 + (4 if with_measurements else 0) + 10
    assert env.observation_space["obs"].shape == (expected_dim,)
    assert env.observation_space["obs"].dtype == np.float32

    obs, _ = env.reset()
    assert obs["obs"].shape == (expected_dim,)
    assert obs["obs"].dtype == np.float32
    assert np.all(np.isfinite(obs["obs"]))
    if with_measurements:
        assert "action_mask" in obs

    obs_step, _, _, _, _ = env.step(0)
    assert obs_step["obs"][7] == pytest.approx(0.01)


@pytest.mark.parametrize("encoding_kwargs", [{}, {"state_encoding": "ego"}])
def test_ego_state_encoding_is_backward_compatible(encoding_kwargs):
    env = StateObservationWrapper(_FakeSinglePlayerEnv(), scenario_name="armory_siege", horizon=100, **encoding_kwargs)
    obs, _ = env.reset()
    expected = np.array(
        # ego: x/2048, y/2048, z/512, cos, sin, health/200, ammo/200, t/H
        [0.25, -0.125, 0.125, 0.0, 1.0, 0.5, 0.2, 0.0]
        # measurements / max(|high|, 1)
        + [0.5, 0.25, 0.1, 0.01]
        # armory task: pid, USER1/500, AMMO1/200, AMMO2, W1, W2, SEL, KILL/100, HIT, ATTACK_READY
        + [0.0, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0],
        dtype=np.float32,
    )
    np.testing.assert_allclose(obs["obs"], expected, atol=1e-6)


def test_ego_rhythm_sync_target_offset_is_unchanged():
    base = _FakeSinglePlayerEnv(with_measurements=False)
    base._make_info = lambda: {"POSITION_X": 100.0, "POSITION_Y": 0.0, "USER33": 868.0, "USER34": -384.0, "USER32": 4.0}
    env = StateObservationWrapper(base, scenario_name="rhythm_sync_dense", horizon=100)
    obs, _ = env.reset()
    task = obs["obs"][8:]
    assert task[8] == pytest.approx(1.0) and task[9] == pytest.approx(-0.5)


def test_entity_state_encoding_declares_placeholder_space():
    from comrad.wrappers.state_observation import ENTITY_STATE_DIM

    env = StateObservationWrapper(_FakeSinglePlayerEnv(), scenario_name="armory_siege", state_encoding="entity")
    assert ENTITY_STATE_DIM == 277
    assert env.observation_space["obs"].shape == (277,)
    assert "measurements" not in env.observation_space.spaces
    obs, _ = env.reset()
    assert obs["obs"].shape == (277,) and not obs["obs"].any()
    assert "action_mask" in obs and "measurements" not in obs
    with pytest.raises(ValueError):
        StateObservationWrapper(_FakeSinglePlayerEnv(), state_encoding="bogus")


def test_blind_observation_wrapper_zeroes_rgb_and_injects_progress():
    base = _FakeSinglePlayerEnv(with_measurements=True)
    env = BlindObservationWrapper(base, horizon=10)
    assert env.observation_space["obs"].shape == (3, 72, 128)
    obs0, _ = env.reset()
    assert np.count_nonzero(obs0["obs"]) == 0
    # Local scalar measurements are preserved
    assert np.allclose(obs0["measurements"], [50.0, 25.0, 10.0, 1.0])

    obs5 = None
    for _ in range(5):
        obs5, *_ = env.step(0)
    assert obs5["obs"].flat[0] == round(0.5 * 255)
    assert np.count_nonzero(obs5["obs"].flat[1:]) == 0


def test_frustum_geometry():
    # Agent at origin facing East (yaw=0) sees target at (100, 0, 0)
    assert is_in_frustum((0.0, 0.0, 0.0, 0.0, 0.0), (100.0, 0.0, 0.0)) is True
    # Target at (100, 90, 0) -> ~42 deg yaw -> inside 45 deg half-FOV
    assert is_in_frustum((0.0, 0.0, 0.0, 0.0, 0.0), (100.0, 90.0, 0.0)) is True
    # Target behind at (-100, 0, 0) -> outside frustum
    assert is_in_frustum((0.0, 0.0, 0.0, 0.0, 0.0), (-100.0, 0.0, 0.0)) is False
    # Target far above (steep elevation > 36 deg) -> outside vertical frustum
    assert is_in_frustum((0.0, 0.0, 0.0, 0.0, 0.0), (50.0, 0.0, 100.0)) is False


def test_frustum_uses_zdoom_pitch_sign_and_eye_height():
    # Observer on a 128-unit platform looking down at a target 150 units away on the floor (Lava Maze observer)
    # ZDoom's PITCH is positive when looking down, so pitch=+30 points the camera at the target
    assert is_in_frustum((0.0, 0.0, 128.0, 0.0, 30.0), (150.0, 0.0, 0.0)) is True
    # The same pitch with the opposite sign looks up, away from the target
    assert is_in_frustum((0.0, 0.0, 128.0, 0.0, -30.0), (150.0, 0.0, 0.0)) is False
    # On flat ground the partner's body spans the eye line at any distance, so level views are unaffected
    assert is_in_frustum((0.0, 0.0, 0.0, 0.0, 0.0), (20.0, 0.0, 0.0)) is True
    assert is_in_frustum((0.0, 0.0, 0.0, 0.0, 0.0), (2000.0, 0.0, 0.0)) is True


def test_visibility_diagnostics_frustum_vs_labels_and_sprite_masking():
    class _FakeJointVisibilityEnv(gym.Env):
        def __init__(self, wall_occluded=False):
            self.num_agents = 2
            self.is_multiagent = True
            self.wall_occluded = wall_occluded
            self.observation_space = gym.spaces.Dict(
                {"obs": gym.spaces.Box(0, 255, shape=(3, 72, 128), dtype=np.uint8)}
            )
            self.action_space = gym.spaces.Discrete(2)

        def _step_data(self, done=False):
            img0 = np.full((3, 72, 128), 40, dtype=np.uint8)
            # Paint a bright "teammate sprite" patch at [20:40, 50:70]
            img0[:, 20:40, 50:70] = 240
            img1 = np.full((3, 72, 128), 60, dtype=np.uint8)

            labels_buf0 = np.zeros((120, 160), dtype=np.uint8)
            labels0 = []
            if not self.wall_occluded:
                # Native resolution labels_buffer has DoomPlayer (value=7) in the corresponding region
                labels_buf0[30:70, 60:90] = 7
                labels0.append(
                    {
                        "value": 7,
                        "object_name": "DoomPlayer",
                        "width": 30,
                        "height": 40,
                        "object_position_x": 200.0,
                        "object_position_y": 0.0,
                        "object_position_z": 0.0,
                    }
                )

            infos = [
                {
                    "POSITION_X": 0.0,
                    "POSITION_Y": 0.0,
                    "POSITION_Z": 0.0,
                    "ANGLE": 0.0,  # Facing Agent 1 at (200, 0)
                    "PITCH": 0.0,
                    "_labels": labels0,
                    "_labels_buffer": labels_buf0,
                },
                {
                    "POSITION_X": 200.0,
                    "POSITION_Y": 0.0,
                    "POSITION_Z": 0.0,
                    "ANGLE": 0.0,  # Facing away from Agent 0
                    "PITCH": 0.0,
                    "_labels": [],
                    "_labels_buffer": np.zeros((120, 160), dtype=np.uint8),
                },
            ]
            return [{"obs": img0}, {"obs": img1}], infos, [done, done]

        def reset(self, **kwargs):
            obs, infos, _ = self._step_data(done=False)
            return obs, infos

        def step(self, actions):
            obs, infos, dones = self._step_data(done=True)
            return obs, [0.0, 0.0], dones, [False, False], infos

    # Case 1: Open room (Agent 0 sees Agent 1 in frustum AND in labels_buffer; sprite is masked)
    env_open = VisibilityDiagnosticsWrapper(
        _FakeJointVisibilityEnv(wall_occluded=False),
        scenario_name="stag_hunt_arena",
        partner_occlusion_mode="sprite",
    )
    env_open.reset()
    obs, _, _, _, infos = env_open.step([0, 0])
    stats = infos[0]["episode_visibility_stats"]
    assert stats["teammate_in_frustum_rate"] == pytest.approx(0.5)
    assert stats["teammate_visible_rate"] == pytest.approx(0.5)
    assert stats["teammate_obscured_rate"] == pytest.approx(0.0)
    # Per-viewpoint rates: agent 0 faces its partner, agent 1 faces away
    assert stats["agent0_teammate_in_frustum_rate"] == pytest.approx(1.0)
    assert stats["agent0_teammate_visible_rate"] == pytest.approx(1.0)
    assert stats["agent1_teammate_in_frustum_rate"] == pytest.approx(0.0)
    assert stats["agent1_teammate_visible_rate"] == pytest.approx(0.0)
    # Sprite pixels in obs[0]["obs"] should be replaced by background median (40)
    assert np.all(obs[0]["obs"] == 40)
    assert "_labels_buffer" not in infos[0]

    # Case 2: Solid wall between agents (in frustum, but 0 pixels in labels_buffer -> obscured_rate == 1.0)
    env_wall = VisibilityDiagnosticsWrapper(
        _FakeJointVisibilityEnv(wall_occluded=True),
        scenario_name="rhythm_sync_dense",
        partner_occlusion_mode="none",
    )
    env_wall.reset()
    _, _, _, _, infos_wall = env_wall.step([0, 0])
    stats_wall = infos_wall[0]["episode_visibility_stats"]
    assert stats_wall["teammate_in_frustum_rate"] == pytest.approx(0.5)
    assert stats_wall["teammate_visible_rate"] == pytest.approx(0.0)
    assert stats_wall["teammate_obscured_rate"] == pytest.approx(1.0)


def test_1d_state_encoder_dispatch_in_vizdoom_and_qmix_models():
    import torch
    import torch.nn as nn
    from comrad.envs.doom_params import default_doom_cfg
    from comrad.models.doom_model import VizdoomEncoder
    from comrad.models.qmix_model import QMixAgentNet
    from sample_factory.model.encoder import MlpEncoder

    cfg = default_doom_cfg(algo="MAPPO", env="armory_siege")
    obs_space = gym.spaces.Dict({"obs": gym.spaces.Box(-5.0, 5.0, shape=(27,), dtype=np.float32)})
    action_space = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(4)))

    viz_enc = VizdoomEncoder(cfg, obs_space)
    assert isinstance(viz_enc.basic_encoder, MlpEncoder)
    assert not any(isinstance(m, nn.Conv2d) for m in viz_enc.modules())
    out = viz_enc({"obs": torch.randn(4, 27)})
    assert out.shape == (4, viz_enc.get_out_size())

    qmix_net = QMixAgentNet(cfg, obs_space, action_space)
    assert isinstance(qmix_net.encoder, MlpEncoder)
    assert not any(isinstance(m, nn.Conv2d) for m in qmix_net.modules())
    q_vals, _ = qmix_net({"obs": torch.randn(4, 27)})
    assert q_vals.shape == (4, 7)
