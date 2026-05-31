import gymnasium as gym
import numpy as np

from comrad.envs.action_space import doom_action_space_lava_maze, doom_action_space_lava_maze_simple
from comrad.envs.doom_params import default_doom_cfg
from comrad.utils.doom_utils import doom_env_by_name, get_extra_wrappers
from comrad.wrappers.action_mask import RoleActionMaskWrapper


FULL_MASKS = {
    0: np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    1: np.array([1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
}
SIMPLE_MASKS = {
    0: np.array([1, 1, 1, 1, 0, 1, 0, 0], dtype=np.float32),
    1: np.array([1, 0, 0, 1, 1, 1, 1, 1], dtype=np.float32),
}


class DummyEnv(gym.Env):
    def __init__(self, player_id, action_space, dict_obs=False):
        self.player_id = player_id
        self.action_space = action_space
        image_space = gym.spaces.Box(0, 255, shape=(4, 4, 3), dtype=np.uint8)
        if dict_obs:
            self.observation_space = gym.spaces.Dict({
                "obs": image_space,
                "measurements": gym.spaces.Box(0, 1, shape=(2,), dtype=np.float32),
            })
        else:
            self.observation_space = image_space
        self.dict_obs = dict_obs

    def _obs(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        if self.dict_obs:
            return {"obs": image, "measurements": np.zeros(2, dtype=np.float32)}
        return image

    def reset(self, **kwargs):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, False, False, {}


def _wrapped_mask(player_id, action_space, masks, dict_obs=False):
    env = RoleActionMaskWrapper(
        DummyEnv(player_id, action_space, dict_obs=dict_obs),
        masks_by_player_id=masks,
    )
    obs, _ = env.reset()
    return obs["action_mask"]


def test_full_lava_maze_masks_match_roles():
    assert np.array_equal(_wrapped_mask(0, doom_action_space_lava_maze(), FULL_MASKS), FULL_MASKS[0])
    assert np.array_equal(_wrapped_mask(1, doom_action_space_lava_maze(), FULL_MASKS), FULL_MASKS[1])


def test_simple_lava_maze_masks_match_roles():
    assert np.array_equal(_wrapped_mask(0, doom_action_space_lava_maze_simple(), SIMPLE_MASKS), SIMPLE_MASKS[0])
    assert np.array_equal(_wrapped_mask(1, doom_action_space_lava_maze_simple(), SIMPLE_MASKS), SIMPLE_MASKS[1])


def test_role_action_mask_preserves_dict_observation():
    env = RoleActionMaskWrapper(
        DummyEnv(1, doom_action_space_lava_maze_simple(), dict_obs=True),
        masks_by_player_id=SIMPLE_MASKS,
    )

    obs, _ = env.reset()

    assert set(obs) == {"obs", "measurements", "action_mask"}
    assert np.array_equal(obs["action_mask"], SIMPLE_MASKS[1])
    assert env.observation_space["action_mask"].shape == (8,)


def test_lava_maze_specs_include_role_action_masks():
    for env_name in ("lava_maze", "lava_maze_simple"):
        cfg = default_doom_cfg(env=env_name)
        cfg.use_additional_input = True

        wrappers = get_extra_wrappers(cfg, doom_env_by_name(env_name))

        assert wrappers[-1][0] is RoleActionMaskWrapper
