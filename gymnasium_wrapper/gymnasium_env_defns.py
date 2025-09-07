import os

from gymnasium.utils import EzPickle

from vizdoom import scenarios_path
from vizdoom.gymnasium_wrapper.base_gymnasium_env import VizdoomEnv


class VizdoomScenarioEnv(VizdoomEnv, EzPickle):
    """Basic ViZDoom environments which reside in the `scenarios` directory"""

    def __init__(
        self,
        scenario_config_file,
        frame_skip=1,
        max_buttons_pressed=1,
        render_mode=None,
        treat_episode_timeout_as_truncation=True,
    ):
        EzPickle.__init__(
            self, scenario_config_file, frame_skip, max_buttons_pressed, render_mode
        )
        super().__init__(
            os.path.join(scenarios_path, scenario_config_file),
            frame_skip,
            max_buttons_pressed,
            render_mode,
            treat_episode_timeout_as_truncation=treat_episode_timeout_as_truncation,
        )
