import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from os.path import join
from typing import Dict, List

import numpy as np
from sample_factory.algo.utils.spaces.discretized import Discretized
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.utils import log

from comrad.utils.doom_utils import doom_env_by_name, make_doom_env_impl

# WSL2: ViZDoom needs an X-server. WSLg works with SDL_SOFTWARE_DRIVER:
# export DISPLAY=:0
# export SDL_RENDER_DRIVER=software


def parse_args():
    parser = argparse.ArgumentParser(description="COMRAD Human Baseline Recorder")
    parser.add_argument("--env", required=True, type=str, help="Scenario name (e.g. stag_hunt_arena)")
    parser.add_argument(
        "--role",
        required=True,
        choices=["host", "client"],
        type=str,
        help="Host creates the game, client joins over network",
    )
    parser.add_argument("--host_ip", default="127.0.0.1", type=str, help="IP of the host machine")
    parser.add_argument("--participant_name", default="anonymous", type=str)
    parser.add_argument("--num_humans", default=2, type=int, help="Total human players (including host)")
    parser.add_argument("--num_episodes", default=10, type=int, help="Recorded episodes per participant")
    parser.add_argument("--num_calibration_episodes", default=2, type=int, help="Practice episodes (not recorded)")
    parser.add_argument("--output_dir", default="results/human_baseline", type=str)
    parser.add_argument("--fps", default=35, type=int, help="Target frames per second")
    parser.add_argument("--env_frameskip", default=4, type=int, help="Frames to skip per step")
    parser.add_argument("--res_w", default=128, type=int)
    parser.add_argument("--res_h", default=72, type=int)
    parser.add_argument("--udp_port", default=40300, type=int, help="UDP port for multiplayer")
    return parser.parse_args()


class HumanBaselineRecorder:
    """Tracks episodes and saves results to JSON."""

    def __init__(
        self,
        scenario: str,
        participant_name: str,
        role: str,
        host_ip: str,
        num_calibration_episodes: int,
        output_dir: str,
    ):
        self.scenario = scenario
        self.participant_name = participant_name
        self.role = role
        self.host_ip = host_ip
        self.num_calibration_episodes = num_calibration_episodes
        self.output_dir = output_dir
        self.episodes: List[Dict] = []
        self.current_episode_idx = 0
        self.current_true_objective = 0.0
        self.current_reward = 0.0
        self.current_game_variables: Dict = {}
        self.is_calibration = True
        os.makedirs(output_dir, exist_ok=True)

    def start_episode(self, episode_idx: int):
        self.current_episode_idx = episode_idx
        self.is_calibration = episode_idx < self.num_calibration_episodes
        self.current_true_objective = 0.0
        self.current_reward = 0.0
        self.current_game_variables = {}
        self._episode_start_time = time.time()

    def update_step(self, reward: float, info: Dict):
        self.current_reward += reward
        if "true_objective" in info:
            self.current_true_objective = info["true_objective"]
        for key, val in info.items():
            if isinstance(val, (int, float)):
                self.current_game_variables[key] = float(val)

    def end_episode(self, terminated: bool, truncated: bool, info: Dict):
        duration = time.time() - self._episode_start_time
        episode_record = {
            "episode": self.current_episode_idx,
            "calibration": self.is_calibration,
            "true_objective": self.current_true_objective,
            "reward": self.current_reward,
            "episode_length": int(info.get("episode_time_tics", 0)),
            "duration_seconds": round(duration, 2),
            "game_variables": self.current_game_variables,
            "terminated": terminated,
            "truncated": truncated,
        }
        if not self.is_calibration:
            self.episodes.append(episode_record)

    def save(self):
        result = self._build_result()
        filepath = join(self.output_dir, self._output_filename())
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)

    def _build_result(self) -> Dict:
        objectives = [e["true_objective"] for e in self.episodes]
        rewards = [e["reward"] for e in self.episodes]
        lengths = [e["episode_length"] for e in self.episodes]
        durations = [e["duration_seconds"] for e in self.episodes]
        summary = {}
        if objectives:
            summary = {
                "mean_true_objective": round(float(np.mean(objectives)), 4),
                "std_true_objective": round(float(np.std(objectives)), 4) if len(objectives) > 1 else 0.0,
                "min_true_objective": round(float(np.min(objectives)), 4),
                "max_true_objective": round(float(np.max(objectives)), 4),
                "mean_reward": round(float(np.mean(rewards)), 4),
                "mean_episode_length": round(float(np.mean(lengths)), 1),
                "mean_duration_seconds": round(float(np.mean(durations)), 2),
            }
        return {
            "scenario": self.scenario,
            "participant_name": self.participant_name,
            "role": self.role,
            "host_ip": self.host_ip,
            "num_calibration_episodes": self.num_calibration_episodes,
            "total_episodes": len(self.episodes) + self.num_calibration_episodes,
            "recorded_episodes": len(self.episodes),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episodes": self.episodes,
            "summary": summary,
        }

    def _output_filename(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self.participant_name.replace(" ", "_").replace("/", "_")
        return f"{self.scenario}_{safe_name}_{ts}.json"


class HumanKeyboardHandler:
    """
    WASD: Move/Strafe
    Arrow LR: Turn
    Space: Jump
    C: Attack (shoot)
    E: Use
    1-7: Select weapon
    Q/R: Next/Prev weapon
    Page Up/Down: Look up/down
    Ctrl+R: Crouch
    Shift: Speed
    ESC: exit
    """

    def __init__(self, doom_config_path: str, num_buttons: int):
        try:
            from pynput.keyboard import Key
        except ImportError:
            log.error("pynput requires an X display. Set DISPLAY to a running X server.")
            raise

        self._esc_key = Key.esc
        self._button_keys = {
            "MOVE_FORWARD": "w",
            "MOVE_BACKWARD": "s",
            "TURN_LEFT": Key.left,
            "TURN_RIGHT": Key.right,
            "MOVE_LEFT": "a",
            "MOVE_RIGHT": "d",
            "ATTACK": "c",
            "USE": "e",
            "JUMP": Key.space,
            "SELECT_WEAPON1": "1",
            "SELECT_WEAPON2": "2",
            "SELECT_WEAPON3": "3",
            "SELECT_WEAPON4": "4",
            "SELECT_WEAPON5": "5",
            "SELECT_WEAPON6": "6",
            "SELECT_WEAPON7": "7",
            "SELECT_NEXT_WEAPON": "q",
            "SELECT_PREV_WEAPON": "r",
            "LOOK_UP": Key.page_up,
            "LOOK_DOWN": Key.page_down,
            "CROUCH": Key.ctrl_r,
            "SPEED": Key.shift,
        }
        self._current_actions = []
        self._terminate = False
        self._listener = None
        self._num_buttons = num_buttons

        button_names = _parse_doom_buttons(doom_config_path) if doom_config_path else []
        self._button_index = {name: i for i, name in enumerate(button_names)}

        self._key_map = {}
        for button_name, key in self._button_keys.items():
            if button_name in self._button_index:
                self._key_map[key] = self._button_index[button_name]

        log.info("Keyboard mapping: %s", {k: button_names[v] for k, v in self._key_map.items() if isinstance(v, int)})

        mapped_indices = {v for v in self._key_map.values() if isinstance(v, int)}
        unmapped = [b for i, b in enumerate(button_names) if i not in mapped_indices]
        if unmapped:
            log.warning("Unmapped doom buttons (no keyboard binding): %s", unmapped)

    def start(self):
        try:
            from pynput.keyboard import Listener
        except ImportError:
            log.error(
                "pynput requires an X display. Set DISPLAY to a running X server.\n"
                "  WSLg:   export DISPLAY=:0\n"
                "  VcXsrv: export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0\n"
                "  Also:   export SDL_RENDER_DRIVER=software"
            )
            raise
        self._listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self):
        self._terminate = True
        if self._listener is not None:
            self._listener.stop()

    def get_actions_flat(self, num_actions: int) -> list:
        actions = [0] * num_actions
        for action in self._current_actions:
            if isinstance(action, int) and action < num_actions:
                actions[action] = 1
        return actions

    @property
    def should_terminate(self) -> bool:
        return self._terminate

    def _on_press(self, key):
        if key == self._esc_key:
            self._terminate = True
            return False
        action = self._resolve_key(key)
        if action is not None and action not in self._current_actions:
            self._current_actions.append(action)

    def _on_release(self, key):
        action = self._resolve_key(key)
        if action is not None and action in self._current_actions:
            self._current_actions.remove(action)

    def _resolve_key(self, key):
        if key in self._key_map:
            return self._key_map[key]
        if hasattr(key, "char") and key.char is not None:
            return self._key_map.get(key.char, None)
        return None


def _make_cfg(args):
    return AttrDict(
        {
            "env_frameskip": args.env_frameskip,
            "fps": args.fps,
            "timelimit": None,
            "record_to": None,
            "wide_aspect_ratio": False,
            "res_w": args.res_w,
            "res_h": args.res_h,
            "pixel_format": "HWC",
            "host_ip": args.host_ip,
            "shared_reward_alpha": 0.0,
            "shared_reward_scalarisation": "sum",
            "algo": "APPO",
            "wandb_record_every": 0,
            "with_wandb": False,
        }
    )


def _parse_doom_buttons(config_path: str) -> List[str]:
    with open(config_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if not re.match(r"available_buttons\s*=\s*", stripped):
            continue
        buttons = []
        collecting = "{" in stripped
        if collecting:
            text = stripped[stripped.index("{") + 1 :]
            for token in text.split():
                if token == "}":
                    return buttons
                buttons.append(token)
        for j in range(i + 1, len(lines)):
            for token in lines[j].split():
                if token == "{":
                    collecting = True
                    continue
                if token == "}":
                    return buttons
                if collecting:
                    buttons.append(token)
        return buttons
    return []


def _count_doom_buttons(doom_env) -> int:
    config_path = getattr(doom_env, "config_path", None)
    if config_path is not None and os.path.isfile(config_path):
        buttons = _parse_doom_buttons(config_path)
        if buttons:
            return len(buttons)
    action_space = getattr(doom_env, "action_space", None)
    if action_space is not None and hasattr(action_space, "spaces"):
        count = 0
        for s in action_space.spaces:
            if isinstance(s, Discretized):
                count += 1
            elif hasattr(s, "n"):
                count += s.n - 1
            elif hasattr(s, "shape"):
                count += s.shape[0]
        return max(count, 1)
    return 14


def run_episode(env, keyboard_handler, recorder, episode_idx, skip_frames: int, fps: int):
    recorder.start_episode(episode_idx)
    doom = env.unwrapped
    doom.mode = "human"
    _, info = env.reset()
    last_render_time = time.time()
    time_between_frames = 1.0 / fps
    num_actions = _count_doom_buttons(doom)
    terminated = truncated = False
    while not terminated and not truncated and not keyboard_handler.should_terminate:
        actions = keyboard_handler.get_actions_flat(num_actions)
        for _ in range(skip_frames):
            doom._actions_flattened = actions
            obs, reward, terminated, truncated, info = env.step(actions)
            recorder.update_step(reward, info)
            time_since = time.time() - last_render_time
            time_wait = time_between_frames - time_since
            if time_wait > 0:
                time.sleep(time_wait)
            last_render_time = time.time()
            if terminated or truncated:
                break
    recorder.end_episode(terminated, truncated, info)


def build_env_for_player(args, player_id: int, num_agents: int, max_num_players: int):
    spec = doom_env_by_name(args.env)
    cfg = _make_cfg(args)
    return make_doom_env_impl(
        spec,
        cfg=cfg,
        env_config=None,
        skip_frames=1,
        player_id=player_id,
        num_agents=num_agents,
        max_num_players=max_num_players,
        num_bots=0,
        render_mode="human",
    )


def main():
    args = parse_args()
    num_agents = 0
    max_num_players = num_agents + args.num_humans
    player_id = 0 if args.role == "host" else 1
    log.info("Human baseline: %s [%s] as %s", args.env, args.role, args.participant_name)
    os.environ["DOOM_DEFAULT_UDP_PORT"] = str(args.udp_port)
    env = build_env_for_player(args, player_id, num_agents, max_num_players)
    doom = env.unwrapped
    num_buttons = _count_doom_buttons(doom)
    keyboard_handler = HumanKeyboardHandler(doom.config_path, num_buttons)
    recorder = HumanBaselineRecorder(
        scenario=args.env,
        participant_name=args.participant_name,
        role=args.role,
        host_ip=args.host_ip,
        num_calibration_episodes=args.num_calibration_episodes,
        output_dir=args.output_dir,
    )
    skip_frames = args.env_frameskip
    total_episodes = args.num_calibration_episodes + args.num_episodes
    keyboard_handler.start()
    try:
        for episode_idx in range(total_episodes):
            if keyboard_handler.should_terminate:
                break
            run_episode(env, keyboard_handler, recorder, episode_idx, skip_frames, args.fps)
    except KeyboardInterrupt:
        pass
    finally:
        keyboard_handler.stop()
        env.close()
    recorder.save()
    log.info("Results saved to: %s", join(args.output_dir, recorder._output_filename()))


if __name__ == "__main__":
    main()
