import os
from contextlib import contextmanager
from os.path import join
from typing import Optional

from filelock import FileLock, Timeout
from sample_factory.utils.network import is_udp_port_available
from sample_factory.utils.utils import log, project_tmp_dir
from comrad.envs.doom_gym import VizdoomEnv

DEFAULT_UDP_PORT = int(os.environ.get("DOOM_DEFAULT_UDP_PORT", 40300))
# log.info('Default UDP port is %r', DEFAULT_UDP_PORT)

# This try except block is to increase the env timeout connection flag in travis
try:
    vizdoom_env_timeout = int(os.environ["TRAVIS_VIZDOOM_ENV_TIMEOUT"])
except KeyError:
    vizdoom_env_timeout = 60


def find_available_port(start_port, increment=1000):
    port = start_port
    while port < 65535 and not is_udp_port_available(port):
        port += increment

    log.debug("Port %r is available", port)
    return port


def udp_port_reservation_file(port: int) -> str:
    return join(project_tmp_dir(), f"doom_udp_port_{port}.lockfile")


@contextmanager
def reserve_available_port(start_port, increment=1000, lock_timeout=0.1):
    """
    Reserve a multiplayer UDP port cooperatively across COMRAD jobs.

    We cannot keep a probe socket bound until VizDoom starts because the host process
    itself must bind that port. Instead we hold a per-port file lock from selection
    through worker initialization so another job using the same allocator cannot steal
    the same candidate between probe and bind.
    """

    port = start_port
    while port < 65535:
        port_lock = FileLock(udp_port_reservation_file(port))
        try:
            port_lock.acquire(timeout=lock_timeout)
        except Timeout:
            port += increment
            continue

        try:
            if is_udp_port_available(port):
                log.debug("Reserved UDP port %r", port)
                yield port
                return
        finally:
            port_lock.release()

        port += increment

    raise RuntimeError(f"Could not reserve an available UDP port starting from {start_port}")


class VizdoomEnvMultiplayer(VizdoomEnv):
    def __init__(
        self,
        action_space,
        config_file,
        player_id,
        num_agents,
        max_num_players,
        num_bots,
        skip_frames,
        async_mode=False,
        forcerespawn=1,
        respawn_delay=0,
        nofreelook=1,
        timelimit=0.0,
        record_to=None,
        render_mode: Optional[str] = None,
        host_ip: str = "127.0.0.1",
    ):
        super().__init__(
            action_space,
            config_file,
            skip_frames=skip_frames,
            async_mode=async_mode,
            record_to=record_to,
            render_mode=render_mode,
        )

        self.worker_index = 0
        self.vector_index = 0

        self.player_id = player_id
        self.num_agents = num_agents  # num agents that are not humans or bots
        self.max_num_players = max_num_players
        self.num_bots = num_bots
        self.timestep = 0
        self.update_state = True

        self.forcerespawn = forcerespawn
        self.nofreelook = nofreelook
        if num_agents == 1: self.forcerespawn = 1 # To overwrite for single agent pitfall

        self.respawn_delay = respawn_delay
        self.timelimit = timelimit
        self.host_ip = host_ip

        self.is_multiplayer = True
        self.init_info = None

    def _is_server(self):
        return self.player_id == 0

    def _ensure_initialized(self):
        if self.initialized:
            # Doom env already initialized!
            return

        self._create_doom_game(self.mode)
        port = DEFAULT_UDP_PORT if self.init_info is None else self.init_info.get("port", DEFAULT_UDP_PORT)

        if self._is_server():
            log.info("Using port %d on host...", port)
            if not is_udp_port_available(port):
                raise Exception("Port %r unavailable", port)

            # This process will function as a host for a multiplayer game with this many players (including the host).
            # It will wait for other machines to connect using the -join parameter and then
            # start the game when everyone is connected.
            game_args_list = [
                f"-host {self.max_num_players}",
                f"-port {port}",
                f"+timelimit {self.timelimit}",  # The game (episode) will end after this many minutes have elapsed.
                "+sv_noautoaim 1",  # Autoaim is disabled for all players.
                "+sv_nocrouch 1",  # Disables crouching.
                f"+sv_nofreelook {self.nofreelook}",  # Disables free look with a mouse (only keyboard).
                f"+sv_forcerespawn {self.forcerespawn}",  # Players will respawn automatically after they die.
                f"+viz_respawn_delay {self.respawn_delay}",  # Sets delay between respanws (in seconds).
                f"+viz_connect_timeout {vizdoom_env_timeout}",
            ]

            self.game.add_game_args(" ".join(game_args_list))

            # Additional commands:
            #
            # disables depth and labels buffer and the ability to use commands
            # that could interfere with multiplayer game (should use this in evaluation)
            # '+viz_nocheat 1'

            # Name your agent and select color
            # colors:
            # 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
            self.game.add_game_args(f"+name AI{self.player_id}_host +colorset 0")

            if self.record_to is not None:
                # reportedly this does not work with bots
                demo_path = self.demo_path(self._num_episodes, self.record_to)
                log.debug("Recording multiplayer demo to %s", demo_path)
                self.game.add_game_args(f"-record {demo_path}")
        else:
            # Join existing game.
            self.game.add_game_args(
                f"-join {self.host_ip}:{port} "  # Connect to a host for a multiplayer game.
                f"+viz_connect_timeout {vizdoom_env_timeout} "
            )

            # Name your agent and select color
            # colors:
            # 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
            self.game.add_game_args(f"+name AI{self.player_id} +colorset 0")

        self.game.set_episode_timeout(int(self.timelimit * 60 * self.game.get_ticrate()))

        self._game_init(with_locking=False)  # locking is handled by the multi-agent wrapper
        log.info("Initialized w:%d v:%d player:%d", self.worker_index, self.vector_index, self.player_id)
        self.initialized = True

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        if self._is_server() and self.num_bots > 0:
            self.game.send_game_command("removebots")

            for _ in range(self.num_bots):
                self.game.send_game_command("addbot")

        self.timestep = 0
        self.update_state = True
        return obs, info

    def step(self, actions):
        if self._actions_flattened is not None or self.skip_frames > 1 or self.num_agents == 1:
            return super().step(actions)

        self._ensure_initialized()

        actions_binary = self._convert_actions(actions)

        self.game.set_action(actions_binary)
        self.game.advance_action(1, self.update_state)
        self.timestep += 1

        if not self.update_state:
            return None, None, None, None, None

        state = self.game.get_state()
        reward = self.game.get_last_reward()
        terminated = self.game.is_episode_finished()

        if self.record_to is not None:
            # send 'stop recording' command 1 tick before the end of the episode
            # otherwise it does not get saved to disk
            if self.game.get_episode_time() + 1 == self.game.get_episode_timeout():
                log.debug("Calling stop recording command!")
                self.game.send_game_command("stop")

        observation, terminated, info = self._process_game_step(state, terminated, {})
        truncated = False
        return observation, reward, terminated, truncated, info
