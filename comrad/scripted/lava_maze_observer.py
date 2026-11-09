from __future__ import annotations

from typing import Any, Optional

from comrad.wrappers.scenario_wrappers.lava_maze_grid import next_step_from_info, safe_int

# BFS direction index == colour / weapon slot: 1 green (+x, forward), 2 red (-x, backward),
# 3 yellow (+y, left), 4 blue (-y, right). The navigator's angle is pinned to 0 by ACS.
NUM_COLOURS = 4


class LavaMazeObserverPolicy:
    """
    Scripted Lava Maze observer. Signals the navigator's BFS next step toward the goal as a colour flash, using only
    ACS telemetry that the observer role receives (USER11/13/14/15/17/18/19/20/21, optional SELECTED_WEAPON).

    Action layout matches doom_action_space_lava_maze(): (fwd/back, right/left, attack, weapon_select).
    State machine per outer step:
      * flash active (USER17 == 1) or no useful direction -> noop
      * wanted colour not equipped -> select it (SELECTED_WEAPON when available, otherwise hold for `hold_steps`)
      * wanted colour equipped -> attack (fires the flash via the weapon's Fire state)
    """

    def __init__(self, action_space=None, hold_steps: int = 3, grid_size: int = 8):
        if hold_steps < 1:
            raise ValueError("hold_steps must be >= 1")
        self.action_space = action_space
        self.hold_steps = hold_steps
        self.grid_size = grid_size
        self.reset()

    def reset(self) -> None:
        self.equipped: Optional[int] = None  # unknown until we select a colour ourselves
        self._selecting: Optional[int] = None
        self._hold = 0

    @staticmethod
    def _noop():
        return (0, 0, 0, 0)

    def wanted_colour(self, info: Optional[dict]) -> int:
        if not info:
            return 0
        return next_step_from_info(info, self.grid_size)

    def _is_equipped(self, colour: int, info: dict) -> bool:
        if "SELECTED_WEAPON" in info:
            return safe_int(info.get("SELECTED_WEAPON"), -1) == colour
        return self.equipped == colour

    def act(self, obs: Any, info: Optional[dict]):
        info = info or {}
        if safe_int(info.get("USER17", 0), 0) == 1:
            return self._noop()

        colour = self.wanted_colour(info)
        if not 1 <= colour <= NUM_COLOURS:
            self._selecting, self._hold = None, 0
            return self._noop()

        if self._is_equipped(colour, info):
            self.equipped = colour
            self._selecting, self._hold = None, 0
            return (0, 0, 1, 0)

        if self._selecting != colour:
            self._selecting, self._hold = colour, 0
        self._hold += 1
        if "SELECTED_WEAPON" not in info and self._hold >= self.hold_steps:
            self.equipped = colour
            self._selecting, self._hold = None, 0
        return (0, 0, 0, colour)
