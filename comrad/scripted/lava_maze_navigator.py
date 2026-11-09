from __future__ import annotations

from typing import Any, Optional, Tuple

from comrad.wrappers.scenario_wrappers.lava_maze_grid import DIRECTIONS, safe_int

# Fixed signal convention, identical to LavaMazeObserverPolicy: colour index == BFS direction index
# 1 green (+x, forward), 2 red (-x, backward), 3 yellow (+y, left), 4 blue (-y, right).
# Action layout matches doom_action_space_lava_maze(): (fwd/back, right/left, attack, weapon_select).
_MOVE_ACTIONS = {
    1: (1, 0, 0, 0),  # +x: MOVE_FORWARD (navigator angle pinned to 0 by ACS)
    2: (2, 0, 0, 0),  # -x: MOVE_BACKWARD
    3: (0, 2, 0, 0),  # +y: MOVE_LEFT
    4: (0, 1, 0, 0),  # -y: MOVE_RIGHT
}


def _direction_towards(cell: Tuple[int, int], target: Tuple[int, int]) -> int:
    dx, dy = target[0] - cell[0], target[1] - cell[1]
    if dx > 0:
        return 1
    if dx < 0:
        return 2
    if dy > 0:
        return 3
    if dy < 0:
        return 4
    return 0


class LavaMazeNavigatorPolicy:
    """
    Scripted Lava Maze navigator (converse of LavaMazeObserverPolicy). It never sees the maze layout or the goal: it
    reacts only to the observer's colour flash (USER16 colour, USER17 flash-active; the learned navigator perceives
    the same flash as a full-screen colour) and to its own grid cell (USER14/USER15), and executes the fixed
    one-colour-per-direction convention.

    Each flash onset (or colour change while a flash is shown) sets a target cell one step from the current cell in
    the signalled direction. The navigator moves until it is in the target cell, then for `centre_steps` more outer
    steps so it stops inside the cell rather than on its boundary, then holds. If momentum carries it out of the target
    cell afterwards, it steers back (it only uses its own position and the cell it was told to reach). Every signal
    gets a budget of `max_steps` moving steps, which bounds attempts into the map edge.
    """

    def __init__(self, action_space=None, centre_steps: int = 0, max_steps: int = 16):
        if centre_steps < 0:
            raise ValueError("centre_steps must be >= 0")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.action_space = action_space
        self.centre_steps = centre_steps
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> None:
        self._prev_flash = 0
        self._prev_colour = 0
        self._target: Optional[Tuple[int, int]] = None
        self._last_direction = 0
        self._extra = 0
        self._steps = 0

    @staticmethod
    def _noop():
        return (0, 0, 0, 0)

    def _new_signal(self, flash: int, colour: int) -> bool:
        if flash != 1 or not 1 <= colour <= len(DIRECTIONS):
            return False
        return self._prev_flash != 1 or colour != self._prev_colour

    def _move(self, direction: int):
        self._steps += 1
        self._last_direction = direction
        return _MOVE_ACTIONS[direction]

    def act(self, obs: Any, info: Optional[dict]):
        info = info or {}
        flash = safe_int(info.get("USER17", 0), 0)
        colour = safe_int(info.get("USER16", 0), 0)
        cell = (safe_int(info.get("USER14", -1), -1), safe_int(info.get("USER15", -1), -1))
        on_grid = cell[0] >= 0 and cell[1] >= 0

        if self._new_signal(flash, colour) and on_grid:
            dx, dy = DIRECTIONS[colour]
            self._target = (cell[0] + dx, cell[1] + dy)
            self._extra = 0
            self._steps = 0
        self._prev_flash = flash
        self._prev_colour = colour if flash == 1 else 0

        if self._target is None or not on_grid or self._steps >= self.max_steps:
            return self._noop()

        if cell != self._target:
            return self._move(_direction_towards(cell, self._target))
        if self._extra < self.centre_steps and self._last_direction:
            self._extra += 1
            return self._move(self._last_direction)
        return self._noop()
