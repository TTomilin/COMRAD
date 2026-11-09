from comrad.scripted.lava_maze_navigator import LavaMazeNavigatorPolicy

NOOP = (0, 0, 0, 0)


def _info(cell, colour=0, flash=0):
    return {"USER14": cell[0], "USER15": cell[1], "USER16": colour, "USER17": flash}


def test_idle_without_signal():
    nav = LavaMazeNavigatorPolicy()
    assert nav.act(None, _info((2, 2))) == NOOP
    assert nav.act(None, {}) == NOOP


def test_each_colour_maps_to_fixed_direction():
    expected = {1: (1, 0, 0, 0), 2: (2, 0, 0, 0), 3: (0, 2, 0, 0), 4: (0, 1, 0, 0)}
    for colour, action in expected.items():
        nav = LavaMazeNavigatorPolicy()
        assert nav.act(None, _info((3, 3), colour, 1)) == action


def test_moves_until_target_cell_then_holds():
    nav = LavaMazeNavigatorPolicy(centre_steps=1)
    assert nav.act(None, _info((3, 3), 1, 1)) == (1, 0, 0, 0)
    assert nav.act(None, _info((3, 3), 1, 1)) == (1, 0, 0, 0)  # same flash, still moving
    assert nav.act(None, _info((4, 3), 1, 1)) == (1, 0, 0, 0)  # reached target, one centring step
    assert nav.act(None, _info((4, 3), 1, 1)) == NOOP
    assert nav.act(None, _info((4, 3))) == NOOP


def test_same_flash_does_not_retrigger_but_new_flash_does():
    nav = LavaMazeNavigatorPolicy()
    nav.act(None, _info((3, 3), 3, 1))
    assert nav.act(None, _info((3, 4), 3, 1)) == NOOP  # still the first flash, target reached
    assert nav.act(None, _info((3, 4), 0, 0)) == NOOP
    assert nav.act(None, _info((3, 4), 3, 1)) == (0, 2, 0, 0)  # new onset -> next cell


def test_steers_back_after_overshoot():
    nav = LavaMazeNavigatorPolicy()
    nav.act(None, _info((3, 3), 1, 1))
    assert nav.act(None, _info((4, 3), 1, 1)) == NOOP
    assert nav.act(None, _info((5, 3))) == (2, 0, 0, 0)  # drifted past the target -> back
    assert nav.act(None, _info((4, 3))) == NOOP


def test_step_budget_bounds_blocked_moves():
    nav = LavaMazeNavigatorPolicy(max_steps=3)
    moves = [nav.act(None, _info((0, 0), 2, 1 if t == 0 else 0)) for t in range(6)]
    assert moves[:3] == [(2, 0, 0, 0)] * 3
    assert moves[3:] == [NOOP] * 3


def test_reset_clears_target():
    nav = LavaMazeNavigatorPolicy()
    nav.act(None, _info((3, 3), 4, 1))
    nav.reset()
    assert nav.act(None, _info((3, 3))) == NOOP
