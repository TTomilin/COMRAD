import pytest

from comrad.wrappers.scenario_wrappers.lava_maze_grid import decode_maze_grid


def test_decode_maze_grid_reads_bits_row_major_across_chunks():
    grid = decode_maze_grid(1 | (1 << 26), 1, 0, 9)
    assert grid[0][0] == 1 and grid[2][8] == 1 and grid[3][0] == 1
    assert sum(map(sum, grid)) == 3


def test_decode_maze_grid_rejects_grids_wider_than_three_chunks():
    assert decode_maze_grid(2**27 - 1, 2**27 - 1, 2**27 - 1, 9) == [[1] * 9 for _ in range(9)]
    with pytest.raises(ValueError):
        decode_maze_grid(0, 0, 0, 10)
