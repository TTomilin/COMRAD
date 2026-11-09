"""
Shared Lava Maze grid helpers (decoding ACS maze bits, BFS distance / next step).

Grid coordinates follow the ACS telemetry: USER14/USER15 = navigator grid x/y, USER13/USER18 = goal grid x/y,
USER19..21 = safe-tile bit chunks (27 bits each), USER11 = grid size. The navigator's angle is pinned to 0 by ACS,
so +x = MOVE_FORWARD, -x = MOVE_BACKWARD, +y = MOVE_LEFT, -y = MOVE_RIGHT.
"""
from collections import deque

UNREACHABLE = 999

# direction index -> (dx, dy); index 0 means "none" (already at goal / no path)
DIRECTIONS = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}
_NEIGHBOUR_ORDER = [(0, 1), (0, -1), (1, 0), (-1, 0)]
_NEXT_STEP_ORDER = [DIRECTIONS[k] for k in sorted(DIRECTIONS)]
_DIRECTION_INDEX = {delta: k for k, delta in DIRECTIONS.items()}


def safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def decode_maze_grid(bits_0, bits_1, bits_2, grid_size):
    """
    Reconstructs the grid from the 3 ACS integer chunks.
    Returns a 2D list indexed [y][x] where 1 is safe, 0 is lava.
    """
    s = grid_size
    chunks = [int(bits_0), int(bits_1), int(bits_2)]
    if s * s > 27 * len(chunks):
        raise ValueError(f"A {s}x{s} grid does not fit in {len(chunks)} chunks of 27 bits")
    grid = [[0 for _ in range(s)] for _ in range(s)]

    for y in range(s):
        for x in range(s):
            bit_idx = y * s + x
            chunk = bit_idx // 27
            shift = bit_idx % 27

            is_safe = (chunks[chunk] & (1 << shift)) != 0
            grid[y][x] = 1 if is_safe else 0

    return grid


def bfs_distance(grid, start_x, start_y, goal_x, goal_y, grid_size=None):
    """Shortest-path distance over safe tiles, UNREACHABLE (999) when standing in lava or no path exists."""
    start_x = safe_int(start_x, -1)
    start_y = safe_int(start_y, -1)
    goal_x = safe_int(goal_x, -1)
    goal_y = safe_int(goal_y, -1)

    if start_x == goal_x and start_y == goal_y:
        return 0

    s = grid_size or len(grid)
    if not (0 <= start_x < s and 0 <= start_y < s) or grid[start_y][start_x] == 0:
        return UNREACHABLE  # Standing in lava

    queue = deque([(start_x, start_y, 0)])
    visited = {(start_x, start_y)}

    while queue:
        cx, cy, dist = queue.popleft()

        if cx == goal_x and cy == goal_y:
            return dist

        for dx, dy in _NEIGHBOUR_ORDER:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < s and 0 <= ny < s and (nx, ny) not in visited and grid[ny][nx] == 1:
                visited.add((nx, ny))
                queue.append((nx, ny, dist + 1))

    return UNREACHABLE  # Fallback if path doesn't exist


def bfs_next_step(grid, start_x, start_y, goal_x, goal_y, grid_size=None):
    """
    Direction index of the first move on a shortest safe path (1:+x, 2:-x, 3:+y, 4:-y), 0 if already at the goal,
    standing in lava / off-grid, or the goal is unreachable. Ties are broken deterministically by index order.
    """
    start_x = safe_int(start_x, -1)
    start_y = safe_int(start_y, -1)
    goal_x = safe_int(goal_x, -1)
    goal_y = safe_int(goal_y, -1)

    if start_x == goal_x and start_y == goal_y:
        return 0

    s = grid_size or len(grid)
    if not (0 <= start_x < s and 0 <= start_y < s) or grid[start_y][start_x] == 0:
        return 0

    queue = deque()
    first_move = {(start_x, start_y): None}
    queue.append((start_x, start_y))

    while queue:
        cx, cy = queue.popleft()
        if cx == goal_x and cy == goal_y:
            return _DIRECTION_INDEX[first_move[(cx, cy)]]

        for dx, dy in _NEXT_STEP_ORDER:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < s and 0 <= ny < s and (nx, ny) not in first_move and grid[ny][nx] == 1:
                first_move[(nx, ny)] = first_move[(cx, cy)] or (dx, dy)
                queue.append((nx, ny))

    return 0


def grid_size_from(info, default=8):
    return max(1, safe_int((info or {}).get("USER11", default), default))


def next_step_from_info(info, default_grid_size=8):
    """BFS next-step direction index for the navigator, computed from Lava Maze USER telemetry."""
    info = info or {}
    grid_size = grid_size_from(info, default_grid_size)
    grid = decode_maze_grid(
        safe_int(info.get("USER19", 0), 0),
        safe_int(info.get("USER20", 0), 0),
        safe_int(info.get("USER21", 0), 0),
        grid_size,
    )
    return bfs_next_step(
        grid,
        info.get("USER14", -1),
        info.get("USER15", -1),
        info.get("USER13", -1),
        info.get("USER18", -1),
        grid_size,
    )
