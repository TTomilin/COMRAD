"""
Utility functions for geometry operations.
"""

import math
from typing import List, Tuple, Optional
from shapely.geometry import Point, LineString
from doomgen.geometry.seeds import Seed

def create_wavy_path(start: Tuple[float, float], end: Tuple[float, float], num_points: int = 15, amplitude: float = 40) -> List[Tuple[float, float]]:
    """
    Creates a wavy path between two points using a sine wave.

    Args:
        start: (x, y) start point
        end: (x, y) end point
        num_points: Number of segments
        amplitude: Amplitude of the wave

    Returns:
        List of (x, y) points
    """
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx*dx + dy*dy)

    if length == 0:
        return [start, end]

    # Perpendicular vector
    px = -dy / length
    py = dx / length

    points = []
    for i in range(num_points + 1):
        t = i / num_points
        lx = x1 + dx * t
        ly = y1 + dy * t

        # Sine wave offset (0 at ends)
        # S-shape using sin(t * 2pi)
        offset = amplitude * math.sin(t * math.pi * 2)

        fx = lx + px * offset
        fy = ly + py * offset
        points.append((fx, fy))

    return points

def add_void_ring(seeds_list: List[Seed], center_x: float, center_y: float, radius: float, exclude_paths: List[Tuple[List[Tuple[float, float]], float]] = [], margin: float = 16) -> None:
    """
    Adds a ring of VOID seeds around a circle, excluding areas near paths.

    Args:
        seeds_list: List to append seeds to
        center_x: Center X
        center_y: Center Y
        radius: Radius of the ring
        exclude_paths: List of (path_points, width) tuples to exclude void seeds near
        margin: Extra margin for exclusion
    """
    circumference = 2 * math.pi * radius
    count = int(circumference / 32) # 32 unit spacing

    for i in range(count):
        angle = 2 * math.pi * i / count
        vx = center_x + radius * math.cos(angle)
        vy = center_y + radius * math.sin(angle)

        # Check exclusion
        excluded = False
        p = Point(vx, vy)
        for path_points, width in exclude_paths:
            line = LineString(path_points)
            if line.distance(p) < (width / 2 + margin):
                excluded = True
                break

        if not excluded:
            seeds_list.append(Seed(vx, vy, "VOID", is_boundary=True))
