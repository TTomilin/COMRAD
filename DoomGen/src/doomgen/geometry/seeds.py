"""
Seed generation engine for Voronoi-first map generation.
Implements Lloyd's relaxation for organic point distribution.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import math
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box

@dataclass
class Seed:
    """A single Voronoi seed."""
    x: float
    y: float
    area_id: str = "VOID"
    is_boundary: bool = False

def generate_relaxed_seeds(
    bounds: Tuple[float, float, float, float],
    num_seeds: int,
    iterations: int = 5,
    seed: Optional[int] = None
) -> List[Seed]:
    """
    Generates a set of seeds distributed organically using Lloyd's relaxation.

    Args:
        bounds: (minx, miny, maxx, maxy)
        num_seeds: Target number of seeds
        iterations: Number of relaxation steps
        seed: Random seed
    """
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny

    # Initial random distribution
    points = rng.uniform(
        low=[minx, miny],
        high=[maxx, maxy],
        size=(num_seeds, 2)
    )

    # Lloyd's Relaxation
    for _ in range(iterations):
        # Add dummy points to constrain the diagram
        # We place points far away in 4 directions
        dummy_points = np.array([
            [minx - width, miny - height],
            [minx - width, maxy + height],
            [maxx + width, miny - height],
            [maxx + width, maxy + height],
            [minx - width, (miny + maxy)/2],
            [maxx + width, (miny + maxy)/2],
            [(minx + maxx)/2, miny - height],
            [(minx + maxx)/2, maxy + height],
        ])

        all_points = np.vstack([points, dummy_points])
        vor = Voronoi(all_points)

        new_points = []
        for i in range(len(points)):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]

            if -1 in region or not region:
                # Should not happen for inner points with dummy points around
                new_points.append(points[i])
                continue

            # Get vertices of the region
            vertices = vor.vertices[region]

            # Compute centroid
            # Polygon centroid formula is better, but mean of vertices is a good approximation for relaxation
            centroid = np.mean(vertices, axis=0)

            # Clamp to bounds to ensure we don't drift out
            centroid[0] = np.clip(centroid[0], minx, maxx)
            centroid[1] = np.clip(centroid[1], miny, maxy)

            new_points.append(centroid)

        points = np.array(new_points)

    # Convert to Seed objects
    seeds = [Seed(p[0], p[1]) for p in points]

    # Add boundary seeds to ensure finite Voronoi cells for all inner seeds
    # Calculate dynamic step based on density to match inner cells
    area = width * height
    if num_seeds > 0:
        avg_area_per_seed = area / num_seeds
        avg_spacing = math.sqrt(avg_area_per_seed)
    else:
        avg_spacing = 32.0

    # Use a margin slightly larger than spacing to avoid interference with edge relaxation
    # but close enough to prevent elongation
    step = avg_spacing
    margin = avg_spacing * 1.5

    bx_min = minx - margin
    bx_max = maxx + margin
    by_min = miny - margin
    by_max = maxy + margin

    # Top and Bottom
    for x in np.arange(bx_min, bx_max + step, step):
        seeds.append(Seed(x, by_min, is_boundary=True))
        seeds.append(Seed(x, by_max, is_boundary=True))

    # Left and Right
    for y in np.arange(by_min + step, by_max, step):
        seeds.append(Seed(bx_min, y, is_boundary=True))
        seeds.append(Seed(bx_max, y, is_boundary=True))

    return seeds
