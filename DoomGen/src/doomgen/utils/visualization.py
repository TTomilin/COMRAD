"""
Visualization tools for DoomGen maps.
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import hashlib

if TYPE_CHECKING:
    from doomgen.geometry.mesh import VoronoiMesh

def visualize_mesh(
    mesh: 'VoronoiMesh',
    output_path: str,
    show_labels: bool = True
) -> None:
    """
    Visualize the VoronoiMesh.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Process all cells
    for point_idx, polygon in mesh.cells.items():
        if polygon.is_empty: continue

        area_id = mesh.point_idx_to_area_id[point_idx]

        # Color based on area_id
        if area_id == "VOID":
            color = "black"
            alpha = 0.1
            edgecolor = "gray"
            linewidth = 0.5
        else:
            # Hash area_id to get a consistent color
            h = int(hashlib.sha256(area_id.encode('utf-8')).hexdigest(), 16)
            r = ((h & 0xFF) / 255.0) * 0.5 + 0.5 # Pastel
            g = (((h >> 8) & 0xFF) / 255.0) * 0.5 + 0.5
            b = (((h >> 16) & 0xFF) / 255.0) * 0.5 + 0.5
            color = (r, g, b)
            alpha = 0.8
            edgecolor = "black"
            linewidth = 1.0

        coords = list(polygon.exterior.coords)
        mpl_poly = MplPolygon(coords, closed=True, facecolor=color, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
        ax.add_patch(mpl_poly)

        # Label
        if show_labels and area_id != "VOID":
            centroid = polygon.centroid
            # Use seed index as label or area name
            ax.text(centroid.x, centroid.y, str(point_idx), fontsize=6, ha='center', va='center', color='black')

    # Plot seeds
    xs = [s.x for s in mesh.seeds]
    ys = [s.y for s in mesh.seeds]
    ax.scatter(xs, ys, s=2, c='red', alpha=0.5)

    # Calculate bounds from non-VOID cells
    minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
    has_valid_cells = False

    for point_idx, polygon in mesh.cells.items():
        if polygon.is_empty: continue
        area_id = mesh.point_idx_to_area_id[point_idx]
        if area_id == "VOID": continue

        has_valid_cells = True
        b = polygon.bounds
        minx = min(minx, b[0])
        miny = min(miny, b[1])
        maxx = max(maxx, b[2])
        maxy = max(maxy, b[3])

    if has_valid_cells:
        margin = 64
        ax.set_xlim(minx - margin, maxx + margin)
        ax.set_ylim(miny - margin, maxy + margin)
    else:
        ax.autoscale()

    ax.set_aspect('equal')
    plt.title(f"Voronoi Mesh ({len(mesh.cells)} cells)")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
