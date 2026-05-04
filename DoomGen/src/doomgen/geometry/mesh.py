"""
Voronoi Mesh handling.
Computes the diagram and manages the topological graph of cells and ridges.
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import heapq
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, LineString, Point

from doomgen.geometry.seeds import Seed
from doomgen.abstraction.layout import LayoutGraph, ConnectionType

class VoronoiMesh:
    """
    Represents the computed Voronoi diagram as a topological mesh.
    Maps cells back to Areas and handles adjacency logic.
    """
    def __init__(self, seeds: List[Seed], graph: Optional[LayoutGraph] = None):
        self.seeds = seeds
        self.graph = graph
        self.vor: Optional[Voronoi] = None

        # Mappings
        self.point_idx_to_seed: Dict[int, Seed] = {}
        self.point_idx_to_area_id: Dict[int, str] = {}

        # Computed Geometry
        self.cells: Dict[int, Polygon] = {} # point_idx -> Polygon
        self.ridges: List[Tuple[int, int, LineString]] = [] # (point_idx_a, point_idx_b, geometry)
        self.adjacency: Dict[int, List[int]] = {}

    def build(self):
        """Compute Voronoi diagram and process topology."""
        if not self.seeds:
            return

        # 1. Prepare points
        points = np.array([[s.x, s.y] for s in self.seeds])

        # 2. Compute Voronoi
        self.vor = Voronoi(points)

        # 3. Map indices
        for i, seed in enumerate(self.seeds):
            self.point_idx_to_seed[i] = seed
            self.point_idx_to_area_id[i] = seed.area_id

        # 4. Build Cells (Polygons)
        # Note: scipy.spatial.Voronoi regions can be -1 (infinite). We skip those or clip them.
        for i, region_idx in enumerate(self.vor.point_region):
            if region_idx == -1: continue

            region = self.vor.regions[region_idx]
            if -1 in region or not region: continue # Infinite region

            vertices = self.vor.vertices[region]
            self.cells[i] = Polygon(vertices)

        # 5. Build Ridges (Edges)
        # ridge_points is a list of pairs of point indices (seeds)
        # ridge_vertices is a list of pairs of vertex indices (corners)
        for (p1, p2), v_indices in zip(self.vor.ridge_points, self.vor.ridge_vertices):
            if -1 in v_indices: continue # Infinite ridge

            v_start = self.vor.vertices[v_indices[0]]
            v_end = self.vor.vertices[v_indices[1]]

            line = LineString([v_start, v_end])
            self.ridges.append((p1, p2, line))

        # 6. Build Adjacency
        self.build_adjacency()

    def build_adjacency(self):
        """Builds the adjacency graph of the mesh."""
        self.adjacency = {}
        for p1, p2, _ in self.ridges:
            if p1 not in self.adjacency: self.adjacency[p1] = []
            if p2 not in self.adjacency: self.adjacency[p2] = []
            self.adjacency[p1].append(p2)
            self.adjacency[p2].append(p1)

    def find_path(self, start_indices: List[int], end_indices: List[int]) -> List[int]:
        """
        Finds a path of cells between start and end sets using A*.
        Returns a list of point indices (seeds) representing the path.
        """
        if not self.adjacency:
            self.build_adjacency()

        start_set = set(start_indices)
        end_set = set(end_indices)

        # Priority queue: (f_score, current_node)
        open_set = []

        # Maps
        came_from = {}
        g_score = {node: float('inf') for node in range(len(self.seeds))}
        f_score = {node: float('inf') for node in range(len(self.seeds))}

        # Initialize start nodes
        for start_node in start_indices:
            g_score[start_node] = 0
            # Heuristic: min distance to any end seed
            h = min(np.linalg.norm(np.array([self.seeds[start_node].x, self.seeds[start_node].y]) -
                                   np.array([self.seeds[end_node].x, self.seeds[end_node].y]))
                    for end_node in end_indices)
            f_score[start_node] = h
            heapq.heappush(open_set, (h, start_node))

        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in end_set:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.adjacency.get(current, []):
                if neighbor in visited:
                    continue

                # Cost calculation
                dist = np.linalg.norm(np.array([self.seeds[current].x, self.seeds[current].y]) -
                                      np.array([self.seeds[neighbor].x, self.seeds[neighbor].y]))

                # Penalty for passing through existing non-VOID rooms (unless it's the destination)
                penalty = 0
                if self.seeds[neighbor].area_id != "VOID" and neighbor not in end_set:
                    penalty = 1000

                tentative_g = g_score[current] + dist + penalty

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    h = min(np.linalg.norm(np.array([self.seeds[neighbor].x, self.seeds[neighbor].y]) -
                                           np.array([self.seeds[end_node].x, self.seeds[end_node].y]))
                            for end_node in end_indices)

                    f_score[neighbor] = tentative_g + h
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def get_ridges_for_translation(self) -> List[Dict]:
        """
        Analyze ridges to determine what they represent in Doom (Walls, Portals, etc.)
        Returns a list of ridge data objects.
        """
        processed_ridges = []

        for p1, p2, line in self.ridges:
            area_id_a = self.point_idx_to_area_id[p1]
            area_id_b = self.point_idx_to_area_id[p2]

            # Case 0: Void Ridge (Outer Wall)
            if area_id_a == "VOID" and area_id_b == "VOID":
                continue # Void-to-Void is invisible

            ridge_type = "solid"
            connection = None

            if area_id_a == "VOID" or area_id_b == "VOID":
                # One side is void -> Map Boundary
                ridge_type = "void_wall"
            elif area_id_a == area_id_b:
                # Internal Ridge -> Skip
                continue
            else:
                # Boundary Ridge (Different Areas)
                connection = self._find_connection(area_id_a, area_id_b)
                if connection:
                    if connection.type == ConnectionType.OPEN:
                        ridge_type = "portal"
                    elif connection.type == ConnectionType.DOOR:
                        ridge_type = "portal"
                    elif connection.type == ConnectionType.WINDOW:
                        ridge_type = "portal"
                    elif connection.type == ConnectionType.SOLID:
                        ridge_type = "solid"

            # Determine orientation
            # We want to identify which area is on the "Front" (Right) side of the line.
            v_start_coords = line.coords[0]
            v_end_coords = line.coords[1]
            x1, y1 = v_start_coords[0], v_start_coords[1]
            x2, y2 = v_end_coords[0], v_end_coords[1]

            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1

            # Right normal vector
            nx, ny = dy, -dx

            # Test point on the right side
            tx, ty = mx + nx, my + ny

            # Seeds
            s_a = self.point_idx_to_seed[p1]
            s_b = self.point_idx_to_seed[p2]

            # Distances squared to test point
            dist_a = (tx - s_a.x)**2 + (ty - s_a.y)**2
            dist_b = (tx - s_b.x)**2 + (ty - s_b.y)**2

            if dist_a < dist_b:
                front_area = area_id_a
                back_area = area_id_b
            else:
                front_area = area_id_b
                back_area = area_id_a

            # Ensure Front is not VOID if possible (for 1-sided walls)
            if front_area == "VOID" and back_area != "VOID":
                # Flip line so valid area is on Front
                line = LineString([line.coords[1], line.coords[0]])
                front_area, back_area = back_area, front_area

            processed_ridges.append({
                "line": line,
                "front_area": front_area,
                "back_area": back_area,
                "type": ridge_type,
                "connection": connection
            })

        return processed_ridges

    def _find_connection(self, id_a: str, id_b: str):
        for conn in self.graph.connections:
            if (conn.area_a_id == id_a and conn.area_b_id == id_b) or \
               (conn.area_a_id == id_b and conn.area_b_id == id_a):
                return conn
        return None
