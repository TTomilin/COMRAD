"""High-level API for procedural Doom map generation."""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon, box

from doomgen.abstraction.layout import Area, AreaConfig, ConnectionType, LayoutGraph, Transform
from doomgen.doom.things import ThingType
from doomgen.doom.translator import MapTranslator
from doomgen.doom.wad import WADWriter, run_zdbsp
from doomgen.geometry.mesh import VoronoiMesh
from doomgen.geometry.seeds import generate_relaxed_seeds
from doomgen.utils.paths import get_bundled_compiler, validate_executable


class ProceduralMapBuilder:
    """Build Doom maps from a relaxed Voronoi cell layout."""
    def __init__(self, bounds: tuple[float, float, float, float] = (-1024, -1024, 1024, 1024),
                 num_seeds: int = 2000, seed: int = 42):
        self.graph = LayoutGraph()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.wad_writer = WADWriter()
        self.map_data = self.wad_writer.create_map("MAP01")
        self.things_to_add: list[dict[str, Any]] = []
        self.name_to_id: dict[str, str] = {}
        self._used_cells: set[int] = set()

        self.seeds = generate_relaxed_seeds(bounds, num_seeds, seed=seed)

        self.mesh = VoronoiMesh(self.seeds, self.graph)
        self.mesh.build()

    def add_area(
        self,
        name: str,
        shape: Polygon | tuple[float, float, float, float],
        floor_height: int = 0,
        ceiling_height: int = 128,
        floor_texture: str = "FLOOR4_8",
        ceiling_texture: str = "CEIL3_5",
        wall_texture: str = "STARTAN3",
        middle_texture: str | None = None,
        lower_texture: str | None = None,
        upper_texture: str | None = None,
        light_level: int = 160,
        special: int = 0,
        tag: int = 0,
        linedef_special: int | None = None,
        linedef_args: list[int] | None = None,
        linedef_flags: int | None = None,
        mode: str = "claim_void",
        **kwargs
    ) -> str:
        """Assign Voronoi cells inside `shape` to a new area."""
        if isinstance(shape, tuple) or isinstance(shape, list):
            x, y, w, h = shape
            poly = box(x - w/2, y - h/2, x + w/2, y + h/2)
        else:
            poly = shape

        config = AreaConfig(
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            floor_texture=floor_texture,
            ceiling_texture=ceiling_texture,
            wall_texture=wall_texture,
            middle_texture=middle_texture,
            lower_texture=lower_texture,
            upper_texture=upper_texture,
            light_level=light_level,
            special=special,
            tag=tag,
            linedef_special=linedef_special,
            linedef_args=linedef_args or [],
            linedef_flags=linedef_flags,
            properties=kwargs
        )

        area = Area(
            name=name,
            transform=Transform(0, 0),
            config=config
        )

        area_id = self.graph.add_area(area)
        self.name_to_id[name] = area_id

        count = 0
        for s in self.seeds:
            should_claim = False
            if mode == "overwrite":
                should_claim = True
            elif mode == "claim_void" and s.area_id == "VOID":
                should_claim = True

            if should_claim:
                if poly.contains(Point(s.x, s.y)):
                    s.area_id = area_id
                    count += 1

        # Overwrite areas smaller than the seed spacing can otherwise vanish entirely.
        if count == 0 and mode == "overwrite":
            cx = poly.centroid.x
            cy = poly.centroid.y
            minx, miny, maxx, maxy = poly.bounds
            max_d = max(maxx - minx, maxy - miny) * 2.0
            max_d2 = max_d * max_d

            best_seed = None
            best_d2 = None
            for s in self.seeds:
                dx = s.x - cx
                dy = s.y - cy
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_seed = s

            if best_seed is not None and best_d2 is not None and best_d2 <= max_d2:
                best_seed.area_id = area_id
                count = 1

        if count == 0:
            print(f"Warning: Area {name} claimed 0 seeds. Check bounds/density.")

        return area_id



    def add_corridor(
        self,
        room_a_name: str,
        room_b_name: str,
        width: int = 1,
        floor_height: int = 0,
        ceiling_height: int = 128,
        floor_texture: str = "FLOOR4_8",
        ceiling_texture: str = "CEIL3_5",
        wall_texture: str = "STARTAN3",
        middle_texture: str | None = None,
        lower_texture: str | None = None,
        upper_texture: str | None = None,
        light_level: int = 160,
        special: int = 0,
        tag: int = 0
    ) -> str:
        """Connect two areas with a corridor along the mesh."""
        id_a = self.name_to_id.get(room_a_name, room_a_name)
        id_b = self.name_to_id.get(room_b_name, room_b_name)

        seeds_a = [i for i, s in enumerate(self.seeds) if s.area_id == id_a]
        seeds_b = [i for i, s in enumerate(self.seeds) if s.area_id == id_b]

        if not seeds_a or not seeds_b:
            print(f"Warning: Cannot connect {room_a_name} and {room_b_name}, one has no seeds.")
            return ""

        path_indices = self.mesh.find_path(seeds_a, seeds_b)

        if not path_indices:
            print(f"Warning: No path found between {room_a_name} and {room_b_name}.")
            return ""

        corridor_name = f"corridor_{room_a_name}_{room_b_name}"
        config = AreaConfig(
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            floor_texture=floor_texture,
            ceiling_texture=ceiling_texture,
            wall_texture=wall_texture,
            middle_texture=middle_texture,
            lower_texture=lower_texture,
            upper_texture=upper_texture,
            light_level=light_level,
            special=special,
            tag=tag
        )

        area = Area(name=corridor_name, transform=Transform(0,0), config=config)
        area_id = self.graph.add_area(area)

        path_set = set(path_indices)

        if width > 1:
            for _ in range(width - 1):
                new_neighbors = set()
                for idx in path_set:
                    for neighbor in self.mesh.adjacency.get(idx, []):
                        if self.seeds[neighbor].area_id == "VOID":
                            new_neighbors.add(neighbor)
                path_set.update(new_neighbors)

        for idx in path_set:
            if self.seeds[idx].area_id == "VOID":
                self.seeds[idx].area_id = area_id

        self.graph.connect(id_a, area_id, ConnectionType.OPEN)
        self.graph.connect(area_id, id_b, ConnectionType.OPEN)

        return area_id

    def add_stairs(
        self,
        room_a_name: str,
        room_b_name: str,
        num_steps: int = 4,
        width: int = 1,
        floor_texture: str = "STEP1",
        ceiling_texture: str = "CEIL3_5",
        wall_texture: str = "BROWN1",
        riser_texture: str = "STEP1"
    ) -> list[str]:
        """Connect two areas with a stepped path."""
        id_a = self.name_to_id.get(room_a_name, room_a_name)
        id_b = self.name_to_id.get(room_b_name, room_b_name)

        area_a = self.graph.get_area(id_a)
        area_b = self.graph.get_area(id_b)

        h_start = area_a.config.floor_height
        h_end = area_b.config.floor_height

        seeds_a = [i for i, s in enumerate(self.seeds) if s.area_id == id_a]
        seeds_b = [i for i, s in enumerate(self.seeds) if s.area_id == id_b]

        if not seeds_a or not seeds_b:
            print(f"Warning: Cannot connect {room_a_name} and {room_b_name} with stairs.")
            return []

        path_indices = self.mesh.find_path(seeds_a, seeds_b)
        if not path_indices:
            print(f"Warning: No path found for stairs between {room_a_name} and {room_b_name}.")
            return []

        if len(path_indices) < num_steps:
            print(f"Warning: Path length ({len(path_indices)}) < requested steps ({num_steps}). Reducing steps.")
            num_steps = max(1, len(path_indices))

        cell_to_step = {}
        for i, cell_idx in enumerate(path_indices):
            k = int((i / len(path_indices)) * num_steps)
            k = min(k, num_steps - 1)
            cell_to_step[cell_idx] = k

        current_set = set(path_indices)
        if width > 1:
            for _ in range(width - 1):
                new_neighbors = set()
                for idx in current_set:
                    step_idx = cell_to_step[idx]
                    for neighbor in self.mesh.adjacency.get(idx, []):
                        if self.seeds[neighbor].area_id == "VOID":
                            if neighbor not in cell_to_step:
                                cell_to_step[neighbor] = step_idx
                                new_neighbors.add(neighbor)
                current_set.update(new_neighbors)

        step_ids = []
        for k in range(num_steps):
            t = (k + 1) / num_steps
            h = int(h_start + (h_end - h_start) * t)

            step_name = f"stairs_{room_a_name}_{room_b_name}_step_{k+1}"

            config = AreaConfig(
                floor_height=h,
                ceiling_height=max(area_a.config.ceiling_height, area_b.config.ceiling_height),
                floor_texture=floor_texture,
                ceiling_texture=ceiling_texture,
                wall_texture=wall_texture,
                lower_texture=riser_texture,
                upper_texture=riser_texture
            )

            area = Area(name=step_name, transform=Transform(0,0), config=config)
            area_id = self.graph.add_area(area)
            step_ids.append(area_id)

            for cell_idx, s_k in cell_to_step.items():
                if s_k == k:
                    self.seeds[cell_idx].area_id = area_id

        self.graph.connect(id_a, step_ids[0], ConnectionType.OPEN)

        for k in range(num_steps - 1):
            self.graph.connect(step_ids[k], step_ids[k+1], ConnectionType.OPEN)

        self.graph.connect(step_ids[-1], id_b, ConnectionType.OPEN)

        return step_ids

    def set_void(self, shape: Polygon | tuple[float, float, float, float]):
        """Mark cells inside `shape` as VOID."""
        if isinstance(shape, tuple) or isinstance(shape, list):
            x, y, w, h = shape
            poly = box(x - w/2, y - h/2, x + w/2, y + h/2)
        else:
            poly = shape

        for s in self.seeds:
            if poly.contains(Point(s.x, s.y)):
                s.area_id = "VOID"

    def add_boundary(
        self,
        room_a_name: str,
        room_b_name: str,
        name: str = "boundary",
        floor_height: int | None = None,
        ceiling_height: int | None = None,
        floor_texture: str = "FLOOR4_8",
        ceiling_texture: str = "CEIL3_5",
        wall_texture: str = "BIGDOOR2",
        lower_texture: str | None = None,
        upper_texture: str | None = None,
        light_level: int = 160,
        special: int = 0,
        tag: int = 0,
        connection_type: ConnectionType = ConnectionType.OPEN,
        linedef_special: int | None = None,
        linedef_args: list[int] | None = None,
        linedef_flags: int | None = None
    ) -> str:
        """Create an area from the boundary cells between two existing areas."""
        id_a = self.name_to_id.get(room_a_name, room_a_name)
        id_b = self.name_to_id.get(room_b_name, room_b_name)

        candidates = []
        seeds_a = [i for i, s in enumerate(self.seeds) if s.area_id == id_a]

        for idx in seeds_a:
            for neighbor in self.mesh.adjacency.get(idx, []):
                if self.seeds[neighbor].area_id == id_b:
                    candidates.append(idx)

        if not candidates:
            print(f"Warning: No boundary found between {room_a_name} and {room_b_name}.")
            return ""

        area_a = self.graph.get_area(id_a)

        config = AreaConfig(
            floor_height=floor_height if floor_height is not None else area_a.config.floor_height,
            ceiling_height=ceiling_height if ceiling_height is not None else area_a.config.ceiling_height,
            floor_texture=floor_texture,
            ceiling_texture=ceiling_texture,
            wall_texture=wall_texture,
            lower_texture=lower_texture,
            upper_texture=upper_texture,
            light_level=light_level,
            special=special,
            tag=tag,
            linedef_special=linedef_special,
            linedef_args=linedef_args or [],
            linedef_flags=linedef_flags
        )

        area = Area(name=name, transform=Transform(0,0), config=config)
        area_id = self.graph.add_area(area)
        self.name_to_id[name] = area_id

        for idx in candidates:
            self.seeds[idx].area_id = area_id

        self.graph.connect(id_a, area_id, connection_type)
        self.graph.connect(area_id, id_b, connection_type)

        return area_id

    def connect_adjacent(self, room_a_name: str, room_b_name: str, type: ConnectionType = ConnectionType.OPEN, silent: bool = False, **kwargs):
        """Connect two areas if they share a mesh boundary."""
        id_a = self.name_to_id.get(room_a_name, room_a_name)
        id_b = self.name_to_id.get(room_b_name, room_b_name)

        is_adjacent = False
        seeds_a = [i for i, s in enumerate(self.seeds) if s.area_id == id_a]
        for idx in seeds_a:
            for neighbor in self.mesh.adjacency.get(idx, []):
                if self.seeds[neighbor].area_id == id_b:
                    is_adjacent = True
                    break
            if is_adjacent: break

        if is_adjacent:
            self.graph.connect(id_a, id_b, type, **kwargs)
        elif not silent:
            print(f"Warning: {room_a_name} and {room_b_name} are not adjacent.")

    def add_thing(self, thing_type: str, x: float, y: float, **kwargs):
        """Adds a thing to the map."""
        self.things_to_add.append({
            "type": thing_type,
            "x": x,
            "y": y,
            **kwargs
        })

    def get_random_point_in_area(self, area_name_or_id: str, safe: bool = True) -> tuple[float, float] | None:
        """Return a random point from an area, preferring interior cells."""
        # Narrow corridors can have only one or two "safe" internal cells. Track
        # previously used cells so repeated placement does not stack actors.
        area_id = self.name_to_id.get(area_name_or_id, area_name_or_id)

        candidate_indices = [i for i, s in enumerate(self.seeds) if s.area_id == area_id]

        if not candidate_indices:
            return None

        if safe:
            safe_indices = []
            for idx in candidate_indices:
                is_internal = True
                neighbors = self.mesh.adjacency.get(idx, [])
                for n_idx in neighbors:
                    if self.seeds[n_idx].area_id != area_id:
                        is_internal = False
                        break
                if is_internal:
                    safe_indices.append(idx)

            if safe_indices:
                unused = [i for i in safe_indices if i not in self._used_cells]
                if not unused:
                    self._used_cells -= set(safe_indices)
                    unused = safe_indices
                idx = self.rng.choice(unused)
                self._used_cells.add(idx)
                s = self.seeds[idx]
                return (s.x, s.y)

        unused = [i for i in candidate_indices if i not in self._used_cells]
        if not unused:
            self._used_cells -= set(candidate_indices)
            unused = candidate_indices
        idx = self.rng.choice(unused)
        self._used_cells.add(idx)
        s = self.seeds[idx]
        return (s.x, s.y)

    def get_area_centroid(self, area_name_or_id: str) -> tuple[float, float] | None:
        """Returns the centroid (average x, y) of the area."""
        area_id = self.name_to_id.get(area_name_or_id, area_name_or_id)
        seeds = [s for s in self.seeds if s.area_id == area_id]
        if not seeds:
            return None
        avg_x = sum(s.x for s in seeds) / len(seeds)
        avg_y = sum(s.y for s in seeds) / len(seeds)
        return (avg_x, avg_y)

    def build(self, output_path: str, zdbsp_path: str | None = None):
        """Generate the WAD file."""
        for i, seed in enumerate(self.seeds):
            self.mesh.point_idx_to_area_id[i] = seed.area_id

        use_hexen = self.map_data.behavior is not None
        translator = MapTranslator(self.map_data, use_hexen_format=use_hexen)
        translator.translate(self.mesh, self.graph)

        for thing in self.things_to_add:
            t_type = 2001
            raw_type = thing["type"]

            if isinstance(raw_type, int):
                t_type = raw_type
            elif isinstance(raw_type, str):
                if hasattr(ThingType, raw_type):
                    t_type = getattr(ThingType, raw_type).value
                elif raw_type.isdigit():
                    t_type = int(raw_type)
                else:
                    print(f"Warning: Unknown thing type '{raw_type}', defaulting to Imp (2001)")

            x = thing["x"]
            y = thing["y"]
            angle = thing.get("angle", 0)
            flags = thing.get("flags", 7)

            extra_args = {k: v for k, v in thing.items() if k not in ["type", "x", "y", "angle", "flags"]}

            self.map_data.add_thing(
                x=x,
                y=y,
                thing_type=t_type,
                angle=angle,
                flags=flags,
                **extra_args
            )

        self.wad_writer.save(output_path)
        print(f"WAD written to {output_path}")

        if zdbsp_path is None:
            zdbsp_path = get_bundled_compiler("zdbsp")

        if zdbsp_path is not None and validate_executable(zdbsp_path):
            found = run_zdbsp(output_path, zdbsp_path=zdbsp_path)
            if found:
                print(f"ZDBSP: nodes built successfully for {output_path}")
            else:
                print(f"Warning: ZDBSP not found at '{zdbsp_path}'.")
        elif zdbsp_path is not None:
            print(f"Warning: ZDBSP not found at '{zdbsp_path}'. "
                  "WAD saved with empty node lumps (ZDoom will rebuild at load time).")
