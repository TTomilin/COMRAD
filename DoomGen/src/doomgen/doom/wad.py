"""WAD file creation and manipulation via omgifol."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional, Any
from pathlib import Path

# omgifol imports - the actual WAD manipulation library
try:
    from omg import WAD, MapEditor as OMGMapEditor
    from omg.mapedit import Vertex, Linedef, Sidedef, Sector, Thing
    from omg.mapedit import ZLinedef, ZThing  # Hexen format for ACS support
    OMGIFOL_AVAILABLE = True
except ImportError:
    OMGIFOL_AVAILABLE = False
    WAD = None


class WADWriter:
    """High-level wrapper for WAD file creation."""

    def __init__(self, wad_type: str = "PWAD"):
        """
        Initialize a new WAD file.

        Args:
            wad_type: Either "PWAD" (patch) or "IWAD" (main).
                     PWAD is appropriate for custom maps.
        """
        if not OMGIFOL_AVAILABLE:
            raise ImportError(
                "omgifol is required for WAD creation. "
                "Install with: pip install omgifol"
            )

        self.wad = WAD()
        self.wad_type = wad_type
        self.maps: dict[str, Any] = {}

    def create_map(self, name: str = "MAP01") -> 'DoomMapData':
        """
        Create a new map in the WAD.

        Args:
            name: Map name (e.g., "MAP01" for Doom 2, "E1M1" for Doom 1).

        Returns:
            DoomMapData object for adding map elements.
        """
        if name in self.maps:
            raise ValueError(f"Map {name} already exists")

        map_data = DoomMapData(name)
        self.maps[name] = map_data
        return map_data

    def get_map(self, name: str) -> 'DoomMapData':
        """
        Get an existing map by name.

        Args:
            name: Map name.

        Returns:
            DoomMapData object.
        """
        if name not in self.maps:
            raise KeyError(f"Map {name} does not exist")
        return self.maps[name]

    def add_lump(self, name: str, data: bytes) -> None:
        """
        Add a global lump to the WAD.

        Args:
            name: Name of the lump (e.g., "DECORATE").
            data: Binary data of the lump.
        """
        from omg.lump import Lump
        self.wad.data[name] = Lump(data)

    def save(self, filepath: str | Path) -> None:
        """
        Save the WAD to a file.

        Args:
            filepath: Output file path (should end in .wad).
        """
        filepath = Path(filepath)

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Build all maps into the WAD
        for name, map_data in self.maps.items():
            self._build_map(name, map_data)

        # Save to file
        self.wad.to_file(str(filepath))

    def _build_map(self, name: str, map_data: 'DoomMapData') -> None:
        """
        Build a map into the WAD structure.

        This converts DoomMapData into omgifol's internal format.
        """
        editor = OMGMapEditor()

        # Determine if we need Hexen format (for ACS support)
        use_hexen_format = map_data.behavior is not None

        if use_hexen_format:
            editor.Linedef = ZLinedef
            editor.Thing = ZThing

        # Add vertices
        for v in map_data.vertices:
            editor.vertexes.append(Vertex(x=v['x'], y=v['y']))

        # Add sectors
        for s in map_data.sectors:
            sector_kwargs = {
                'z_floor': s.get('floor_height', 0),
                'z_ceil': s.get('ceiling_height', 128),
                'tx_floor': s.get('floor_texture', 'FLOOR4_8'),
                'tx_ceil': s.get('ceiling_texture', 'CEIL3_5'),
                'light': s.get('light_level', 160),
                'type': s.get('special', 0),
                'tag': s.get('tag', 0)
            }
             # Add any extra properties provided by user (e.g. aliases)
            standard_keys = {'floor_height', 'ceiling_height', 'floor_texture', 'ceiling_texture', 'light_level', 'special', 'tag'}
            for k, v in s.items():
                if k not in standard_keys:
                    sector_kwargs[k] = v

            editor.sectors.append(Sector(**sector_kwargs))

        # Add sidedefs
        for sd in map_data.sidedefs:
            side_kwargs = {
                'off_x': sd.get('x_offset', 0),
                'off_y': sd.get('y_offset', 0),
                'tx_up': sd.get('upper_texture', '-') if sd.get('upper_texture') != '-' else '-',
                'tx_low': sd.get('lower_texture', '-') if sd.get('lower_texture') != '-' else '-',
                'tx_mid': sd.get('middle_texture', 'STARTAN2') if sd.get('middle_texture') != '-' else '-',
                'sector': sd.get('sector', 0)
            }
            # Add extra properties
            standard_keys = {'x_offset', 'y_offset', 'upper_texture', 'lower_texture', 'middle_texture', 'sector'}
            for k, v in sd.items():
                if k not in standard_keys:
                    side_kwargs[k] = v

            editor.sidedefs.append(Sidedef(**side_kwargs))

        # Add linedefs
        if use_hexen_format:
            for ld in map_data.linedefs:
                editor.linedefs.append(ZLinedef(
                    vx_a=ld['start_vertex'],
                    vx_b=ld['end_vertex'],
                    flags=ld.get('flags', 1),
                    action=ld.get('special', 0),
                    arg0=ld.get('arg0', 0),
                    arg1=ld.get('arg1', 0),
                    arg2=ld.get('arg2', 0),
                    arg3=ld.get('arg3', 0),
                    arg4=ld.get('arg4', 0),
                    front=ld.get('front_sidedef', -1),
                    back=ld.get('back_sidedef', -1)
                ))
        else:
            for ld in map_data.linedefs:
                editor.linedefs.append(Linedef(
                    vx_a=ld['start_vertex'],
                    vx_b=ld['end_vertex'],
                    flags=ld.get('flags', 1),
                    action=ld.get('special', 0),
                    tag=ld.get('tag', 0),
                    front=ld.get('front_sidedef', -1),
                    back=ld.get('back_sidedef', -1)
                ))

        # Add things
        if use_hexen_format:
            # Hexen format thing flags:
            # Bits 0-2: Skill levels (1=Easy, 2=Medium, 4=Hard) -> 7 = all skills
            # Bits 8-10: Game mode appearance (256=SP, 512=Coop, 1024=DM)
            # Default: 7 | 256 | 512 | 1024 = 1799 = appear in all modes
            HEXEN_DEFAULT_FLAGS = 7 | 256 | 512 | 1024  # 1799
            for t in map_data.things:
                thing = ZThing()
                thing.x = t['x']
                thing.y = t['y']
                thing.type = t['type']
                thing.angle = t.get('angle', 0)
                # Use provided flags, or default to all skills + all game modes
                f = t.get('flags', HEXEN_DEFAULT_FLAGS)
                # Fix: If flags are standard Doom flags (low values), they lack Hexen game mode bits.
                # If no game mode bits (256, 512, 1024) are set, assume we want all modes.
                if f < 256:
                    f |= (256 | 512 | 1024)
                thing.flags = f

                thing.tid = t.get('tid', 0)
                thing.height = t.get('height', 0)
                thing.action = t.get('action', 0)
                thing.arg0 = t.get('arg0', 0)
                thing.arg1 = t.get('arg1', 0)
                thing.arg2 = t.get('arg2', 0)
                thing.arg3 = t.get('arg3', 0)
                thing.arg4 = t.get('arg4', 0)
                editor.things.append(thing)
        else:
            for t in map_data.things:
                editor.things.append(Thing(
                    x=t['x'],
                    y=t['y'],
                    angle=t.get('angle', 0),
                    type=t['type'],
                    flags=t.get('flags', 7)
                ))

        # This is docs from omgifol MapEditor:
        #   Currently present but unused:
        #       segs          List containing Seg objects
        #       ssectors      List containing SubSector objects
        #       nodes         List containing Node objects
        #       blockmap      Lump object containing blockmap data
        #       reject        Lump object containing reject table data
        #       (These five lumps are not updated when saving; you will need to use
        #       an external node builder utility)

        # This doesnt build BSP nodes, ZDoom's internal ZDBSP rebuilds them at load time
        # SLADE also ignores many of these https://github.com/sirjuddington/SLADE/blob/030cab09eb2108c65b47c088d1ce97d27e671b9a/src/MapEditor/MapBackupManager.cpp#L55

        map_lumps = editor.to_lumps()

        # Add BEHAVIOR lump if we have compiled ACS
        if map_data.behavior:
            from omg.lump import Lump
            map_lumps['BEHAVIOR'] = Lump(map_data.behavior)

        # Add SCRIPTS lump if we have ACS source code
        if map_data.scripts:
            from omg.lump import Lump
            # Encode source as bytes (null-terminated for WAD compatibility)
            scripts_data = map_data.scripts.encode('utf-8') + b'\x00'
            map_lumps['SCRIPTS'] = Lump(scripts_data)

        self.wad.maps[name] = map_lumps


def run_zdbsp(wad_path: str | Path, zdbsp_path: str = "zdbsp") -> bool:
    """
    "ZDBSP is ZDoom's (internal and external) node builder. This node builder was "
    "written with two design goals in mind: speed and minimization of polyobject "
    "bleeding." - from zdbsp.c

    Run ZDBSP on a WAD file to build BSP nodes, blockmap, and reject table.
    This is an optional post-processing step. ZDoom 2.8.1+ will rebuild nodes at load time if the lumps are empty, but pre-building with ZDBSP can catch geometry errors early and avoids runtime nodebuilding overhead.

    :param wad_path: Path to the WAD file to process (modified in-place).
    :param zdbsp_path: Path to the ZDBSP executable (default: "zdbsp" on PATH).
    :returns: True if ZDBSP ran successfully, False if ZDBSP was not found.
    :raises subprocess.CalledProcessError: If ZDBSP exits with a non-zero status (degenerate geometry that cannot be noded)
    """
    wad_path = Path(wad_path)
    resolved = shutil.which(zdbsp_path)
    if resolved is None: return False

    # ZDBSP overwrites in-place with -o pointing to the same file. Use tmp file
    tmp_path = wad_path.with_suffix(".zdbsp_tmp.wad")
    try:
        subprocess.run(
            [resolved, "-o", tmp_path, wad_path],
            capture_output=True,
            text=True,
            check=True,
        )
        # Replace original with noded version
        tmp_path.replace(wad_path) # This removes tmp_path from the filesystem
        return True
    finally:
        # If subprocess fails, tmp_path still exists and gets deleted
        # If subprocess succeeds, replace() already removed it, so this does nothing
        if tmp_path.exists():
            tmp_path.unlink()


class DoomMapData:
    """
    Container for map geometry data before WAD compilation.

    This is an intermediate representation that holds vertices,
    linedefs, sidedefs, sectors, and things before they are
    written to the WAD file.
    """

    def __init__(self, name: str):
        """
        Initialize map data container.

        Args:
            name: Map name (e.g., "MAP01").
        """
        self.name = name
        self.vertices: list[dict] = []
        self.linedefs: list[dict] = []
        self.sidedefs: list[dict] = []
        self.sectors: list[dict] = []
        self.things: list[dict] = []
        self.behavior: Optional[bytes] = None  # Compiled ACS bytecode
        self.scripts: Optional[str] = None  # ACS source code

        # Vertex deduplication index
        self._vertex_index: dict[tuple[int, int], int] = {}

    def add_vertex(self, x: int, y: int) -> int:
        """
        Add a vertex, deduplicating if it already exists.

        Args:
            x: X coordinate (integer).
            y: Y coordinate (integer).

        Returns:
            Index of the vertex.
        """
        # Ensure integer coordinates
        x, y = int(round(x)), int(round(y))
        key = (x, y)

        if key in self._vertex_index:
            return self._vertex_index[key]

        index = len(self.vertices)
        self.vertices.append({'x': x, 'y': y})
        self._vertex_index[key] = index
        return index

    def add_sector(
        self,
        floor_height: int = 0,
        ceiling_height: int = 128,
        floor_texture: str = 'FLOOR4_8',
        ceiling_texture: str = 'CEIL3_5',
        light_level: int = 160,
        special: int = 0,
        tag: int = 0,
        **kwargs
    ) -> int:
        """
        Add a sector.

        Args:
            floor_height: Floor height in map units.
            ceiling_height: Ceiling height in map units.
            floor_texture: Floor texture name.
            ceiling_texture: Ceiling texture name.
            light_level: Light level (0-255).
            special: Sector special type.
            tag: Sector tag for scripting.
            **kwargs: Additional properties (e.g. for specific backend mappings).

        Returns:
            Index of the sector.
        """
        index = len(self.sectors)
        sector_data = {
            'floor_height': floor_height,
            'ceiling_height': ceiling_height,
            'floor_texture': floor_texture,
            'ceiling_texture': ceiling_texture,
            'light_level': light_level,
            'special': special,
            'tag': tag
        }
        sector_data.update(kwargs)
        self.sectors.append(sector_data)
        return index

    def add_sidedef(
        self,
        sector: int,
        upper_texture: str = '-',
        middle_texture: str = 'STARTAN2',
        lower_texture: str = '-',
        x_offset: int = 0,
        y_offset: int = 0,
        **kwargs
    ) -> int:
        """
        Add a sidedef.

        Args:
            sector: Index of the sector this sidedef faces.
            upper_texture: Upper texture name.
            middle_texture: Middle texture name.
            lower_texture: Lower texture name.
            x_offset: Texture X offset.
            y_offset: Texture Y offset.
            **kwargs: Additional properties.

        Returns:
            Index of the sidedef.
        """
        index = len(self.sidedefs)
        side_data = {
            'sector': sector,
            'upper_texture': upper_texture,
            'middle_texture': middle_texture,
            'lower_texture': lower_texture,
            'x_offset': x_offset,
            'y_offset': y_offset
        }
        side_data.update(kwargs)
        self.sidedefs.append(side_data)
        return index

    def add_linedef(
        self,
        start_vertex: int,
        end_vertex: int,
        front_sidedef: int,
        back_sidedef: int = -1,
        flags: int = 1,
        special: int = 0,
        tag: int = 0,
        args: Optional[list[int]] = None
    ) -> int:
        """
        Add a linedef.

        Args:
            start_vertex: Index of start vertex.
            end_vertex: Index of end vertex.
            front_sidedef: Index of front sidedef.
            back_sidedef: Index of back sidedef (-1 for none).
            flags: Linedef flags (1=impassable, etc.).
            special: Linedef special type.
            tag: Linedef tag for scripting.
            args: List of arguments for Hexen format specials (arg0-arg4).

        Returns:
            Index of the linedef.
        """
        index = len(self.linedefs)
        ld = {
            'start_vertex': start_vertex,
            'end_vertex': end_vertex,
            'front_sidedef': front_sidedef,
            'back_sidedef': back_sidedef,
            'flags': flags,
            'special': special,
            'tag': tag
        }
        if args:
            for i, arg in enumerate(args):
                if i < 5:
                    ld[f'arg{i}'] = arg

        self.linedefs.append(ld)
        return index

    def add_thing(
        self,
        x: int,
        y: int,
        thing_type: int,
        angle: int = 0,
        flags: int = 7,
        tid: int = 0,
        **kwargs
    ) -> int:
        """
        Add a thing (entity).

        Args:
            x: X coordinate.
            y: Y coordinate.
            thing_type: Doom thing type ID.
            angle: Facing angle in degrees (0=East, 90=North).
            flags: Thing flags (skill level appearance).
            tid: Thing ID for ACS scripts (Hexen/ZDoom format only).
            **kwargs: Additional properties (action, args, etc.) for Hexen format.

        Returns:
            Index of the thing.
        """
        index = len(self.things)
        thing_data = {
            'x': int(round(x)),
            'y': int(round(y)),
            'type': thing_type,
            'angle': angle,
            'flags': flags,
            'tid': tid
        }

        # Handle args list if provided (unpack to arg0-4)
        if 'args' in kwargs:
            args_list = kwargs.pop('args')
            for i, arg in enumerate(args_list):
                if i < 5:
                    kwargs[f'arg{i}'] = arg

        thing_data.update(kwargs)
        self.things.append(thing_data)
        return index
