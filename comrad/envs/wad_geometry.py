"""Static level geometry read directly from a WAD (Doom or Hexen map format).

Used by the global-state observation mode. ViZDoom 1.3.0 copies each sector's lines into a fixed
``lines[128]`` array without a bounds check, so enabling ``sectors_info`` segfaults on maps with larger
sectors (e.g. Co-op Puzzle). The layout is therefore always taken from the WAD; the engine is only asked for
live sector heights (door state) when every sector stays within that limit and the map is small enough that
the per-frame sector export (every line of every sector) does not dominate env throughput.
"""

import os
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

VIZ_SECTOR_LINE_LIMIT = 128
LIVE_SECTORS_MAX_LINES = 1024  # Rhythm Sync (~4k lines) halves env FPS with live sectors, Armory Siege (366) costs ~5%
PLAYER_HEIGHT = 56.0  # sectors with less headroom than this cannot be entered (closed doors, raised walls)

_ML_BLOCKING = 0x0001
_NO_SIDEDEF = 0xFFFF
_MAP_LUMPS = ("THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SECTORS")


@dataclass(frozen=True)
class WadGeometry:
    segments: np.ndarray  # float32 (L, 4): x1, y1, x2, y2
    impassable: np.ndarray  # bool (L,): one-sided or flagged blocking
    front: np.ndarray  # int32 (L,): front sector index
    back: np.ndarray  # int32 (L,): back sector index, -1 if one-sided
    floor: np.ndarray  # float32 (S,): floor heights at map start
    ceiling: np.ndarray  # float32 (S,): ceiling heights at map start
    max_lines_per_sector: int

    @property
    def num_sectors(self) -> int:
        return int(self.floor.shape[0])

    @property
    def supports_live_sectors(self) -> bool:
        return (
            self.max_lines_per_sector <= VIZ_SECTOR_LINE_LIMIT
            and self.segments.shape[0] <= LIVE_SECTORS_MAX_LINES
        )

    def closed_sectors(self, floor=None, ceiling=None) -> np.ndarray:
        floor = self.floor if floor is None else np.asarray(floor, dtype=np.float32)
        ceiling = self.ceiling if ceiling is None else np.asarray(ceiling, dtype=np.float32)
        return (ceiling - floor) < PLAYER_HEIGHT

    def lines_array(self, closed: np.ndarray) -> np.ndarray:
        """float32 (L, 5): x1, y1, x2, y2, blocked (wall, impassable line, or bordering a closed sector)."""
        closed = np.asarray(closed, dtype=bool)
        blocked = self.impassable | closed[self.front]
        two_sided = self.back >= 0
        blocked[two_sided] |= closed[self.back[two_sided]]
        return np.concatenate([self.segments, blocked[:, None].astype(np.float32)], axis=1)


def _read_directory(data: bytes):
    _ident, num_lumps, dir_offset = struct.unpack_from("<4sii", data, 0)
    lumps = []
    for i in range(num_lumps):
        offset, size, name = struct.unpack_from("<ii8s", data, dir_offset + 16 * i)
        lumps.append((name.rstrip(b"\0").decode("ascii", errors="replace").upper(), offset, size))
    return lumps


def _find_map(lumps, map_name: Optional[str]):
    for i, (name, _offset, _size) in enumerate(lumps):
        if map_name is not None and name != map_name.upper():
            continue
        if i + 1 < len(lumps) and lumps[i + 1][0] in ("THINGS", "TEXTMAP"):
            found = {}
            for lump_name, offset, size in lumps[i + 1 : i + 12]:
                if lump_name in found:
                    break
                found[lump_name] = (offset, size)
            return name, found
    raise ValueError(f"Map {map_name or '<first>'} not found in WAD")


@lru_cache(maxsize=64)
def _load_cached(wad_path: str, map_name: Optional[str], _mtime: float) -> WadGeometry:
    with open(wad_path, "rb") as f:
        data = f.read()
    _name, lumps = _find_map(_read_directory(data), map_name)
    if "TEXTMAP" in lumps:
        raise ValueError(f"UDMF maps are not supported: {wad_path}")
    missing = [lump for lump in _MAP_LUMPS if lump not in lumps]
    if missing:
        raise ValueError(f"WAD map is missing lumps {missing}: {wad_path}")

    def records(lump, fmt):
        offset, size = lumps[lump]
        rec = struct.calcsize(fmt)
        return [struct.unpack_from(fmt, data, offset + i * rec) for i in range(size // rec)]

    vertices = np.asarray(records("VERTEXES", "<hh"), dtype=np.float32).reshape(-1, 2)
    sides = records("SIDEDEFS", "<hh8s8s8sH")
    sectors = records("SECTORS", "<hh8s8shhh")
    if "BEHAVIOR" in lumps:  # Hexen format: v1, v2, flags, special, args[5], front, back
        linedefs = [(r[0], r[1], r[2], r[9], r[10]) for r in records("LINEDEFS", "<HHHB5BHH")]
    else:  # Doom format: v1, v2, flags, special, tag, front, back
        linedefs = [(r[0], r[1], r[2], r[5], r[6]) for r in records("LINEDEFS", "<HHHHHHH")]

    num_sectors = len(sectors)
    segs, impassable, front, back = [], [], [], []
    lines_per_sector = np.zeros(num_sectors, dtype=np.int64)
    for v1, v2, flags, side_front, side_back in linedefs:
        if side_front == _NO_SIDEDEF or side_front >= len(sides) or v1 >= len(vertices) or v2 >= len(vertices):
            continue
        f_sec = int(sides[side_front][5])
        b_sec = int(sides[side_back][5]) if side_back != _NO_SIDEDEF and side_back < len(sides) else -1
        if not 0 <= f_sec < num_sectors:
            continue
        if b_sec >= num_sectors:
            b_sec = -1
        segs.append((*vertices[v1], *vertices[v2]))
        impassable.append(b_sec < 0 or bool(flags & _ML_BLOCKING))
        front.append(f_sec)
        back.append(b_sec)
        lines_per_sector[f_sec] += 1
        if b_sec >= 0 and b_sec != f_sec:
            lines_per_sector[b_sec] += 1

    return WadGeometry(
        segments=np.asarray(segs, dtype=np.float32).reshape(-1, 4),
        impassable=np.asarray(impassable, dtype=bool),
        front=np.asarray(front, dtype=np.int32),
        back=np.asarray(back, dtype=np.int32),
        floor=np.asarray([s[0] for s in sectors], dtype=np.float32),
        ceiling=np.asarray([s[1] for s in sectors], dtype=np.float32),
        max_lines_per_sector=int(lines_per_sector.max()) if num_sectors else 0,
    )


def load_wad_geometry(wad_path: str, map_name: Optional[str] = None) -> WadGeometry:
    wad_path = os.path.abspath(wad_path)
    return _load_cached(wad_path, map_name.upper() if map_name else None, os.path.getmtime(wad_path))


def scenario_wad_and_map(config_path: str):
    """(wad_path, map_name) referenced by a ViZDoom .cfg; map_name is None if the cfg does not set doom_map."""
    wad, map_name = None, None
    with open(config_path, "r") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            key = key.lower().replace("_", "")
            if key == "doomscenariopath":
                wad = value
            elif key == "doommap":
                map_name = value
    if wad is None:
        raise ValueError(f"No doom_scenario_path in {config_path}")
    if not os.path.isabs(wad):
        wad = os.path.join(os.path.dirname(os.path.abspath(config_path)), wad)
    return wad, map_name
