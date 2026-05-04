"""Geometry and WAD validation utilities."""

from __future__ import annotations

from typing import Optional, Any, Union
from pathlib import Path

from shapely.geometry import Polygon, Point, MultiPolygon, GeometryCollection
from shapely.validation import make_valid as shapely_make_valid

from doomgen.utils.constants import (
    MIN_SECTOR_AREA,
    MIN_CORRIDOR_WIDTH,
    MAP_MIN_COORD,
    MAP_MAX_COORD,
)


# =============================================================================
# Coordinate Operations (moved from geometry/operations.py)
# =============================================================================

def snap_to_grid(value: float, grid_size: int = 8) -> int:
    """
    Snap a coordinate value to the nearest grid point.

    Doom coordinates should be integers, and aligning to a grid
    (typically 8 units) helps avoid rendering issues.

    Args:
        value: The coordinate value to snap.
        grid_size: Grid spacing (default 8).

    Returns:
        The snapped integer value.
    """
    return int(round(value / grid_size) * grid_size)


def snap_polygon_to_grid(polygon: Polygon, grid_size: int = 8) -> Polygon:
    """
    Snap all vertices of a polygon to grid coordinates.

    Args:
        polygon: The polygon to snap.
        grid_size: Grid spacing (default 8).

    Returns:
        A new polygon with snapped coordinates.
    """
    coords = list(polygon.exterior.coords)
    snapped = [(snap_to_grid(x, grid_size), snap_to_grid(y, grid_size)) for x, y in coords]
    return Polygon(snapped)


def to_integer_coords(polygon: Polygon) -> Polygon:
    """
    Convert polygon coordinates to integers.

    Args:
        polygon: The polygon to convert.

    Returns:
        A new polygon with integer coordinates.
    """
    coords = list(polygon.exterior.coords)
    int_coords = [(int(round(x)), int(round(y))) for x, y in coords]
    return Polygon(int_coords)


def make_valid(geom: Union[Polygon, MultiPolygon, GeometryCollection]) -> Union[Polygon, MultiPolygon]:
    """
    Ensure geometry is valid.

    Wrapper around Shapely's make_valid for consistency.

    Args:
        geom: Input geometry.

    Returns:
        Valid geometry.
    """
    return shapely_make_valid(geom)


def ensure_polygon(geom: Union[Polygon, MultiPolygon, GeometryCollection]) -> Polygon:
    """
    Ensure geometry is a single Polygon.

    When operations produce MultiPolygon or GeometryCollection,
    this extracts the largest polygon.

    Args:
        geom: Input geometry (Polygon, MultiPolygon, or GeometryCollection).

    Returns:
        A single Polygon (the largest if multiple).

    Raises:
        ValueError: If no polygon can be extracted.
    """
    if isinstance(geom, Polygon):
        return geom

    if isinstance(geom, MultiPolygon):
        if len(geom.geoms) == 0:
            raise ValueError("Empty MultiPolygon")
        return max(geom.geoms, key=lambda p: p.area)

    if isinstance(geom, GeometryCollection):
        polygons = [g for g in geom.geoms if isinstance(g, Polygon)]
        if not polygons:
            raise ValueError("GeometryCollection contains no polygons")
        return max(polygons, key=lambda p: p.area)

    raise ValueError(f"Cannot convert {type(geom).__name__} to Polygon")


# =============================================================================
# Validation Functions
# =============================================================================


def validate_polygon(
    polygon: Polygon,
    min_area: float = MIN_SECTOR_AREA,
    check_bounds: bool = True
) -> tuple[bool, list[str]]:
    """
    Validate a polygon for Doom map generation.

    Checks for:
    - Valid Shapely geometry
    - Non-empty area
    - Minimum area threshold
    - No self-intersections
    - Coordinates within map bounds

    Args:
        polygon: The polygon to validate.
        min_area: Minimum required area.
        check_bounds: Whether to check coordinate bounds.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    issues = []

    # Check basic validity
    if not polygon.is_valid:
        issues.append(f"Invalid geometry: {polygon.geom_type}")

    if polygon.is_empty:
        issues.append("Polygon is empty")
        return False, issues

    # Check area
    if polygon.area < min_area:
        issues.append(f"Area too small: {polygon.area:.2f} < {min_area}")

    # Check for self-intersection
    if not polygon.is_simple:
        issues.append("Polygon has self-intersections")

    # Check bounds
    if check_bounds:
        bounds = polygon.bounds
        min_x, min_y, max_x, max_y = bounds

        if min_x < MAP_MIN_COORD or max_x > MAP_MAX_COORD:
            issues.append(f"X coordinates out of bounds: [{min_x}, {max_x}]")
        if min_y < MAP_MIN_COORD or max_y > MAP_MAX_COORD:
            issues.append(f"Y coordinates out of bounds: [{min_y}, {max_y}]")

    # Check vertex count
    num_vertices = len(polygon.exterior.coords) - 1  # Exclude closing point
    if num_vertices < 3:
        issues.append(f"Too few vertices: {num_vertices}")

    return len(issues) == 0, issues


def validate_corridor_width(
    polygon: Polygon,
    min_width: float = MIN_CORRIDOR_WIDTH
) -> tuple[bool, float]:
    """
    Check if a corridor polygon is wide enough.

    Uses the polygon's buffer erosion to estimate minimum width.

    Args:
        polygon: The corridor polygon.
        min_width: Minimum required width.

    Returns:
        Tuple of (is_valid, estimated_min_width).
    """
    # Erode by half the minimum width
    eroded = polygon.buffer(-min_width / 2)

    if eroded.is_empty:
        # Polygon is narrower than min_width
        # Estimate actual width by binary search
        low, high = 0.0, min_width
        while high - low > 1.0:
            mid = (low + high) / 2
            test = polygon.buffer(-mid / 2)
            if test.is_empty:
                high = mid
            else:
                low = mid

        return False, low

    return True, min_width


def validate_thing_placement(
    x: float,
    y: float,
    polygon: Polygon,
    radius: float = 16.0
) -> tuple[bool, str]:
    """
    Validate that a thing can be placed at the given position.

    Args:
        x: X coordinate.
        y: Y coordinate.
        polygon: Sector polygon.
        radius: Thing collision radius.

    Returns:
        Tuple of (is_valid, message).
    """
    point = Point(x, y)

    # Check if point is inside polygon
    if not polygon.contains(point):
        return False, "Position is outside sector"

    # Check clearance from walls
    buffered_point = point.buffer(radius)
    if not polygon.contains(buffered_point):
        return False, "Too close to walls"

    return True, "Valid"


def validate_wad(filepath: str | Path) -> tuple[bool, list[str]]:
    """
    Validate a WAD file structure.

    Note: This is a basic validation. For full validation,
    use a proper node builder or map editor.

    Args:
        filepath: Path to WAD file.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    filepath = Path(filepath)
    issues = []

    if not filepath.exists():
        return False, ["File does not exist"]

    # Read WAD header
    with open(filepath, 'rb') as f:
        header = f.read(12)

        if len(header) < 12:
            return False, ["File too small for WAD header"]

        # Check signature
        signature = header[:4].decode('ascii', errors='ignore')
        if signature not in ('IWAD', 'PWAD'):
            issues.append(f"Invalid WAD signature: {signature}")

        # Read lump count and directory offset
        import struct
        num_lumps = struct.unpack('<I', header[4:8])[0]
        dir_offset = struct.unpack('<I', header[8:12])[0]

        # Basic sanity checks
        file_size = filepath.stat().st_size

        if dir_offset > file_size:
            issues.append(f"Directory offset beyond file size: {dir_offset} > {file_size}")

        expected_dir_size = num_lumps * 16
        if dir_offset + expected_dir_size > file_size:
            issues.append("Directory extends beyond file size")

        # Check for map markers
        f.seek(dir_offset)
        has_map = False

        for _ in range(num_lumps):
            entry = f.read(16)
            if len(entry) < 16:
                break

            name = entry[8:16].rstrip(b'\x00').decode('ascii', errors='ignore')

            # Check for map markers (MAP01, E1M1, etc.)
            if name.startswith('MAP') or (len(name) == 4 and name[0] == 'E' and name[2] == 'M'):
                has_map = True
                break

        if not has_map:
            issues.append("No map markers found")

    return len(issues) == 0, issues


def check_sector_closure(vertices: list[tuple[int, int]]) -> tuple[bool, str]:
    """
    Check if vertices form a closed sector.

    Args:
        vertices: List of (x, y) vertex coordinates.

    Returns:
        Tuple of (is_closed, message).
    """
    if len(vertices) < 3:
        return False, "Too few vertices for a sector"

    # Check if first and last vertices match (or are close)
    first = vertices[0]
    last = vertices[-1]

    if first != last:
        # Check if they're within tolerance
        dx = abs(first[0] - last[0])
        dy = abs(first[1] - last[1])
        if dx > 1 or dy > 1:
            return False, f"Sector not closed: gap from {last} to {first}"

    return True, "Sector is closed"


def estimate_bsp_complexity(polygon: Polygon) -> dict:
    """
    Estimate BSP tree complexity for a polygon.

    This gives a rough idea of how complex the resulting
    BSP tree might be.

    Args:
        polygon: The sector polygon.

    Returns:
        Dictionary with complexity metrics.
    """
    vertices = len(polygon.exterior.coords) - 1

    # Count holes
    holes = len(polygon.interiors)

    # Estimate based on vertex count and shape complexity
    # (This is a very rough heuristic)
    convex_hull = polygon.convex_hull
    convexity_ratio = polygon.area / convex_hull.area if convex_hull.area > 0 else 1.0

    return {
        'vertex_count': vertices,
        'hole_count': holes,
        'convexity_ratio': convexity_ratio,
        'estimated_segs': vertices * 2,  # Rough estimate
        'estimated_ssectors': max(1, vertices // 3),
        'complexity_score': vertices * (1 + holes) / convexity_ratio,
    }
