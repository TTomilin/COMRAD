"""
DoomGen - Voronoi-first procedural Doom map generation library.
"""

__version__ = "0.3.0"

# Abstract Layer
from doomgen.abstraction.layout import (
    LayoutGraph,
    Area,
    AreaConfig,
    Transform,
    ConnectionType
)

# Geometry Layer
from doomgen.geometry.seeds import generate_relaxed_seeds
from doomgen.geometry.mesh import VoronoiMesh

# Doom Layer
from doomgen.doom.wad import WADWriter, DoomMapData, run_zdbsp
from doomgen.doom.translator import MapTranslator
from doomgen.doom.things import ThingType

# Builder Layer
from doomgen.builder import ProceduralMapBuilder as MapBuilder

__all__ = [
    "LayoutGraph", "Area", "AreaConfig", "Transform", "ConnectionType",
    "generate_relaxed_seeds",
    "VoronoiMesh",
    "WADWriter", "DoomMapData", "run_zdbsp", "MapTranslator", "ThingType",
    "MapBuilder"
]
