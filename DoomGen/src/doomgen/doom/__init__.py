# Doom module
"""
Doom-specific structures and WAD file manipulation.

This module contains everything Doom-specific:
- WAD file I/O
- Sector/linedef translation
- Thing types and linedef specials
"""

# Configuration
from doomgen.doom.config import (
    SectorConfig,
    WallConfig,
    PrefabDoomConfig,
)

# WAD I/O
from doomgen.doom.wad import WADWriter, run_zdbsp

# Map Translation
from doomgen.doom.translator import MapTranslator

# Things
from doomgen.doom.things import ThingType

# Tags
from doomgen.doom.tags import TagRegistry

# Specials
from doomgen.doom.specials import (
    LinedefFlags,
    DoorAction,
    LiftAction,
    TeleportAction,
    FloorAction,
    CeilingAction,
    CrusherAction,
    StairAction,
    ExitAction,
    LightAction,
    SectorSpecial,
    PolyobjAction,
)

from doomgen.doom.polyobjects import (
    PolyobjectBuilder,
    PolyobjectConfig,
)

__all__ = [
    # Config
    "SectorConfig",
    "WallConfig",
    "PrefabDoomConfig",

    # WAD
    "WADWriter",
    "run_zdbsp",

    # Translation
    "MapTranslator",

    # Things
    "ThingType",

    # Tags
    "TagRegistry",

    # Specials
    "LinedefFlags",
    "DoorAction",
    "LiftAction",
    "TeleportAction",
    "FloorAction",
    "CeilingAction",
    "CrusherAction",
    "StairAction",
    "ExitAction",
    "LightAction",
    "SectorSpecial",
    "PolyobjAction",
    "PolyobjectBuilder",
    "PolyobjectConfig",
]
