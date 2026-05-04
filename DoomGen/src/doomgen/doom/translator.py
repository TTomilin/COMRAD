"""
MapTranslator - Converts VoronoiMesh to Doom Map Data.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from shapely.geometry import LineString

from doomgen.doom.wad import DoomMapData
from doomgen.geometry.mesh import VoronoiMesh
from doomgen.abstraction.layout import LayoutGraph, Area, ConnectionType
from doomgen.doom.specials import LinedefFlags

class MapTranslator:
    """
    Translates the topological VoronoiMesh into Doom sectors and linedefs.
    """
    def __init__(self, map_data: DoomMapData, use_hexen_format: bool = False):
        self.map_data = map_data
        self.use_hexen_format = use_hexen_format
        self.area_to_sector_idx: Dict[str, int] = {} # Maps Area ID to Doom Sector Index

    def translate(self, mesh: VoronoiMesh, graph: LayoutGraph):
        """
        Main translation process.
        1. Create Sectors for each Area.
        2. Create Linedefs for each Ridge.
        """

        # 1. Create Sectors
        # In this Voronoi-first approach, one Area = One Sector (usually).
        # Unless we want to split it for lighting, but let's keep it simple.
        for area_id, area in graph.areas.items():
            sector_idx = self.map_data.add_sector(
                floor_height=area.config.floor_height,
                ceiling_height=area.config.ceiling_height,
                floor_texture=area.config.floor_texture,
                ceiling_texture=area.config.ceiling_texture,
                light_level=area.config.light_level,
                special=area.config.special,
                tag=area.config.tag,
                **area.config.properties
            )
            self.area_to_sector_idx[area_id] = sector_idx

        # 2. Process Ridges (Linedefs)
        ridges = mesh.get_ridges_for_translation()

        for ridge in ridges:
            line_geom: LineString = ridge["line"]
            front_area = ridge["front_area"]
            back_area = ridge["back_area"]
            rtype = ridge["type"]

            # Get coordinates (rounded to int before snapping)
            # This should be round instead of floor
            x1, y1 = int(round(line_geom.coords[0][0])), int(round(line_geom.coords[0][1]))
            x2, y2 = int(round(line_geom.coords[1][0])), int(round(line_geom.coords[1][1]))

            # Skip zero-length lines
            if x1 == x2 and y1 == y2: continue

            # Get Sector Indices
            sec_front = self.area_to_sector_idx.get(front_area)
            sec_back = self.area_to_sector_idx.get(back_area)

            # Create Vertices
            v1_idx = self.map_data.add_vertex(x1, y1)
            v2_idx = self.map_data.add_vertex(x2, y2)

            # Get Configs (handle VOID)
            config_front = graph.get_area(front_area).config if front_area != "VOID" else None
            config_back = graph.get_area(back_area).config if back_area != "VOID" else None

            side_front = -1
            side_back = -1

            if rtype == "void_wall":
                # One-sided wall facing the valid sector (Front)
                if sec_front is not None:
                    side_front = self.map_data.add_sidedef(
                        sector=sec_front,
                        middle_texture=config_front.wall_texture,
                        **config_front.properties
                    )
                    # No back side

            elif rtype == "solid":
                # Two-sided blocking wall
                if sec_front is not None:
                    side_front = self.map_data.add_sidedef(
                        sector=sec_front,
                        middle_texture=config_front.wall_texture,
                        **config_front.properties
                    )
                    self.map_data.add_linedef(
                        start_vertex=v1_idx,
                        end_vertex=v2_idx,
                        front_sidedef=side_front,
                        back_sidedef=-1,
                        flags=LinedefFlags.BLOCKING
                    )

                if sec_back is not None:
                    side_back = self.map_data.add_sidedef(
                        sector=sec_back,
                        middle_texture=config_back.wall_texture,
                        **config_back.properties
                    )
                    self.map_data.add_linedef(
                        start_vertex=v2_idx, # Reversed for back side
                        end_vertex=v1_idx,
                        front_sidedef=side_back,
                        back_sidedef=-1,
                        flags=LinedefFlags.BLOCKING
                    )

                continue # Skip the default add_linedef at the end

            elif rtype == "portal":
                # Two-sided portal (passable)
                # Check for height differences to set lower/upper textures

                # Get sector heights
                # Note: We need to access the actual sector data we created earlier
                # But map_data.sectors is a list of dicts.
                s_front = self.map_data.sectors[sec_front]
                s_back = self.map_data.sectors[sec_back]

                f_front = s_front['floor_height']
                c_front = s_front['ceiling_height']
                f_back = s_back['floor_height']
                c_back = s_back['ceiling_height']

                # Determine textures
                # Logic:
                # Lower Texture: Needed on the side with the LOWER floor (to fill the step up).
                # Upper Texture: Needed on the side with the HIGHER ceiling (to fill the step down).

                # Logic for Lower/Upper Textures of side_front (Sector Front)
                # side_front faces Sector Front.
                # If Sector Back is Higher, side_front exposes the Wall of Sector Back.

                # Default assignments (Always populate to support dynamic movement)
                if config_back:
                    lower_front = config_back.wall_texture
                    if config_back.lower_texture: lower_front = config_back.lower_texture
                else:
                    lower_front = "-"

                # Upper Front: Reflects Front Sector
                lower_back = "-"

                upper_front = config_front.wall_texture
                if config_front.upper_texture: upper_front = config_front.upper_texture

                # Upper Back: Reflects Back Sector
                if config_back:
                    upper_back = config_back.wall_texture
                    if config_back.upper_texture: upper_back = config_back.upper_texture
                else:
                    upper_back = "-"

                # Lower Back: Reflects Front Sector (Seen from Back if Back is Low)
                lower_back = config_front.wall_texture
                if config_front.lower_texture: lower_back = config_front.lower_texture

                # Optimization: Optional cleanup if no height diffs (but we disable for dynamic support)
                # if f_front >= f_back: lower_front = "-"
                # if f_back >= f_front: lower_back = "-"
                # if c_front <= c_back: upper_front = "-"
                # if c_back <= c_front: upper_back = "-"

                side_front = self.map_data.add_sidedef(
                    sector=sec_front,
                    middle_texture=config_front.middle_texture if config_front.middle_texture else "-",
                    lower_texture=lower_front,
                    upper_texture=upper_front,
                    **config_front.properties
                )

                side_back = self.map_data.add_sidedef(
                    sector=sec_back,
                    middle_texture=config_back.middle_texture if config_back.middle_texture else "-",
                    lower_texture=lower_back,
                    upper_texture=upper_back,
                    **config_back.properties
                )

            # Create Linedef
            flags = 0
            special = 0
            tag = 0

            connection = ridge.get("connection")

            # Removed check rtype == solid, solid block upper uses continue so this is unreachable anw
            if rtype == "portal":
                flags |= LinedefFlags.TWO_SIDED

                # Check for Window/Fence
                mid_tex = "-"
                if connection and connection.type == ConnectionType.WINDOW:
                    mid_tex = connection.properties.get("middle_texture", "MIDGRATE")
                    # Windows usually block movement but allow sight/bullets (unless specified)
                    # If it has a texture, we might want to block movement.
                    if connection.properties.get("blocking", True):
                        flags |= LinedefFlags.BLOCKING

                # Apply explicit linedef flags from connection properties if present
                if connection and "linedef_flags" in connection.properties:
                    flags |= connection.properties["linedef_flags"]

                # Handle Door Connection
                if connection and connection.type == ConnectionType.DOOR:
                    # Identify which side is the door (usually closed or specific texture)
                    # For manual doors (Type 1), the door sector must be on the BACK.

                    is_front_door = (s_front['ceiling_height'] == s_front['floor_height'])
                    is_back_door = (s_back['ceiling_height'] == s_back['floor_height'])

                    # Fix Texture: Use Door's texture for the interface
                    # If Front is Room and Back is Door, Front Upper needs Door Texture
                    if is_back_door and not is_front_door:
                        # Front side needs upper texture (Room -> Door)
                        # Currently it uses config_front (Room). Force it to use config_back (Door).
                        if side_front != -1:
                            # We need to update the sidedef we just created.
                            # Since we can't easily edit it in map_data (it's a list),
                            # we should have done this earlier.
                            # But we can access it by index.
                            sd = self.map_data.sidedefs[side_front]
                            if sd['upper_texture'] != "-":
                                sd['upper_texture'] = config_back.wall_texture
                            if sd['lower_texture'] != "-":
                                sd['lower_texture'] = config_back.wall_texture

                    elif is_front_door and not is_back_door:
                        # Back side needs upper texture (Room -> Door)
                        if side_back != -1:
                            sd = self.map_data.sidedefs[side_back]
                            if sd['upper_texture'] != "-":
                                sd['upper_texture'] = config_front.wall_texture
                            if sd['lower_texture'] != "-":
                                sd['lower_texture'] = config_front.wall_texture

                    if is_front_door and not is_back_door:
                        # Door is on Front. Flip linedef.
                        v1_idx, v2_idx = v2_idx, v1_idx
                        side_front, side_back = side_back, side_front
                        # Now door is on Back.
                        if self.use_hexen_format:
                            # Translator sets special=1 for door linedefs, but we use ACS scripts which force hexen
                            # In hexen action 1 = Polyobj_StartLine, so instead should use door_raise(12) or special=0 + ACS
                            # In armory siege door is controlled through ACS, so linedef special is pointless, so here set to 0 (no action)
                            # Also door linedefs are Polyobj_StartLine(0) during map initialization, which it cant without matching spawn spot or anchor
                            special = 0
                        else:
                            special = 1 # DR Door Open Wait Close
                            # REMOVED: flags |= LinedefFlags.BLOCKING
                            # Doors should be passable (Two Sided). The sector height blocks movement.

                    elif is_back_door and not is_front_door:
                        # Door is on Back. Good.
                        if self.use_hexen_format:
                            special = 0
                        else:
                            special = 1

                    # If both or neither, we can't decide easily.
                    # Maybe check textures?
                    elif "DOOR" in config_front.wall_texture or "DOOR" in config_back.wall_texture:
                         if self.use_hexen_format:
                             special = 0
                         else:
                             special = 1

                    # Check for custom linedef special in the Door Area config
                    # The door area is the one that is the "door" (is_front_door or is_back_door)
                    door_config = None
                    if is_front_door:
                        door_config = config_front
                    elif is_back_door:
                        door_config = config_back

                    if door_config and door_config.linedef_special is not None:
                        special = door_config.linedef_special
                        # We also need to handle args if we are in Hexen format
                        # But MapTranslator doesn't know about Hexen format directly here?
                        # Actually it does, it creates Linedefs.
                        # But the Linedef object in omgifol (ZLinedef) has arg0-4.
                        # The standard Linedef doesn't.
                        # We are storing args in a list.
                        # We need to pass them to add_linedef.

                # Update sidedefs with mid_tex if needed
                if mid_tex != "-":
                    if side_front != -1:
                        self.map_data.sidedefs[side_front]['middle_texture'] = mid_tex
                    if side_back != -1:
                        self.map_data.sidedefs[side_back]['middle_texture'] = mid_tex

            elif rtype == "void_wall":
                flags |= LinedefFlags.BLOCKING # Implicit for 1-sided, but good to be explicit if needed

            # Extract args if available
            args = []

            # Apply properties from connection (overrides defaults)
            if connection:
                if "special" in connection.properties:
                    special = connection.properties["special"]
                if "tag" in connection.properties:
                    tag = connection.properties["tag"]
                if "args" in connection.properties:
                    args = connection.properties["args"]
                if "flags" in connection.properties:
                    flags |= connection.properties["flags"]

            if special > 0 and not args:
                # Check if we have args from the door config (legacy support)
                door_config = None
                if connection and connection.type == ConnectionType.DOOR:
                    if is_front_door: door_config = config_front
                    elif is_back_door: door_config = config_back

                if door_config:
                    if door_config.linedef_args:
                        args = door_config.linedef_args
                    if door_config.linedef_flags is not None:
                        flags |= door_config.linedef_flags

            self.map_data.add_linedef(
                start_vertex=v1_idx,
                end_vertex=v2_idx,
                front_sidedef=side_front,
                back_sidedef=side_back,
                flags=flags,
                special=special,
                tag=tag,
                args=args
            )

    def add_player_start(self, area_id: str, x: float, y: float, angle: int = 0):
        """Add player start thing."""
        self.map_data.add_thing(
            x=int(x), y=int(y),
            thing_type=1, # Player 1 Start
            angle=angle,
            flags=7 # Skill 1-5
        )
