from doomgen.builder import ProceduralMapBuilder
from shapely.geometry import box, Point
import os

def create_organic_corridor_map():
    print("Generating Organic Corridor Map...")

    # Initialize Builder (creates global relaxed field)
    builder = ProceduralMapBuilder(bounds=(-512, -512, 512, 512), num_seeds=1000, seed=123)

    # Add Room A
    builder.add_area(
        name="Start Room",
        shape=(-300, 0, 128, 128), # x, y, w, h
        floor_texture="FLOOR4_8"
    )

    # Add Room B
    builder.add_area(
        name="End Room",
        shape=(300, 0, 128, 128),
        floor_texture="FLOOR4_8"
    )

    # Add Corridor (Pathfinding)
    corridor_id = builder.add_corridor(
        "Start Room", "End Room",
        width=2, # 2 cells wide
        floor_texture="RROCK19"
    )

    # Add Door at Start Room -> Corridor
    # Note: add_boundary finds boundary cells between two areas.
    # Since we just connected Start Room and Corridor, we can add a door between them.
    from doomgen.abstraction.layout import ConnectionType
    builder.add_boundary(
        "Start Room",
        corridor_id,
        connection_type=ConnectionType.DOOR,
        floor_height=0,
        ceiling_height=0 # Closed door
    )

    # Add Player Start
    builder.add_thing("PLAYER1_START", -300, 0, angle=0)

    # Build
    os.makedirs("examples/output", exist_ok=True)
    builder.build("examples/output/organic_corridor.wad")

    # Visualize
    from doomgen.utils.visualization import visualize_mesh
    visualize_mesh(builder.mesh, "examples/output/organic_corridor.png")
    print("Visualization saved to examples/output/organic_corridor.png")

if __name__ == "__main__":
    create_organic_corridor_map()
