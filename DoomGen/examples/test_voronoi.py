from doomgen.builder import ProceduralMapBuilder
from doomgen.abstraction.layout import ConnectionType
from shapely.geometry import Point, box
import os

def create_voronoi_map():
    print("Generating Voronoi Map...")

    # Initialize Builder with larger bounds to cover all rooms comfortably
    builder = ProceduralMapBuilder(bounds=(-600, -600, 600, 600), num_seeds=2000, seed=42)

    # Main Hall (Rectangle)
    # Positioned at (0, 0)
    builder.add_area(
        name="Main Hall",
        shape=(0, 0, 384, 384),
        floor_height=0,
        ceiling_height=128
    )

    # Cave (Circle-ish via polygon)
    # Positioned close to Main Hall so they touch.
    # Main Hall is x: -192 to 192.
    # Let's put Cave center at x=300. Radius 128 means it goes from 172 to 428.
    # Overlap: 172 < 192. So they overlap by 20 units.

    circle_poly = Point(300, 0).buffer(128)

    builder.add_area(
        name="Cave",
        shape=circle_poly,
        floor_height=-16,
        floor_texture="NUKAGE1"
    )

    # Connect them directly without a door/boundary area
    # We just want them to be logically connected so the translator makes the shared wall passable.
    builder.connect_adjacent("Main Hall", "Cave", ConnectionType.OPEN)

    # Add Pillar in Main Hall using generic add_area with overwrite mode
    # A pillar is just a solid area (VOID) or a raised area inside another area.
    # Let's make a decorative pillar (raised floor)
    pillar_shape = Point(0, 0).buffer(48)
    builder.add_area(
        name="Center Pillar",
        shape=pillar_shape,
        floor_height=32,
        wall_texture="SUPPORT3",
        mode="overwrite"
    )

    # Add Void Pillar (Hole) in Cave using set_void
    hole_shape = Point(300, 0).buffer(32)
    builder.set_void(hole_shape)

    # Add Player Start
    builder.add_thing("PLAYER1_START", -100, 0, angle=0)

    # Build
    os.makedirs("examples/output", exist_ok=True)
    builder.build("examples/output/voronoi_test.wad")

    # Visualize
    from doomgen.utils.visualization import visualize_mesh
    visualize_mesh(builder.mesh, "examples/output/voronoi_test.png")
    print("Visualization saved to examples/output/voronoi_test.png")

if __name__ == "__main__":
    create_voronoi_map()
