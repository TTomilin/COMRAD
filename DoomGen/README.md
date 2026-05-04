# DoomGen

A Python library for procedurally generating Doom WAD files, specifically designed for creating benchmarks for **Multi-Agent Reinforcement Learning (MARL)**.

The library uses a **Voronoi-First** approach: the map is generated as a mesh of relaxed Voronoi cells (Lloyd's algorithm), which are then grouped into logical areas (rooms, corridors, doors). This ensures organic, non-grid-aligned geometry and robust topology suitable for navigation tasks.

## Features

- **Voronoi-Based Geometry**: Organic, mesh-based map generation using Lloyd's relaxation.
- **Logical Layouts**: Define areas (rooms) and connections (doors, open passages) abstractly.
- **Pathfinding Generation**: Automatically generate corridors and stairs between disconnected areas using A* on the cell mesh.
- **ACS Scripting**: Inject game logic (puzzles, rewards, state tracking) via a Pythonic ACS builder.
- **Deterministic**: Fully seedable for reproducible RL benchmarks.
- **Doom 2 Compatible**: Generates standard UDMF or Hexen format WADs (via `omgifol`).

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/mitkozh/doomgen.git
cd doomgen
pip install -e ".[dev]"
```

## Quick Start

```python
from doomgen.builder import ProceduralMapBuilder
from doomgen.abstraction.layout import ConnectionType

# 1. Initialize Builder with bounds and seed
builder = ProceduralMapBuilder(bounds=(-1024, -1024, 1024, 1024), num_seeds=2000, seed=42)

# 2. Define Areas (Claim Voronoi Cells)
# "claim_void" (default): Only claims unassigned cells.
builder.add_area("Main Hall", shape=(0, 0, 512, 512), floor_texture="FLOOR4_8")
builder.add_area("Armory", shape=(600, 0, 256, 256), floor_texture="CEIL5_1")

# 3. Connect Areas
# Create a door between them (automatically finds shared boundary cells)
builder.add_boundary("Main Hall", "Armory", name="Door1", connection_type=ConnectionType.DOOR)

# 4. Add Things
builder.add_thing("PLAYER1_START", 0, 0, angle=0)
builder.add_thing("Shotgun", 600, 0, angle=0)

# 5. Build WAD
builder.build("output/my_map.wad")
```

## Architecture

- **`ProceduralMapBuilder`**: High-level facade for map generation.
- **`VoronoiMesh`**: Manages the underlying graph of Voronoi cells and their adjacency.
- **`ACSBuilder`**: Generates Action Code Script (ACS) bytecode for map logic.
- **`MapTranslator`**: Converts the abstract Voronoi mesh into Doom sectors and linedefs.

## Requirements

- Python 3.10+
- `scipy`, `numpy`, `shapely` (for geometry)
- `omgifol` (for WAD I/O)
- `acc` (ACS Compiler) - usually included with GZDoom or available separately.

## Benchmark Scenarios

Run the new chained-platform benchmark:

```bash
python examples/benchmark/platform_chain.py
```

This generates:

- `examples/benchmark/output/platform_chain.wad`

Batch-generate varied platform-chain configurations:

```bash
python examples/benchmark/batches/generate_platform_chain_batch.py
```

Batch output directory:

- `examples/benchmark/output/batch_platform_chain/`

## License

MIT License
