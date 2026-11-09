from scenario_generation.paths import RUNTIME_SCENARIOS
import random
from typing import Any, Dict
import io
from PIL import Image

from doomgen.abstraction.layout import ConnectionType
from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.things import ThingType
from doomgen.logic.acs_builder import ACSBuilder


class TextMazeScenario(Scenario):
    """
    Two-player cooperative maze with text-based communication.

    Player 1 (Navigator) moves through a procedural lava maze.
    Player 2 (Guide) observes from stands and broadcasts direction
    tokens via the notifications buffer to guide navigation.

    Curriculum: 2 direction tokens -> 3 -> 4 as levels progress.
    """

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "seed": 42,
            "physical_size": 9,
            "cell_size": 128,
            "initial_maze_size": 4,
            "max_maze_size": 20,
            "lava_damage": 35,
            "lava_depth": 64,
            "step_tics": 12,
            "step_cooldown_tics": 4,
            "stand_thickness": 50,
            "stand_gap": 500,
            "dir_2_levels": 3,
            "dir_3_levels": 3,
        }

    def generate(self, output_path: str) -> None:
        cfg = self.config
        PHYSICAL_SIZE = int(cfg["physical_size"])
        CELL_SIZE = int(cfg["cell_size"])
        INITIAL_MAZE_SIZE = int(cfg.get("initial_maze_size", 4))
        MAX_MAZE_SIZE = int(cfg.get("max_maze_size", 14))
        LAVA_DAMAGE = int(cfg["lava_damage"])
        LAVA_DEPTH = int(cfg.get("lava_depth", 64))
        STEP_TICS = int(cfg.get("step_tics", 12))
        STEP_COOLDOWN_TICS = int(cfg.get("step_cooldown_tics", 4))
        STANDS_THICKNESS = int(cfg.get("stand_thickness", 200))
        STANDS_GAP = int(cfg.get("stand_gap", 64))
        DIR_2_LEVELS = int(cfg.get("dir_2_levels", 3))
        DIR_3_LEVELS = int(cfg.get("dir_3_levels", 3))
        seed = cfg["seed"] if cfg["seed"] is not None else random.randint(0, 999999)

        MAZE_WIDTH_UNITS = PHYSICAL_SIZE * CELL_SIZE
        OFFSET_X = -(MAZE_WIDTH_UNITS // 2)
        OFFSET_Y = -(MAZE_WIDTH_UNITS // 2)

        builder = ProceduralMapBuilder(
            bounds=(-2400, -2400, 2400, 2400),
            num_seeds=24000,
            seed=seed,
        )

        # Generate texture lumps for stand walls, maze walls, and markers
        def create_color_lump(name: str, color_rgb: tuple[int, int, int]):
            img = Image.new('RGB', (1920, 1080), color=color_rgb)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            builder.wad_writer.add_lump(name, buf.getvalue())

        lump_files = {
            "S_GREEN": (0, 255, 0),
            "S_RED":   (255, 0, 0),
            "S_YELLOW":(255, 255, 0),
            "S_BLUE":  (0, 0, 255),
            "S_BLACK": (0, 0, 0),
            "S_WALL":  (200, 200, 200),
            "F_START": (0, 255, 255),
            "F_END":   (255, 0, 255),
        }
        builder.wad_writer.add_lump("TX_START", b"")
        for lump_name, color in lump_files.items():
            create_color_lump(lump_name, color)
        builder.wad_writer.add_lump("TX_END", b"")

        # --- Lava pit ---
        builder.add_area(
            "LavaPit",
            shape=(0, 0, 4400, 4400),
            floor_height=-LAVA_DEPTH,
            ceiling_height=800,
            floor_texture="LAVA1",
            mode="overwrite",
            light_level=255,
            tag=99,
        )

        # --- Perimeter stands ---
        STANDS_HEIGHT = 350
        STANDS_INNER = (MAZE_WIDTH_UNITS // 2) + STANDS_GAP

        pos_y = STANDS_INNER + STANDS_THICKNESS / 2
        stands_width = 2 * STANDS_INNER + 2 * STANDS_THICKNESS
        builder.add_area("Stands_N", shape=(0, pos_y, stands_width, STANDS_THICKNESS), floor_height=STANDS_HEIGHT, ceiling_height=1024, mode="overwrite", tag=201, wall_texture="S_YELLOW", light_level=255)
        builder.add_area("Stands_S", shape=(0, -pos_y, stands_width, STANDS_THICKNESS), floor_height=STANDS_HEIGHT, ceiling_height=1024, mode="overwrite", tag=203, wall_texture="S_BLUE", light_level=255)

        pos_x = STANDS_INNER + STANDS_THICKNESS / 2
        h_span = 2 * STANDS_INNER + 2 * STANDS_THICKNESS
        builder.add_area("Stands_E", shape=(pos_x, 0, STANDS_THICKNESS, h_span), floor_height=STANDS_HEIGHT, ceiling_height=1024, mode="overwrite", tag=200, wall_texture="S_GREEN", light_level=255)
        builder.add_area("Stands_W", shape=(-pos_x, 0, STANDS_THICKNESS, h_span), floor_height=STANDS_HEIGHT, ceiling_height=1024, mode="overwrite", tag=202, wall_texture="S_RED", light_level=255)

        # --- Grid cells ---
        base_tag = 1000
        center_idx = PHYSICAL_SIZE // 2
        for py in range(PHYSICAL_SIZE):
            for px in range(PHYSICAL_SIZE):
                cx = OFFSET_X + (px * CELL_SIZE) + (CELL_SIZE // 2)
                cy = OFFSET_Y + (py * CELL_SIZE) + (CELL_SIZE // 2)
                tag = base_tag + (py * PHYSICAL_SIZE + px)
                if px == center_idx and py == center_idx:
                    f_tex, f_h = "FLOOR4_8", 0
                else:
                    f_tex, f_h = "LAVA1", -LAVA_DEPTH
                builder.add_area(
                    f"Grid_{px}_{py}",
                    shape=(cx, cy, CELL_SIZE, CELL_SIZE),
                    floor_height=f_h,
                    ceiling_height=1024,
                    tag=tag,
                    mode="overwrite",
                    floor_texture=f_tex,
                    wall_texture="S_WALL",
                    light_level=255,
                )

        # Stands connectivity
        builder.connect_adjacent("Stands_N", "Stands_E", ConnectionType.OPEN)
        builder.connect_adjacent("Stands_E", "Stands_S", ConnectionType.OPEN)
        builder.connect_adjacent("Stands_S", "Stands_W", ConnectionType.OPEN)
        builder.connect_adjacent("Stands_W", "Stands_N", ConnectionType.OPEN)

        barrier_args = {"blocking": True, "middle_texture": "-"}
        for stand in ("Stands_N", "Stands_S", "Stands_E", "Stands_W"):
            builder.connect_adjacent(stand, "LavaPit", ConnectionType.WINDOW, **barrier_args)

        # All grid cells connected (ACS carve constrains to active directions)
        for py in range(PHYSICAL_SIZE):
            for px in range(PHYSICAL_SIZE):
                curr = f"Grid_{px}_{py}"
                if px < PHYSICAL_SIZE - 1:
                    builder.connect_adjacent(curr, f"Grid_{px+1}_{py}", ConnectionType.OPEN, silent=True)
                if py < PHYSICAL_SIZE - 1:
                    builder.connect_adjacent(curr, f"Grid_{px}_{py+1}", ConnectionType.OPEN, silent=True)
                if px < PHYSICAL_SIZE - 1 and py < PHYSICAL_SIZE - 1:
                    builder.connect_adjacent(curr, f"Grid_{px+1}_{py+1}", ConnectionType.OPEN, silent=True)
                if px > 0 and py < PHYSICAL_SIZE - 1:
                    builder.connect_adjacent(curr, f"Grid_{px-1}_{py+1}", ConnectionType.OPEN, silent=True)
                builder.connect_adjacent(curr, "LavaPit", ConnectionType.OPEN, silent=True)

        # Player starts
        p1_x = OFFSET_X + (center_idx * CELL_SIZE) + (CELL_SIZE // 2)
        p1_y = OFFSET_Y + (center_idx * CELL_SIZE) + (CELL_SIZE // 2)
        builder.add_thing(ThingType.PLAYER1_START, x=p1_x, y=p1_y, angle=0)

        p2_y = STANDS_INNER + (STANDS_THICKNESS // 2)  # NORTH stand
        builder.add_thing(ThingType.PLAYER2_START, x=0, y=p2_y, angle=270)

        acs = ACSBuilder()
        acs.add_include("zcommon.acs")

        acs.add_map_var("level_clear_lock", initial=1)

        acs.add_global_var("tm_p1_grid_x",      1, "int")
        acs.add_global_var("tm_p1_grid_y",      2, "int")
        acs.add_global_var("tm_goal_grid_x",    3, "int")
        acs.add_global_var("tm_goal_grid_y",    4, "int")
        acs.add_global_var("tm_maze_size",      5, "int")
        acs.add_global_var("tm_level",          6, "int")
        acs.add_global_var("tm_direction_count", 7, "int")
        acs.add_global_var("tm_can_see",        8, "int")

        acs.scripts = []

        STANDS_RADIUS_UNITS = int(STANDS_INNER + STANDS_THICKNESS / 2)

        maze_logic_script = f"""
    #define MAP_WIDTH {PHYSICAL_SIZE}
    #define CELL_SIZE {CELL_SIZE}
    #define BASE_TAG {base_tag}
    #define MAX_TILES {MAX_MAZE_SIZE}
    #define PLATFORM_RADIUS {STANDS_RADIUS_UNITS}
    #define TM_STEP_TICS {STEP_TICS}
    #define TM_STEP_COOLDOWN_TICS {STEP_COOLDOWN_TICS}
    #define DIR_2_LEVELS {DIR_2_LEVELS}
    #define DIR_3_LEVELS {DIR_3_LEVELS}

    int max_unlocked_size = {INITIAL_MAZE_SIZE};
    int current_maze_size = {INITIAL_MAZE_SIZE};
    int current_start_tag = 0;
    int current_end_tag = 0;

    int g_start_x = MAP_WIDTH / 2;
    int g_start_y = MAP_WIDTH / 2;

    // Bit-packed safe cells for the current maze
    int maze_bits_chunk[3];

    // Current direction vocabulary state
    int direction_bits = 3;  // bit 0=EAST, 1=WEST, 2=SOUTH, 3=NORTH
    str dir_tokens[4];

    function int get_tag(int x, int y) {{
        return BASE_TAG + (y * MAP_WIDTH + x);
    }}

    function int get_world_x(int gx) {{
        return {OFFSET_X} + (gx * {CELL_SIZE}) + ({CELL_SIZE} / 2);
    }}

    function int get_world_y(int gy) {{
        return {OFFSET_Y} + (gy * {CELL_SIZE}) + ({CELL_SIZE} / 2);
    }}

    function int cell_is_safe(int x, int y) {{
        if (x < 0 || x >= MAP_WIDTH || y < 0 || y >= MAP_WIDTH) return FALSE;
        int bit_idx = y * MAP_WIDTH + x;
        int chunk = bit_idx / 27;
        int shift = bit_idx % 27;
        int mask = 1 << shift;
        if (chunk == 0) return ((maze_bits_chunk[0] & mask) != 0);
        if (chunk == 1) return ((maze_bits_chunk[1] & mask) != 0);
        if (chunk == 2) return ((maze_bits_chunk[2] & mask) != 0);
        return FALSE;
    }}

    function void reset_map(int keep_tag) {{
        int t;
        int i;
        for (i = 0; i < MAP_WIDTH * MAP_WIDTH; i++) {{
            t = BASE_TAG + i;
            if (t != keep_tag) {{
                Floor_MoveToValue(t, 2048, -{LAVA_DEPTH}, 0);
                ChangeFloor(t, "LAVA1");
                Sector_SetDamage(t, {LAVA_DAMAGE}, 14);
                Sector_SetColor(t, 255, 255, 255);
            }}
        }}
    }}

    function void carve(int x, int y) {{
        int tag = get_tag(x, y);
        Floor_MoveToValue(tag, 2048, 0, 0);
        ChangeFloor(tag, "FLOOR4_8");
        Sector_SetDamage(tag, 0, 0);
        Light_ChangeToValue(tag, 255);  // normal light
        if (x >= 0 && x < MAP_WIDTH && y >= 0 && y < MAP_WIDTH) {{
            int bit_idx = y * MAP_WIDTH + x;
            int chunk = bit_idx / 27;
            int shift = bit_idx % 27;
            if (chunk < 3) maze_bits_chunk[chunk] = maze_bits_chunk[chunk] | (1 << shift);
        }}
    }}

    function void init_direction_tokens(void) {{
        // Called after direction_bits is set
        dir_tokens[0] = "EAST";
        dir_tokens[1] = "WEST";
        dir_tokens[2] = "SOUTH";
        dir_tokens[3] = "NORTH";
    }}

    function int count_directions(void) {{
        int count = 0;
        int i;
        for (i = 0; i < 4; i++) {{
            if (direction_bits & (1 << i)) count++;
        }}
        return count;
    }}

    // Script placeholders -- filled in Tasks 2-4
    script 1 OPEN {{
        Sector_SetDamage(99, {LAVA_DAMAGE}, 14);

        g_start_x = MAP_WIDTH / 2;
        g_start_y = MAP_WIDTH / 2;

        current_start_tag = get_tag(g_start_x, g_start_y);
        tm_p1_grid_x         = -1;
        tm_p1_grid_y         = -1;
        tm_goal_grid_x       = -1;
        tm_goal_grid_y       = -1;
        tm_maze_size         = MAP_WIDTH;
        tm_level             = 0;
        tm_direction_count   = 2;
        tm_can_see           = 0;
        direction_bits       = 3;    // bits 0+1 = EAST,WEST
        level_clear_lock     = 1;

        init_direction_tokens();

        // Belt-and-suspenders: ensure start cell has no damage before GenerateMaze
        Sector_SetDamage(current_start_tag, 0, 0);

        ACS_NamedExecuteAlways("GenerateMaze", 0, g_start_x, g_start_y, current_maze_size);

        while (TRUE) {{
            tm_can_see = CheckSight(1001, 1000, 0);
            Delay(5);
        }}
    }}
    script "GenerateMaze" (int sx, int sy, int size) {{
        int start_tag = get_tag(sx, sy);
        int i;
        reset_map(start_tag);
        Delay(1);

        for (i = 0; i < 3; i++) maze_bits_chunk[i] = 0;

        carve(sx, sy);
        ChangeFloor(start_tag, "F_START");
        Sector_SetColor(start_tag, 255, 255, 255);
        Light_ChangeToValue(start_tag, 255);
        Sector_SetFade(start_tag, 0, 0, 0);

        // Determine which directions are active based on tm_level
        int dir_count = 2;
        direction_bits = 3;  // EAST | WEST
        if (tm_level >= DIR_2_LEVELS + DIR_3_LEVELS) {{
            // Stage 3: full 4 directions
            direction_bits = 15;  // EAST | WEST | SOUTH | NORTH
            dir_count = 4;
        }} else if (tm_level >= DIR_2_LEVELS) {{
            // Stage 2: 3 directions -- randomly exclude one axis
            int exclude_axis = Random(0, 1);
            if (exclude_axis == 0) {{
                direction_bits = 11;  // exclude SOUTH (bit 2)
            }} else {{
                direction_bits = 7;   // exclude NORTH (bit 3)
            }}
            dir_count = 3;
        }}
        tm_direction_count = dir_count;

        // Stage 0 (2 directions): use simple 1D line generation
        if (dir_count == 2) {{
            int dir = 1;
            int can_pos = (sx + size - 1 < MAP_WIDTH);
            int can_neg = (sx - size + 1 >= 0);
            if (can_pos && can_neg) {{
                dir = (Random(0, 1) * 2) - 1;
            }} else if (can_pos) {{
                dir = 1;
            }} else {{
                dir = -1;
            }}

            int current_x = sx;
            for (i = 1; i < size; i++) {{
                current_x = current_x + dir;
                carve(current_x, sy);
            }}
            current_end_tag = get_tag(current_x, sy);
            tm_goal_grid_x = current_x;
            tm_goal_grid_y = sy;
        }} else {{
            // Stage 1+ (3 or 4 directions): 2D DFS carve
            int visited[MAP_WIDTH * MAP_WIDTH];
            int stack[100];
            for (i = 0; i < MAP_WIDTH * MAP_WIDTH; i++) visited[i] = 0;

            int stack_ptr = 0;
            int cx = sx;
            int cy = sy;
            int count_carved = 1;

            visited[cy * MAP_WIDTH + cx] = 1;
            stack[stack_ptr++] = (cx << 8) | cy;

            int neighbors[4];
            int n_count;
            int nx, ny;
            int last_x = sx;
            int last_y = sy;

            while (count_carved < size && stack_ptr > 0) {{
                int packed = stack[stack_ptr - 1];
                cx = packed >> 8;
                cy = packed & 0xFF;

                n_count = 0;

                // NORTH (cy+2, bit 3)
                if ((direction_bits & 8) && (cy + 2 < MAP_WIDTH) && visited[(cy + 2) * MAP_WIDTH + cx] == 0)
                    neighbors[n_count++] = (cx << 8) | (cy + 2);
                // SOUTH (cy-2, bit 2)
                if ((direction_bits & 4) && (cy - 2 >= 0) && visited[(cy - 2) * MAP_WIDTH + cx] == 0)
                    neighbors[n_count++] = (cx << 8) | (cy - 2);
                // EAST (cx+2, bit 0)
                if ((direction_bits & 1) && (cx + 2 < MAP_WIDTH) && visited[cy * MAP_WIDTH + (cx + 2)] == 0)
                    neighbors[n_count++] = ((cx + 2) << 8) | cy;
                // WEST (cx-2, bit 1)
                if ((direction_bits & 2) && (cx - 2 >= 0) && visited[cy * MAP_WIDTH + (cx - 2)] == 0)
                    neighbors[n_count++] = ((cx - 2) << 8) | cy;

                if (n_count > 0) {{
                    int r = Random(0, n_count - 1);
                    int next_packed = neighbors[r];
                    nx = next_packed >> 8;
                    ny = next_packed & 0xFF;

                    carve((cx + nx) / 2, (cy + ny) / 2);
                    carve(nx, ny);

                    visited[ny * MAP_WIDTH + nx] = 1;
                    count_carved += 2;
                    last_x = nx;
                    last_y = ny;

                    stack[stack_ptr++] = next_packed;
                }} else {{
                    stack_ptr--;
                }}
            }}

            current_end_tag = get_tag(last_x, last_y);
            tm_goal_grid_x = last_x;
            tm_goal_grid_y = last_y;
        }}

        ChangeFloor(current_end_tag, "F_END");
        Sector_SetColor(current_end_tag, 255, 255, 255);
        Light_ChangeToValue(current_end_tag, 255);
        Sector_SetFade(current_end_tag, 0, 0, 0);

        level_clear_lock = 0;
    }}
    script 2 ENTER {{
        if (PlayerNumber() != 0) terminate;

        Thing_ChangeTID(0, 1000);
        SetActorProperty(0, APROP_Health, 100);
        SetPlayerProperty(0, 1, 0); // PROP_FROZEN

        int wx = get_world_x(g_start_x) << 16;
        int wy = get_world_y(g_start_y) << 16;

        SetActorPosition(0, wx, wy, 0, 0);
        // Force zero damage on whatever subsector the player landed in
        Sector_SetDamage(0, 0, 0);
        SetActorProperty(0, APROP_ViewHeight, 110.0);
        SetActorProperty(0, APROP_Mass, 0x7FFFFFFF);
        SetActorProperty(0, APROP_ScaleX, 5.5);
        SetActorProperty(0, APROP_ScaleY, 5.5);
        SetActorProperty(0, 21, 0);

        // Black HUD blind
        SetHudSize(320, 200, 1);
        SetFont("S_BLACK");
        HudMessage(s:"A"; HUDMSG_PLAIN, 1, CR_UNTRANSLATED, 160.0, 100.0, 0.0);
        SetFont("SMALLFONT");

        ClearInventory();

        int p1_cell_x = g_start_x;
        int p1_cell_y = g_start_y;
        int step_active = 0;
        int step_timer = 0;
        int step_from_x = g_start_x;
        int step_from_y = g_start_y;
        int step_to_x = g_start_x;
        int step_to_y = g_start_y;
        int step_cooldown = 0;
        int step_into_lava = 0;
        int prev_cell_x = g_start_x;
        int prev_cell_y = g_start_y;

        // Always 4-direction grid movement

        while (TRUE) {{
            if (GetActorProperty(0, APROP_HEALTH) <= 0) {{
                ACS_NamedExecute("EndGame", 0);
                terminate;
            }}

            if (step_active) {{
                step_timer++;
                int s_step_from_x = get_world_x(step_from_x);
                int s_step_to_x   = get_world_x(step_to_x);
                int s_step_from_y = get_world_y(step_from_y);
                int s_step_to_y   = get_world_y(step_to_y);
                int interp_x = s_step_from_x + (((s_step_to_x - s_step_from_x) * step_timer) / TM_STEP_TICS);
                int interp_y = s_step_from_y + (((s_step_to_y - s_step_from_y) * step_timer) / TM_STEP_TICS);
                SetActorPosition(0, interp_x << 16, interp_y << 16, 0, 0);
                SetActorVelocity(0, 0, 0, 0, FALSE, FALSE);

                if (step_timer >= TM_STEP_TICS) {{
                    p1_cell_x = step_to_x;
                    p1_cell_y = step_to_y;
                    SetActorPosition(0, get_world_x(p1_cell_x) << 16, get_world_y(p1_cell_y) << 16, 0, 0);
                    SetActorVelocity(0, 0, 0, 0, FALSE, FALSE);
                    step_active = 0;
                    step_timer = 0;
                    step_cooldown = TM_STEP_COOLDOWN_TICS;

                    // Print the direction of movement to notifications buffer
                    int dx = p1_cell_x - prev_cell_x;
                    int dy = p1_cell_y - prev_cell_y;
                    if (dx == 1)  PrintBold(s:"WEST");
                    if (dx == -1) PrintBold(s:"EAST");
                    if (dy == 1)  PrintBold(s:"SOUTH");
                    if (dy == -1) PrintBold(s:"NORTH");
                    prev_cell_x = p1_cell_x;
                    prev_cell_y = p1_cell_y;

                    if (step_into_lava) {{
                        SetActorProperty(0, APROP_Health, 0);
                    }}
                    step_into_lava = 0;
                }}
            }} else if (step_cooldown > 0) {{
                step_cooldown--;
            }} else if (level_clear_lock == 0) {{
                int btns = GetPlayerInput(-1, INPUT_BUTTONS);
                int target_x = p1_cell_x;
                int target_y = p1_cell_y;

                // WASD: W=south, S=north, A=east, D=west
                if ((btns & BT_FORWARD) && !(btns & BT_BACK)) {{
                    target_y = p1_cell_y - 1;
                }}
                if ((btns & BT_BACK) && !(btns & BT_FORWARD)) {{
                    target_y = p1_cell_y + 1;
                }}
                if ((btns & BT_MOVELEFT) && !(btns & BT_MOVERIGHT)) {{
                    target_x = p1_cell_x + 1;
                }}
                if ((btns & BT_MOVERIGHT) && !(btns & BT_MOVELEFT)) {{
                    target_x = p1_cell_x - 1;
                }}

                if (target_x != p1_cell_x || target_y != p1_cell_y) {{
                    step_into_lava = !cell_is_safe(target_x, target_y);
                    step_active = 1;
                    step_timer = 0;
                    step_from_x = p1_cell_x;
                    step_from_y = p1_cell_y;
                    step_to_x = target_x;
                    step_to_y = target_y;
                }}
            }}

            if (!step_active) {{
                SetActorPosition(0, get_world_x(p1_cell_x) << 16, get_world_y(p1_cell_y) << 16, 0, 0);
                SetActorVelocity(0, 0, 0, 0, FALSE, FALSE);
            }}

            if (p1_cell_x >= 0 && p1_cell_x < MAP_WIDTH && p1_cell_y >= 0 && p1_cell_y < MAP_WIDTH) {{
                tm_p1_grid_x = p1_cell_x;
                tm_p1_grid_y = p1_cell_y;
                int my_tag = get_tag(p1_cell_x, p1_cell_y);

                if (my_tag == current_end_tag && level_clear_lock == 0) {{
                    level_clear_lock = 1;
                    tm_level++;
                    max_unlocked_size += 2;
                    if (max_unlocked_size > MAX_TILES) max_unlocked_size = MAX_TILES;

                    int next_size = Random({INITIAL_MAZE_SIZE}, max_unlocked_size);
                    current_maze_size = next_size;
                    tm_maze_size = MAP_WIDTH;

                    ACS_NamedExecuteAlways("GenerateMaze", 0, p1_cell_x, p1_cell_y, current_maze_size);
                }}
            }} else {{
                tm_p1_grid_x = -1;
                tm_p1_grid_y = -1;
            }}

            // Keep blind overlay active
            SetFont("S_BLACK");
            HudMessage(s:"A"; HUDMSG_PLAIN, 1, CR_UNTRANSLATED, 160.0, 100.0, 0.0);
            SetFont("SMALLFONT");

            SetActorAngle(1000, 0);
            SetActorPitch(1000, 0);

            Delay(1);
        }}
    }}
    script 3 ENTER {{
        if (PlayerNumber() != 1) terminate;

        Thing_ChangeTID(0, 1001);
        SetActorProperty(0, APROP_Health, 100);
        SetPlayerProperty(0, 1, 0); // PROP_FROZEN
        ClearInventory();

        int r_fix = PLATFORM_RADIUS << 16;

        while(TRUE) {{
            // Broadcast direction via WASD buttons
            int btns = GetPlayerInput(1, INPUT_BUTTONS);
            if (btns & BT_FORWARD)    PrintBold(s:"NORTH");
            if (btns & BT_BACK)       PrintBold(s:"SOUTH");
            if (btns & BT_MOVELEFT)   PrintBold(s:"WEST");
            if (btns & BT_MOVERIGHT)  PrintBold(s:"EAST");

            // Auto-track Navigator on NORTH stand (no side-flip)
            int px = GetActorX(1000);
            int obs_x = px;
            int obs_y = r_fix;
            int obs_z = GetActorZ(0);

            int bound = r_fix - (128 << 16);
            if (obs_x > bound) obs_x = bound;
            if (obs_x < -bound) obs_x = -bound;

            SetActorPosition(0, obs_x, obs_y, obs_z, 0);

            // Always look at Navigator
            int dx = GetActorX(1000) - GetActorX(0);
            int dy = GetActorY(1000) - GetActorY(0);
            int dz = (GetActorZ(1000) + (GetActorProperty(1000, APROP_Height) / 2)) - (GetActorZ(0) + GetActorViewHeight(0));

            int target_angle = VectorAngle(dx, dy);
            int xy_dist = VectorLength(dx, dy);
            int target_pitch = VectorAngle(xy_dist, dz);

            if (target_pitch > 0.5) {{
                target_pitch -= 1.0;
            }}
            target_pitch = -target_pitch;

            SetActorAngle(0, target_angle);
            SetActorPitch(0, target_pitch);

            Delay(1);
        }}
    }}
    script "EndGame" (void) {{
        Delay(1);
        Exit_Normal(0);
    }}
    script 999 DEATH {{
        if (PlayerNumber() != -1) {{
            ACS_NamedExecuteAlways("EndGame", 0);
        }}
    }}
    """

        acs.add_global_code(maze_logic_script)

        source_acs = acs.to_code()
        builder.map_data.scripts = source_acs
        try:
            builder.map_data.behavior = acs.compile()
        except Exception as e:
            print(f"ACS Compile Warning: {e}")
            if hasattr(e, "stderr") and e.stderr:
                print(f"ACC Stderr: {e.stderr}")

        builder.build(output_path)
        print(f"Built: {output_path}")


if __name__ == "__main__":
    TextMazeScenario().generate(str(RUNTIME_SCENARIOS / "text_maze.wad"))
