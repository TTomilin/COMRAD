import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.things import ThingType
from doomgen.logic.acs_builder import ACSBuilder, ScriptType


class ResourceGreedScenario(Scenario):

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "seed": 42,
            "arena_radius": 750,         # half-side of the square combat arena
            "room_dist": 1150,           # center-to-center distance from origin to ammo room
            "room_size": 380,            # side length of each square ammo room
            "corridor_width": 3,         # corridor width in Voronoi cells
            "num_enemy_spots": 8,        # perimeter spawn spots inside the arena
            "enemy_spawn_interval": 5,   # seconds between spawns
            "max_enemies": 12,
            "max_world_ammo": 16,        # global cap for ammo items on the ground
            "starting_ammo": 80,
        }

    def generate(self, output_path: str) -> None:
        cfg = self.config
        seed = cfg["seed"] if cfg["seed"] is not None else random.randint(0, 999999)
        random.seed(seed)

        arena_hw       = cfg["arena_radius"]       # 750
        room_dist      = cfg["room_dist"]          # 1150
        room_size      = cfg["room_size"]          # 380
        room_half      = room_size // 2            # 190
        n_enemy_spots  = cfg["num_enemy_spots"]
        spawn_tics     = cfg["enemy_spawn_interval"] * 35
        max_enemies    = cfg["max_enemies"]
        max_world_ammo = cfg["max_world_ammo"]
        starting_ammo  = cfg["starting_ammo"]

        # Bounds must contain all four ammo rooms
        far_edge = room_dist + room_half            # 1340
        pad      = 400
        bounds   = (-(far_edge + pad), -(far_edge + pad),
                     far_edge + pad,    far_edge + pad)

        builder = ProceduralMapBuilder(bounds=bounds, num_seeds=10000, seed=seed)

        # ------------------------------------------------------------------ #
        # DECORATE                                                            #
        # ------------------------------------------------------------------ #
        decorate_str = """
        ACTOR CHZombie : Zombieman 15020 {
          DropItem "None"
          States {
          Death:
            POSS H 0 ACS_ExecuteAlways(998, 0)
            POSS H 5
            POSS I 5 A_Scream
            POSS J 5 A_NoBlocking
            POSS K 5
            POSS L 5
            POSS M 1 A_FadeOut(0.1)
            Wait
          }
        }
        ACTOR CHImp : DoomImp 15021 {
          DropItem "None"
          States {
          Death:
            TROO I 0 ACS_ExecuteAlways(998, 0)
            TROO I 8
            TROO J 8 A_Scream
            TROO K 6
            TROO L 6 A_NoBlocking
            TROO M 1 A_FadeOut(0.1)
            Wait
          }
        }
        ACTOR CHShotgunGuy : ShotgunGuy 15022 {
          DropItem "None"
          States {
          Death:
            SPOS H 0 ACS_ExecuteAlways(998, 0)
            SPOS H 5
            SPOS I 5 A_Scream
            SPOS J 5 A_NoBlocking
            SPOS K 5
            SPOS L 1 A_FadeOut(0.1)
            Wait
          }
        }
        """
        builder.wad_writer.add_lump("DECORATE", decorate_str.encode("utf-8"))

        # ------------------------------------------------------------------ #
        # ACS builder + globals                                               #
        # ------------------------------------------------------------------ #
        acs = ACSBuilder()
        acs.add_include("zcommon.acs")
        acs.add_global_var("ch_kills_global",       51, "int")
        acs.add_global_var("ch_shared_ammo_global", 52, "int")
        acs.add_global_var("ch_world_ammo_global",  53, "int")
        acs.add_global_var("ch_enemy_count_global", 54, "int")

        # Script 1/2: player setup
        script_setup = f"""
        int pn = PlayerNumber();
        Thing_ChangeTID(0, 1000 + pn);
        SetActorProperty(0, APROP_Health, 100);
        ClearInventory();
        SetAmmoCapacity("Clip", 400);
        GiveInventory("Chaingun", 1);
        GiveInventory("Clip", {starting_ammo});
        GiveInventory("GreenArmor", 1);
        SetWeapon("Chaingun");
        ACS_ExecuteAlways(10, 0, 0, 0, 0);
        """
        acs.add_script(ScriptType.ENTER,   script_setup, number=1)
        acs.add_script(ScriptType.RESPAWN, script_setup, number=2)

        # Script 998: kill reward (called from DECORATE death state)
        acs.add_script(ScriptType.VOID, """
        SetActivatorToTarget(0);
        if (PlayerNumber() >= 0) {
            ch_kills_global++;
        }
        """, number=998)

        # ------------------------------------------------------------------ #
        # MAP LAYOUT                                                          #
        # ------------------------------------------------------------------ #

        # Central combat arena
        builder.add_area(
            "Arena",
            shape=(0, 0, arena_hw * 2, arena_hw * 2),
            floor_height=0,
            ceiling_height=384,
            floor_texture="FLOOR4_8",
            ceiling_texture="CEIL3_5",
            wall_texture="STONE3",
            light_level=192,
        )

        # Four ammo rooms at cardinal positions, outside the arena.
        # Each is connected via a corridor that blocks line-of-sight
        # to the arena — agents cannot shoot enemies from inside a room.
        room_specs: List[Tuple[str, int, int]] = [
            ("AmmoRoom_N",  0,           room_dist),
            ("AmmoRoom_S",  0,          -room_dist),
            ("AmmoRoom_E",  room_dist,   0),
            ("AmmoRoom_W", -room_dist,   0),
        ]

        for name, rx, ry in room_specs:
            builder.add_area(
                name,
                shape=(rx, ry, room_size, room_size),
                floor_height=0,
                ceiling_height=256,
                floor_texture="CEIL5_1",   # bright texture — visually distinct
                ceiling_texture="CEIL3_5",
                wall_texture="BROWN96",
                light_level=210,
            )
            builder.add_corridor(
                "Arena", name,
                width=cfg["corridor_width"],
                floor_texture="FLOOR4_8",
                ceiling_height=200,
                wall_texture="STONE3",
                light_level=160,
            )

        # ------------------------------------------------------------------ #
        # THINGS                                                              #
        # ------------------------------------------------------------------ #

        # Player starts (center of arena)
        builder.add_thing(ThingType.PLAYER1_START, -64,  0, angle=0)
        builder.add_thing(ThingType.PLAYER2_START,  64,  0, angle=0)
        builder.add_thing(ThingType.PLAYER3_START, -64, 64, angle=0)
        builder.add_thing(ThingType.PLAYER4_START,  64, 64, angle=0)

        # Enemy spawn spots around the arena perimeter (TID 2000+)
        enemy_spot_r = int(arena_hw * 0.82)
        for i in range(n_enemy_spots):
            angle = (2 * math.pi * i) / n_enemy_spots
            builder.add_thing(
                "MAP_SPOT",
                int(math.cos(angle) * enemy_spot_r),
                int(math.sin(angle) * enemy_spot_r),
                tid=2000 + i,
            )

        # Ammo spawn spots inside each room (TID 3000+)
        ammo_spot_base = 3000
        spots_per_room = 6
        total_spots    = 0
        # Per-room spot ranges (start, end) for ACS Random() — None if room is empty
        room_spot_ranges: List[Optional[Tuple[int, int]]] = []

        for name, rx, ry in room_specs:
            start = total_spots
            for _ in range(spots_per_room):
                pt = builder.get_random_point_in_area(name, safe=False)
                if pt:
                    builder.add_thing(
                        "MAP_SPOT", pt[0], pt[1],
                        tid=ammo_spot_base + total_spots,
                    )
                    total_spots += 1
            end = total_spots - 1
            room_spot_ranges.append((start, end) if end >= start else None)

        # ------------------------------------------------------------------ #
        # ACS SCRIPTS                                                         #
        # ------------------------------------------------------------------ #

        # Script 3 (OPEN): shared ammo equalization every 4 ticks
        script_eq = f"""
        int p1;
        int p2;
        int total;
        int half;
        int other;

        ch_kills_global       = 0;
        ch_shared_ammo_global = 0;
        ch_world_ammo_global  = 0;
        ch_enemy_count_global = 0;

        while (TRUE) {{
            if (PlayerInGame(0) && PlayerInGame(1)
                    && ThingCount(T_NONE, 1000) > 0
                    && ThingCount(T_NONE, 1001) > 0) {{
                p1    = CheckActorInventory(1000, "Clip");
                p2    = CheckActorInventory(1001, "Clip");
                total = p1 + p2;
                ch_shared_ammo_global = total;
                half  = total / 2;
                other = total - half;
                if (p1 > half)       TakeActorInventory(1000, "Clip", p1 - half);
                else if (p1 < half)  GiveActorInventory(1000, "Clip", half - p1);
                if (p2 > other)      TakeActorInventory(1001, "Clip", p2 - other);
                else if (p2 < other) GiveActorInventory(1001, "Clip", other - p2);
            }} else if (PlayerInGame(0) && ThingCount(T_NONE, 1000) > 0) {{
                ch_shared_ammo_global = CheckActorInventory(1000, "Clip");
            }} else if (PlayerInGame(1) && ThingCount(T_NONE, 1001) > 0) {{
                ch_shared_ammo_global = CheckActorInventory(1001, "Clip");
            }}
            ch_world_ammo_global  = ThingCount(T_NONE, 800);
            ch_enemy_count_global = ThingCount(T_NONE, 700);
            Delay(4);
        }}
        """
        acs.add_script(ScriptType.OPEN, script_eq, number=3)

        # Script 5 (OPEN): presence-triggered ammo spawning, once per second
        # For each room, ammo only spawns if a player is physically inside.
        detect_pad = 120
        spawn_blocks: List[str] = []
        for i, (name, rx, ry) in enumerate(room_specs):
            rng = room_spot_ranges[i]
            if rng is None:
                continue   # room claimed no cells, skip
            s, e = rng
            xmin = rx - room_half - detect_pad
            xmax = rx + room_half + detect_pad
            ymin = ry - room_half - detect_pad
            ymax = ry + room_half + detect_pad
            spawn_blocks.append(f"""
            // {name}
            in_room = 0;
            if (PlayerInGame(0) && ThingCount(T_NONE, 1000) > 0) {{
                px = GetActorX(1000) >> 16;
                py = GetActorY(1000) >> 16;
                if (px >= {xmin} && px <= {xmax} && py >= {ymin} && py <= {ymax}) in_room = 1;
            }}
            if (in_room == 0 && PlayerInGame(1) && ThingCount(T_NONE, 1001) > 0) {{
                px = GetActorX(1001) >> 16;
                py = GetActorY(1001) >> 16;
                if (px >= {xmin} && px <= {xmax} && py >= {ymin} && py <= {ymax}) in_room = 1;
            }}
            if (in_room && ThingCount(T_NONE, 800) < {max_world_ammo}) {{
                SpawnSpot("Clip", {ammo_spot_base} + Random({s}, {e}), 800, 0);
            }}
            """)

        spawn_code = "\n".join(spawn_blocks)
        script_spawn = f"""
        int in_room;
        int px;
        int py;

        Delay(35 * 3);

        while (TRUE) {{
            {spawn_code}
            Delay(35);
        }}
        """
        acs.add_script(ScriptType.OPEN, script_spawn, number=5)

        # Script 4 (OPEN): enemy wave spawner
        script_spawner = f"""
        Delay(35 * 5);
        while (TRUE) {{
            if (ThingCount(T_NONE, 700) < {max_enemies}) {{
                int spot_tid = 2000 + Random(0, {n_enemy_spots - 1});
                if (ThingCount(T_NONE, spot_tid) > 0) {{
                    int r = Random(0, 9);
                    str etype = "CHZombie";
                    if (r > 5) etype = "CHImp";
                    if (r > 8) etype = "CHShotgunGuy";
                    SpawnSpotFacing(etype, spot_tid, 700);
                }}
            }}
            Delay({spawn_tics});
        }}
        """
        acs.add_script(ScriptType.OPEN, script_spawner, number=4)

        # Script 10 (VOID): per-player HUD loop
        script_hud = f"""
        while (TRUE) {{
            SetFont("SMALLFONT");
            int total_ammo = ch_shared_ammo_global;
            int kills      = ch_kills_global;
            int enemies    = ch_enemy_count_global;
            int on_ground  = ch_world_ammo_global;
            str ammo_color = "\\cC";
            if (total_ammo < 40) ammo_color = "\\cR";
            else if (total_ammo < 80) ammo_color = "\\cE";
            HudMessage(s:"SHARED AMMO: ", s:ammo_color, d:total_ammo;
                HUDMSG_PLAIN, 1, CR_WHITE, 0.5, 0.80, 0.1);
            HudMessage(s:"KILLS: ", d:kills, s:"  ENEMIES: ", d:enemies;
                HUDMSG_PLAIN, 2, CR_WHITE, 0.5, 0.83, 0.1);
            HudMessage(s:"ON GROUND: ", d:on_ground;
                HUDMSG_PLAIN, 3, CR_GOLD, 0.5, 0.86, 0.1);
            Delay(1);
        }}
        """
        acs.add_script(ScriptType.VOID, script_hud, number=10)

        # ------------------------------------------------------------------ #
        # Compile ACS + build WAD                                             #
        # ------------------------------------------------------------------ #
        try:
            builder.map_data.behavior = acs.compile()
        except Exception as e:
            print(f"ACS compile warning: {e}")
            if hasattr(e, "stderr") and e.stderr:
                print(f"ACC stderr: {e.stderr}")

        builder.build(output_path)


if __name__ == "__main__":
    out_dir = os.path.join("examples", "benchmark", "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "common_harvest_doom.wad")
    print(f"Generating {path}...")
    ResourceGreedScenario().generate(path)
    print("Done.")
