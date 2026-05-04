import os
import random
import math
from shapely.geometry import Polygon
from typing import Any, Dict, List

from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.things import ThingType
from doomgen.logic.acs_builder import ACSBuilder, ScriptType

class CoopHealthGatheringScenario(Scenario):
    """
    Cooperative health gathering scenario.
    Players must navigate a toxic maze while chained together.
    Health items spawn randomly around the map.
    Game ends if any player dies.
    """

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "seed": 42,
            "max_players": 4,
            "map_radius": 2000,
            "chain_max_len": 400,
            "toxic_damage": 5,
            "max_health_kits": 80,
            "maze_density": 0.6,
        }

    def _decorate_lump(self) -> str | None:
        return """
ACTOR ChainNodeCalm {
  Radius 1
  Height 1
  Scale 0.13
  RenderStyle Add
  Alpha 0.55
  +NOBLOCKMAP +NOGRAVITY +NOINTERACTION +DONTSPLASH
  States {
    Spawn:
      APLS AB 3 BRIGHT // Arachnotron plasma (Green/Blue)
      Loop
  }
}

ACTOR ChainNodeWarn {
  Radius 1
  Height 1
  Scale 0.13
  RenderStyle Add
  Alpha 0.6
  +NOBLOCKMAP +NOGRAVITY +NOINTERACTION +DONTSPLASH
  States {
    Spawn:
      BAL1 AB 3 BRIGHT // Mancubus fireball (Orange)
      Loop
  }
}

ACTOR ChainNodeHot {
  Radius 1
  Height 1
  Scale 0.13
  RenderStyle Add
  Alpha 0.65
  Translation "160:166=176:182,225:227=176:178"
  +NOBLOCKMAP +NOGRAVITY +NOINTERACTION +DONTSPLASH
  States {
    Spawn:
      BAL1 AB 3 BRIGHT // Round orb (translated to red)
      Loop
  }
}
"""

    def _enter_visual_script(self, max_players: int, chain_max_len: int) -> str:
        bar_cases = []
        for i in range(21):
            filled = "=" * i
            empty = "." * (20 - i)
            bar_cases.append(f'case {i}: tether_bar = "o{filled}{empty}o"; break;')
        bar_switch = "\n                    ".join(bar_cases)

        return f"""
        int overlay_tick = 0;
        int chain_limit = {chain_max_len} << 16;

        while (TRUE)
        {{
            if (overlay_tick % 5 == 0)
            {{
                int farthest = 0;
                int teammate_found = 0;
                int my_x = GetActorX(1000 + pn);
                int my_y = GetActorY(1000 + pn);

                int other;
                for (other = 0; other < {max_players}; other++)
                {{
                    if (other == pn) continue;
                    if (!PlayerInGame(other)) continue;

                    int other_tid = 1000 + other;
                    if (ThingCount(0, other_tid) <= 0) continue;

                    int dx = my_x - GetActorX(other_tid);
                    int dy = my_y - GetActorY(other_tid);
                    int adx = dx;
                    int ady = dy;
                    if (adx < 0) adx = -adx;
                    if (ady < 0) ady = -ady;

                    int d;
                    if (adx > ady) {{
                        d = adx + (ady >> 1);
                    }} else {{
                        d = ady + (adx >> 1);
                    }}

                    if (!teammate_found || d > farthest) farthest = d;
                    teammate_found = 1;
                }}

                SetHudSize(320, 240, 1);
                SetFont("SMALLFONT");

                if (!teammate_found)
                {{
                    HudMessage(s:""; HUDMSG_PLAIN, 211, CR_WHITE, 0.5, 0.86, 0.2);
                    HudMessage(s:""; HUDMSG_PLAIN, 212, CR_WHITE, 0.5, 0.90, 0.2);
                }}
                else
                {{
                    int tension_idx = 0;
                    if (chain_limit > 0) {{
                        tension_idx = (farthest * 20) / chain_limit;
                    }}
                    if (tension_idx < 0) tension_idx = 0;
                    if (tension_idx > 20) tension_idx = 20;

                    str tether_bar = "";
                    switch (tension_idx) {{
                        {bar_switch}
                    }}

                    if (chg_dragged_links_flash_global > 0 || farthest > chain_limit)
                    {{
                        HudMessage(s:tether_bar; HUDMSG_PLAIN, 212, CR_RED, 0.5, 0.90, 0.2);
                    }}
                    else if (farthest > (chain_limit * 9) / 10)
                    {{
                        HudMessage(s:tether_bar; HUDMSG_PLAIN, 212, CR_RED, 0.5, 0.90, 0.2);
                    }}
                    else if (farthest > (chain_limit * 3) / 4)
                    {{
                        HudMessage(s:tether_bar; HUDMSG_PLAIN, 212, CR_GOLD, 0.5, 0.90, 0.2);
                    }}
                    else
                    {{
                        HudMessage(s:tether_bar; HUDMSG_PLAIN, 212, CR_GREEN, 0.5, 0.90, 0.2);
                    }}
                }}
            }}

            Delay(1);
            overlay_tick++;
        }}
        """

    def _chain_visual_lines(self, follower: int) -> str:
        tether_nodes = 30
        tether_tid_base = 6000 + (follower * 64)
        return f"""
                        if (tick % 5 == 0) {{
                            int node_idx_{follower};
                            for (node_idx_{follower} = 0; node_idx_{follower} < {tether_nodes}; node_idx_{follower}++) {{
                                Thing_Remove({tether_tid_base} + node_idx_{follower});
                            }}

                            if (dist_{follower} > 0) {{
                                int tether_az_{follower} = az_{follower} + (20 << 16);
                                int tether_bz_{follower} = bz_{follower} + (20 << 16);
                                int tether_dx_{follower} = render_bx_{follower} - render_ax_{follower};
                                int tether_dy_{follower} = render_by_{follower} - render_ay_{follower};
                                int tether_dz_{follower} = tether_bz_{follower} - tether_az_{follower};

                                for (node_idx_{follower} = 0; node_idx_{follower} < {tether_nodes}; node_idx_{follower}++) {{
                                    int frac_{follower} = FixedDiv((node_idx_{follower} + 1) << 16, {tether_nodes + 1} << 16);
                                    int sx_{follower} = render_ax_{follower} + FixedMul(tether_dx_{follower}, frac_{follower});
                                    int sy_{follower} = render_ay_{follower} + FixedMul(tether_dy_{follower}, frac_{follower});
                                    int sz_{follower} = tether_az_{follower} + FixedMul(tether_dz_{follower}, frac_{follower});
                                    int spawn_tid_{follower} = {tether_tid_base} + node_idx_{follower};

                                    if (tether_state_{follower} >= 2) {{
                                        Spawn("ChainNodeHot", sx_{follower}, sy_{follower}, sz_{follower}, spawn_tid_{follower});
                                    }} else if (tether_state_{follower} == 1) {{
                                        Spawn("ChainNodeWarn", sx_{follower}, sy_{follower}, sz_{follower}, spawn_tid_{follower});
                                    }} else {{
                                        Spawn("ChainNodeCalm", sx_{follower}, sy_{follower}, sz_{follower}, spawn_tid_{follower});
                                    }}
                                }}
                            }}
                        }}
        """

    def generate(self, output_path: str) -> None:
        cfg = self.config
        seed = cfg.get("seed")
        if seed is None:
            seed = random.randint(0, 999999)
        random.seed(seed)

        max_players = int(cfg["max_players"])
        map_radius = int(cfg["map_radius"])
        chain_max_len = int(cfg["chain_max_len"])
        toxic_damage_amt = int(cfg["toxic_damage"])
        max_health_kits = int(cfg["max_health_kits"])
        maze_density = float(cfg["maze_density"])

        builder = ProceduralMapBuilder(
            bounds=(-map_radius, -map_radius, map_radius, map_radius),
            num_seeds=8000,
            seed=seed
        )

        decorate_lump = self._decorate_lump()
        if decorate_lump:
            builder.wad_writer.add_lump("DECORATE", decorate_lump.encode("utf-8"))

        floor_tag = 900

        num_points = random.randint(12, 24)
        base_points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            r = random.uniform(map_radius * 0.6, map_radius)
            base_points.append((r * math.cos(angle), r * math.sin(angle)))

        base_shape = Polygon(base_points)

        builder.add_area(
            "BaseArena",
            shape=base_shape,
            floor_height=0,
            ceiling_height=384,
            floor_texture="FLAT1",
            ceiling_texture="CEIL1_2",
            wall_texture="BRICK6",
            light_level=200,
            tag=floor_tag,
            mode="claim_void"
        )

        block_size = 140
        num_pillars = int(map_radius * map_radius * 4 / (block_size * block_size * 4) * maze_density)

        block_idx = 0
        for _ in range(num_pillars):
            pt = builder.get_random_point_in_area("BaseArena", safe=True)
            if not pt:
                continue

            if abs(pt[0]) < 256 and abs(pt[1]) < 256:
                continue

            w = random.randint(48, 128)
            h = random.randint(48, 128)
            x, y = pt

            builder.add_area(
                f"Pillar_{block_idx}",
                shape=(x, y, w, h),
                floor_height=0,
                ceiling_height=0,
                floor_texture="FLAT1",
                wall_texture="ASHWALL6",
                light_level=200,
                mode="overwrite"
            )
            block_idx += 1

        spawn_center = builder.get_random_point_in_area("BaseArena", safe=True)
        if not spawn_center:
            spawn_center = (0, 0)

        spawn_points = []
        for _ in range(100):
            if len(spawn_points) >= max_players:
                break

            pt = builder.get_random_point_in_area("BaseArena", safe=True)
            if not pt:
                continue

            dist_sq = (pt[0] - spawn_center[0])**2 + (pt[1] - spawn_center[1])**2
            if dist_sq < (chain_max_len / 2)**2:
                too_close = False
                for sp in spawn_points:
                    if (pt[0] - sp[0])**2 + (pt[1] - sp[1])**2 < 64**2:
                        too_close = True
                        break
                if not too_close:
                    spawn_points.append(pt)

        while len(spawn_points) < max_players:
            spawn_points.append((spawn_center[0] + random.randint(-64, 64),
                                 spawn_center[1] + random.randint(-64, 64)))

        player_types = [
            ThingType.PLAYER1_START,
            ThingType.PLAYER2_START,
            ThingType.PLAYER3_START,
            ThingType.PLAYER4_START
        ]

        for i in range(max_players):
            if i < len(player_types):
                builder.add_thing(player_types[i], spawn_points[i][0], spawn_points[i][1], angle=random.randint(0, 359))

        num_spawn_spots = 300
        spot_tid_base = 3000

        spawned_spots = []
        attempts = 0
        min_distance_sq = (map_radius * 0.15) ** 2  # Keep spots reasonably apart

        while len(spawned_spots) < num_spawn_spots and attempts < num_spawn_spots * 10:
            attempts += 1
            pt = builder.get_random_point_in_area("BaseArena", safe=True)
            if not pt:
                continue

            # Check distance against already placed spots for more even distribution
            too_close = False
            for spt in spawned_spots:
                dist_sq = (pt[0] - spt[0])**2 + (pt[1] - spt[1])**2
                if dist_sq < min_distance_sq:
                    too_close = True
                    break

            if not too_close:
                builder.add_thing("MAP_SPOT", pt[0], pt[1], z=0, tid=spot_tid_base + len(spawned_spots))
                spawned_spots.append(pt)

        # If we failed to place enough because of the strict distance, fallback to random
        while len(spawned_spots) < num_spawn_spots:
            pt = builder.get_random_point_in_area("BaseArena", safe=True)
            if pt:
                builder.add_thing("MAP_SPOT", pt[0], pt[1], z=0, tid=spot_tid_base + len(spawned_spots))
                spawned_spots.append(pt)

        acs = ACSBuilder()
        acs.add_include("zcommon.acs")

        acs.add_global_var("chg_active_health_kits_global", 1, "int")
        acs.add_global_var("chg_total_spawned_kits_global", 2, "int")
        acs.add_global_var("chg_chain_stretches_global", 3, "int")
        acs.add_global_var("chg_total_pickups_global", 4, "int")
        acs.add_global_var("chg_dragged_links_flash_global", 5, "int")

        acs.add_script(
            ScriptType.VOID,
            "chg_total_pickups_global++;",
            number=10
        )

        script_setup = f"""
        int pn = PlayerNumber();
        if (pn >= 0 && pn < {max_players}) {{
            Thing_ChangeTID(0, 1000 + pn);
            SetActorProperty(0, APROP_Health, 100);
            TakeInventory("Fist", 999);
            TakeInventory("Pistol", 999);
        }}

        {self._enter_visual_script(max_players, chain_max_len)}
        """
        acs.add_script(ScriptType.ENTER, script_setup, number=1)
        acs.add_script(ScriptType.RESPAWN, script_setup, number=2)

        chain_lines: List[str] = []
        for follower in range(1, max_players):
            leader = follower - 1
            chain_lines.append(
                f"""
                if (PlayerInGame({leader}) && PlayerInGame({follower})) {{
                    int a_tid_{follower} = {1000 + leader};
                    int b_tid_{follower} = {1000 + follower};
                    if (ThingCount(0, a_tid_{follower}) > 0 && ThingCount(0, b_tid_{follower}) > 0) {{
                        int ax_{follower} = GetActorX(a_tid_{follower});
                        int ay_{follower} = GetActorY(a_tid_{follower});
                        int az_{follower} = GetActorZ(a_tid_{follower});
                        int bx_{follower} = GetActorX(b_tid_{follower});
                        int by_{follower} = GetActorY(b_tid_{follower});
                        int bz_{follower} = GetActorZ(b_tid_{follower});

                        int dx_{follower} = ax_{follower} - bx_{follower};
                        int dy_{follower} = ay_{follower} - by_{follower};
                        int adx_{follower} = dx_{follower};
                        int ady_{follower} = dy_{follower};

                        if (adx_{follower} < 0) adx_{follower} = -adx_{follower};
                        if (ady_{follower} < 0) ady_{follower} = -ady_{follower};

                        int dist_{follower};
                        if (adx_{follower} > ady_{follower}) {{
                            dist_{follower} = adx_{follower} + (ady_{follower} >> 1);
                        }} else {{
                            dist_{follower} = ady_{follower} + (adx_{follower} >> 1);
                        }}

                        int chain_limit_{follower} = {chain_max_len} << 16;
                        int render_ax_{follower} = ax_{follower};
                        int render_ay_{follower} = ay_{follower};
                        int render_bx_{follower} = bx_{follower};
                        int render_by_{follower} = by_{follower};
                        int tether_state_{follower} = 0;

                        if (dist_{follower} > (chain_limit_{follower} * 9) / 10) {{
                            tether_state_{follower} = 2;
                        }} else if (dist_{follower} > (chain_limit_{follower} * 3) / 4) {{
                            tether_state_{follower} = 1;
                        }}

                        if (dist_{follower} > chain_limit_{follower} && dist_{follower} > 0) {{
                            int keep_half_{follower} = ({max(64, chain_max_len - 24)} << 15);
                            int ratio_{follower} = FixedDiv(keep_half_{follower}, dist_{follower});

                            int mx_{follower} = (ax_{follower} + bx_{follower}) >> 1;
                            int my_{follower} = (ay_{follower} + by_{follower}) >> 1;
                            int nax_{follower} = mx_{follower} + FixedMul(dx_{follower}, ratio_{follower});
                            int nay_{follower} = my_{follower} + FixedMul(dy_{follower}, ratio_{follower});
                            int nbx_{follower} = mx_{follower} - FixedMul(dx_{follower}, ratio_{follower});
                            int nby_{follower} = my_{follower} - FixedMul(dy_{follower}, ratio_{follower});

                            SetActorPosition(a_tid_{follower}, nax_{follower}, nay_{follower}, GetActorZ(a_tid_{follower}), 0);
                            SetActorPosition(b_tid_{follower}, nbx_{follower}, nby_{follower}, GetActorZ(b_tid_{follower}), 0);

                            render_ax_{follower} = nax_{follower};
                            render_ay_{follower} = nay_{follower};
                            render_bx_{follower} = nbx_{follower};
                            render_by_{follower} = nby_{follower};
                            tether_state_{follower} = 2;

                            chg_chain_stretches_global++;
                            chg_dragged_links_flash_global = 6;
                        }}

                        {self._chain_visual_lines(follower)}
                    }}
                }}
                """
            )

        chain_block = "".join(chain_lines)

        script_main = f"""
        Sector_SetDamage({floor_tag}, {toxic_damage_amt}, 14);

        int tick = 0;
        int active_health_kits = 0;
        int health_kit_tid = 800;

        chg_active_health_kits_global = 0;
        chg_total_spawned_kits_global = 0;
        chg_chain_stretches_global = 0;
        chg_dragged_links_flash_global = 0;
        chg_total_pickups_global = 0;

        while (TRUE) {{
            {chain_block}

            if (chg_dragged_links_flash_global > 0) {{
                chg_dragged_links_flash_global--;
            }}

            active_health_kits = ThingCount(T_NONE, health_kit_tid);
            chg_active_health_kits_global = active_health_kits;

            if (tick % 35 == 0) {{
                if (active_health_kits < {max_health_kits}) {{
                    int spawns = 0;
                    for (spawns = 0; spawns < 3; spawns++) {{
                        if (active_health_kits + spawns >= {max_health_kits}) break;

                        int r_spot = Random(0, {num_spawn_spots - 1});
                        int target_spot_tid = {spot_tid_base} + r_spot;

                        str type = "Stimpack";
                        if (Random(0, 3) == 0) type = "Medikit";

                        SpawnSpot(type, target_spot_tid, health_kit_tid, 0);
                        SetThingSpecial(health_kit_tid, 80, 10, 0, 0, 0, 0);
                        chg_total_spawned_kits_global++;
                    }}
                    chg_active_health_kits_global = ThingCount(T_NONE, health_kit_tid);
                }}
            }}

            tick++;
            Delay(1);
        }}
        """
        acs.add_script(ScriptType.OPEN, script_main, number=3)

        acs.add_script(
            ScriptType.DEATH,
            """
            Delay(1);
            Exit_Normal(0);
            """,
            number=4
        )

        source_acs = acs.to_code()
        builder.map_data.scripts = source_acs
        try:
            builder.map_data.behavior = acs.compile()
        except Exception as e:
            print(f"ACS Compile Warning: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"ACC Stderr: {e.stderr}")

        builder.build(output_path)

if __name__ == "__main__":
    out_dir = os.path.join("examples", "benchmark", "output")
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, "coop_health_gathering.wad")
    print(f"Generating {path}...")

    scen = CoopHealthGatheringScenario()
    scen.generate(path)
    print("Done.")
