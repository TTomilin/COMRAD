import os
import random
from typing import Any, Dict, List, Tuple

from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.things import ThingType
from doomgen.logic.acs_builder import ACSBuilder, ScriptType


class StealthLabyrinthScenario(Scenario):
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "seed": 42,
            "start_room_size": 320,
            "hub_size": 448,
            "num_branches": 1,
            "rooms_per_branch": 2,
            "branch_room_width": 416,
            "branch_room_length": 896,
            "start_center_y": -560,
            "branch_offset": 980,
            "turret_wall_margin": 144,
            "first_room_turret_wall_margin": 320,
            "corridor_width": 4,
            "min_enemies_per_room": 1,
            "max_enemies_per_room": 1,
            "turret_health": 140,
            "first_room_turret_health": 90,
            "turret_damage": 2,
            "turret_projectile_speed": 14, # hard is 18
            "turret_burst_count": 2,
            "turret_refire_tics": 14,
            # Learnability-reference variants (COMRAD paper, App. very_difficult); defaults keep the canonical task.
            "always_lit": False,  # rooms/corridors permanently lit; turret invulnerability still gated by the scout
            "scripted_scout": False,  # ACS drives the scout (player 0) to each active room threshold
            "scout_speed": 8,  # map units per tic
            "scout_hold_depth": 96,  # how far inside the active room the scout holds
            "scout_strafe": 96,  # lateral dodge amplitude while holding
            # tics per full dodge cycle; ~2x the turret bolt flight time so the scout never sweeps back into the aim point
            "scout_strafe_period": 72,
            # converse reference: ACS drives the gunner (player 1), which trails the scout and shoots vulnerable turrets
            "scripted_gunner": False,
            "gunner_speed": 12,  # map units per tic along the scout's trail
            "gunner_follow_dist": 96,  # gunner stops advancing when this close to the scout
            "gunner_trail_spacing": 24,  # scout breadcrumb spacing
            "gunner_damage": 10,  # per hitscan bullet (chaingun average)
            "gunner_refire_tics": 4,
            "gunner_range": 2048,
        }

    def validate_config(self) -> None:
        if int(self.config["start_room_size"]) < 192:
            raise ValueError("start_room_size must be >= 192")
        if int(self.config["hub_size"]) < 256:
            raise ValueError("hub_size must be >= 256")
        if int(self.config["num_branches"]) < 1 or int(self.config["num_branches"]) > 3:
            raise ValueError("num_branches must be in [1, 3]")
        if int(self.config["rooms_per_branch"]) < 1:
            raise ValueError("rooms_per_branch must be >= 1")
        if int(self.config["branch_room_width"]) < 256:
            raise ValueError("branch_room_width must be >= 256")
        if int(self.config["branch_room_length"]) < 512:
            raise ValueError("branch_room_length must be >= 512")
        if int(self.config["start_center_y"]) > -192:
            raise ValueError("start_center_y must stay below the hub")
        if int(self.config["branch_offset"]) < 640:
            raise ValueError("branch_offset must be >= 640")
        if int(self.config["turret_wall_margin"]) < 64:
            raise ValueError("turret_wall_margin must be >= 64")
        if int(self.config["first_room_turret_wall_margin"]) < int(self.config["turret_wall_margin"]):
            raise ValueError("first_room_turret_wall_margin must be >= turret_wall_margin")
        if int(self.config["corridor_width"]) < 3:
            raise ValueError("corridor_width must be >= 3")
        if int(self.config["min_enemies_per_room"]) < 1:
            raise ValueError("min_enemies_per_room must be >= 1")
        if int(self.config["max_enemies_per_room"]) < int(self.config["min_enemies_per_room"]):
            raise ValueError("max_enemies_per_room must be >= min_enemies_per_room")
        if int(self.config["max_enemies_per_room"]) > 2:
            raise ValueError("max_enemies_per_room must be <= 2")
        if int(self.config["turret_health"]) < 40:
            raise ValueError("turret_health must be >= 40")
        if int(self.config["first_room_turret_health"]) < 20:
            raise ValueError("first_room_turret_health must be >= 20")
        if int(self.config["first_room_turret_health"]) > int(self.config["turret_health"]):
            raise ValueError("first_room_turret_health must be <= turret_health")
        if int(self.config["turret_damage"]) < 1:
            raise ValueError("turret_damage must be >= 1")
        if int(self.config["turret_projectile_speed"]) < 8:
            raise ValueError("turret_projectile_speed must be >= 8")
        if int(self.config["turret_burst_count"]) < 1:
            raise ValueError("turret_burst_count must be >= 1")
        if int(self.config["turret_refire_tics"]) < 4:
            raise ValueError("turret_refire_tics must be >= 4")
        for key in ("always_lit", "scripted_scout", "scripted_gunner"):
            if not isinstance(self.config[key], bool):
                raise ValueError(f"{key} must be a bool")
        # scripted_scout + scripted_gunner together is the fully scripted upper reference (no learner acts)
        for key, low, high in (
            ("gunner_speed", 1, 24),
            ("gunner_follow_dist", 32, 512),
            ("gunner_trail_spacing", 8, 128),
            ("gunner_damage", 1, 100),
            ("gunner_refire_tics", 1, 35),
            ("gunner_range", 256, 8192),
        ):
            if not low <= int(self.config[key]) <= high:
                raise ValueError(f"{key} must be in [{low}, {high}]")
        if int(self.config["scout_speed"]) < 1 or int(self.config["scout_speed"]) > 24:
            raise ValueError("scout_speed must be in [1, 24]")
        if int(self.config["scout_hold_depth"]) < 0 or int(self.config["scout_hold_depth"]) > int(self.config["branch_room_length"]) // 2:
            raise ValueError("scout_hold_depth must be in [0, branch_room_length / 2]")
        if int(self.config["scout_strafe"]) < 0 or int(self.config["scout_strafe"]) > int(self.config["branch_room_width"]) // 2 - 48:
            raise ValueError("scout_strafe must be in [0, branch_room_width / 2 - 48]")
        if int(self.config["scout_strafe_period"]) < 4:
            raise ValueError("scout_strafe_period must be >= 4")

    @staticmethod
    def _square_check(px: str, py: str, cx: int, cy: int, radius: int) -> str:
        return StealthLabyrinthScenario._rect_check(px, py, cx, cy, radius, radius)

    @staticmethod
    def _rect_check(px: str, py: str, cx: int, cy: int, half_w: int, half_h: int) -> str:
        scale = 1 << 16
        return (
            f"{px} >= {(cx - half_w) * scale} && {px} <= {(cx + half_w) * scale} && "
            f"{py} >= {(cy - half_h) * scale} && {py} <= {(cy + half_h) * scale}"
        )

    @staticmethod
    def _scout_thresholds(branch_specs, hold_depth: int):
        """Per stage (branch-major, room-minor): hold point `hold_depth` units inside the room's near edge, the hold
        facing angle (ACS fixed-point, 1.0 = 65536 = full turn) and the lateral strafe axis."""
        stages = []
        for branch in branch_specs:
            sign = int(branch["sign"])
            for (cx, cy), (width, height) in zip(branch["room_centers"], branch["room_dims"]):
                if branch["axis"] == "y":
                    point = (cx, cy - sign * (height // 2 - hold_depth))
                    angle = 16384 if sign > 0 else 49152
                    strafe_axis = (1, 0)
                else:
                    point = (cx - sign * (width // 2 - hold_depth), cy)
                    angle = 0 if sign > 0 else 32768
                    strafe_axis = (0, 1)
                stages.append({"point": point, "angle": angle, "strafe": strafe_axis})
        return stages

    @staticmethod
    def _walkable_route(builder, area_ids, start: Tuple[int, int], goal: Tuple[int, int], clearance: int = 24):
        """Waypoints strictly between `start` and `goal` that keep a `clearance`-wide body inside cells of `area_ids`.
        Corridors are A* chains of Voronoi cells and can bend, so the straight start->goal line may leave the floor.
        Returns [] if the straight line is clear, otherwise a string-pulled path of cell seeds."""
        import heapq

        import numpy as np
        from scipy.spatial import cKDTree

        pts = np.array([[s.x, s.y] for s in builder.seeds], dtype=float)
        allowed = set(area_ids)
        walk = np.array([s.area_id in allowed for s in builder.seeds])
        tree = cKDTree(pts)
        offsets = [(0, 0), (clearance, 0), (-clearance, 0), (0, clearance), (0, -clearance)]

        def clear(p, q) -> bool:
            p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
            n = max(2, int(np.linalg.norm(q - p) / 8) + 1)
            for t in np.linspace(0.0, 1.0, n):
                x = p + t * (q - p)
                for off in offsets:
                    if not walk[tree.query(x + off)[1]]:
                        return False
            return True

        if clear(start, goal):
            return []
        src = int(tree.query(start)[1])
        dst = int(tree.query(goal)[1])
        dist = {src: 0.0}
        prev: Dict[int, int] = {}
        heap = [(0.0, src)]
        while heap:
            d, cur = heapq.heappop(heap)
            if cur == dst:
                break
            if d > dist.get(cur, float("inf")):
                continue
            for nb in builder.mesh.adjacency.get(cur, []):
                if not walk[nb]:
                    continue
                nd = d + float(np.linalg.norm(pts[nb] - pts[cur]))
                if nd < dist.get(nb, float("inf")):
                    dist[nb] = nd
                    prev[nb] = cur
                    heapq.heappush(heap, (nd, nb))
        if dst not in dist:
            raise RuntimeError(f"no walkable route from {start} to {goal}")
        chain = [dst]
        while chain[-1] != src:
            chain.append(prev[chain[-1]])
        path = [tuple(start)] + [(int(round(pts[i][0])), int(round(pts[i][1]))) for i in reversed(chain)] + [tuple(goal)]
        # string pulling: from each anchor jump to the furthest path point still in clear line of sight
        out: List[Tuple[int, int]] = []
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not clear(path[i], path[j]):
                j -= 1
            if j < len(path) - 1:
                out.append(path[j])
            i = j
        return out

    @classmethod
    def _scripted_scout_acs(
        cls,
        branch_specs,
        rooms_per_branch: int,
        start_center: Tuple[int, int],
        start_corr_points: List[Tuple[int, int]],
        hub_center: Tuple[int, int],
        hold_depth: int,
        speed: int,
        strafe: int,
        strafe_period: int,
    ) -> Tuple[str, str]:
        """
        ACS for the scripted-scout variant. Returns (code placed in the OPEN loop that mirrors the active stage into
        `sl_act_room` / USER52, global code with waypoint helpers and the scout `script 4 ENTER`).
        The scout (player 0, TID 1000) walks start -> start corridor waypoints -> hub -> active room hold point with
        SetActorPosition steps, holds there strafing laterally (to dodge turret bolts) until the room is cleared,
        and returns through the hub when the next active room lies on a different branch. It never attacks.
        """
        stages = cls._scout_thresholds(branch_specs, hold_depth)
        hub_wi = len(start_corr_points) + 1  # waypoint index of the hub; hub_wi + 1 is the stage hold point
        corr_x = "".join(f"    if (wi == {i + 1}) return {p[0]};\n" for i, p in enumerate(start_corr_points))
        corr_y = "".join(f"    if (wi == {i + 1}) return {p[1]};\n" for i, p in enumerate(start_corr_points))

        mirror = ["sl_act_room = -1;"]
        for branch_idx in range(len(branch_specs)):
            keyword = "if" if branch_idx == 0 else "else if"
            mirror.append(
                f"{keyword} (branch_active_room_{branch_idx} >= 0) {{ "
                f"sl_act_room = {branch_idx * rooms_per_branch} + branch_active_room_{branch_idx}; }}"
            )
        stage_mirror_code = "\n        ".join(mirror)

        def stage_switch(values, fallback):
            lines = [f"    if (st == {idx}) return {value};" for idx, value in enumerate(values)]
            return "\n".join(lines) + f"\n    return {fallback};"

        wp_x = stage_switch([s["point"][0] for s in stages], hub_center[0])
        wp_y = stage_switch([s["point"][1] for s in stages], hub_center[1])
        ang = stage_switch([s["angle"] for s in stages], 16384)
        str_x = stage_switch([s["strafe"][0] for s in stages], 0)
        str_y = stage_switch([s["strafe"][1] for s in stages], 0)

        scout_code = f"""
function int SLWpX(int st, int wi)
{{
    if (wi <= 0) return {start_center[0]};
{corr_x}    if (wi == {hub_wi}) return {hub_center[0]};
{wp_x}
}}

function int SLWpY(int st, int wi)
{{
    if (wi <= 0) return {start_center[1]};
{corr_y}    if (wi == {hub_wi}) return {hub_center[1]};
{wp_y}
}}

function int SLAng(int st)
{{
{ang}
}}

function int SLStrX(int st)
{{
{str_x}
}}

function int SLStrY(int st)
{{
{str_y}
}}

script 4 ENTER
{{
    if (PlayerNumber() != 0) terminate;
    int st = -1;
    int prev = -1;
    int wi = 0;
    int hold = 0;
    int t = 0;
    int ang = 16384;
    int x, y, tx, ty, dx, dy, d, nx, ny, ph, sof;
    Delay(1);
    SetPlayerProperty(0, 1, PROP_FROZEN);
    while (TRUE)
    {{
        if (ThingCount(T_NONE, 1000) <= 0) terminate;
        if (GetActorProperty(1000, APROP_Health) <= 0) terminate;
        st = sl_act_room;
        if (!sl_started) st = -1;
        if (st != prev)
        {{
            hold = 0;
            t = 0;
            if (prev >= 0 && st >= 0 && (st / {rooms_per_branch}) != (prev / {rooms_per_branch}) && wi > {hub_wi}) wi = {hub_wi};
        }}
        if (st >= 0)
        {{
            x = GetActorX(1000) >> 16;
            y = GetActorY(1000) >> 16;
            tx = SLWpX(st, wi);
            ty = SLWpY(st, wi);
            if (hold)
            {{
                t++;
                ph = t % {strafe_period};
                sof = (ph * {4 * strafe}) / {strafe_period};
                if (sof > {2 * strafe}) sof = {4 * strafe} - sof;
                sof = sof - {strafe};
                tx = tx + SLStrX(st) * sof;
                ty = ty + SLStrY(st) * sof;
            }}
            dx = tx - x;
            dy = ty - y;
            d = VectorLength(dx, dy);
            if (d <= {speed})
            {{
                nx = tx;
                ny = ty;
                if (wi < {hub_wi + 1}) wi++;
                else hold = 1;
            }}
            else
            {{
                nx = x + (dx * {speed}) / d;
                ny = y + (dy * {speed}) / d;
            }}
            if (!SetActorPosition(1000, nx << 16, ny << 16, GetActorZ(1000), 0))
            {{
                SetActorPosition(1000, (nx + SLStrX(st) * 40) << 16, (ny + SLStrY(st) * 40) << 16, GetActorZ(1000), 0);
            }}
            SetActorVelocity(1000, 0, 0, 0, FALSE, FALSE);
            if (hold) ang = SLAng(st);
            else if (d > 0) ang = VectorAngle(dx << 16, dy << 16);
            SetActorAngle(1000, ang);
        }}
        prev = st;
        Delay(1);
    }}
}}
"""
        return stage_mirror_code, scout_code

    @staticmethod
    def _scripted_gunner_acs(
        enemy_tids: List[int],
        speed: int,
        follow_dist: int,
        trail_spacing: int,
        damage: int,
        refire_tics: int,
        fire_range: int,
        trail_len: int = 512,
    ) -> str:
        """
        ACS for the scripted-gunner variant (`script 5 ENTER`, player 1, TID 1001). The scout (player 0) is learned.
        The gunner records the scout's path as breadcrumbs every `trail_spacing` units and walks along them with
        SetActorPosition steps (so it only crosses ground the scout crossed), stopping within `follow_dist` of the
        scout. Whenever a turret is alive, not invulnerable (i.e. the scout lights its room) and in the gunner's line
        of sight, the gunner turns to it and fires a hitscan bullet (`LineAttack`, credited to the gunner) every
        `refire_tics` tics. It never fires at invulnerable turrets.
        """
        target_checks = "\n".join(
            f"    if (SLAlive({tid}) && !GetActorProperty({tid}, APROP_Invulnerable) && CheckSight(1001, {tid}, 0)"
            f" && VectorLength((GetActorX({tid}) - GetActorX(1001)) >> 16, (GetActorY({tid}) - GetActorY(1001)) >> 16)"
            f" <= {fire_range}) return {tid};"
            for tid in enemy_tids
        )
        return f"""
int sg_tx[{trail_len}];
int sg_ty[{trail_len}];

function int SGTarget(void)
{{
{target_checks}
    return 0;
}}

script 5 ENTER
{{
    if (PlayerNumber() != 1) terminate;
    int head = 0;
    int tail = 0;
    int lx, ly, sx, sy, gx, gy, tx, ty, dx, dy, d, nx, ny, tgt, i;
    int ang = 16384;
    int fire_cd = 0;
    Delay(1);
    SetPlayerProperty(0, 1, PROP_FROZEN);
    lx = GetActorX(1000) >> 16;
    ly = GetActorY(1000) >> 16;
    sx = lx;
    sy = ly;
    while (TRUE)
    {{
        if (!SLAlive(1001)) terminate;
        if (SLAlive(1000))
        {{
            sx = GetActorX(1000) >> 16;
            sy = GetActorY(1000) >> 16;
            if (VectorLength(sx - lx, sy - ly) >= {trail_spacing})
            {{
                if (tail - head >= {trail_len}) head++;
                i = tail % {trail_len};
                sg_tx[i] = sx;
                sg_ty[i] = sy;
                tail++;
                lx = sx;
                ly = sy;
            }}
        }}
        gx = GetActorX(1001) >> 16;
        gy = GetActorY(1001) >> 16;
        if (head < tail && VectorLength(sx - gx, sy - gy) > {follow_dist})
        {{
            i = head % {trail_len};
            tx = sg_tx[i];
            ty = sg_ty[i];
            dx = tx - gx;
            dy = ty - gy;
            d = VectorLength(dx, dy);
            if (d <= {speed})
            {{
                nx = tx;
                ny = ty;
                head++;
            }}
            else
            {{
                nx = gx + (dx * {speed}) / d;
                ny = gy + (dy * {speed}) / d;
            }}
            SetActorPosition(1001, nx << 16, ny << 16, GetActorZ(1001), 0);
            SetActorVelocity(1001, 0, 0, 0, FALSE, FALSE);
            if (d > 0) ang = VectorAngle(dx << 16, dy << 16);
        }}
        tgt = SGTarget();
        if (tgt)
        {{
            ang = VectorAngle(GetActorX(tgt) - GetActorX(1001), GetActorY(tgt) - GetActorY(1001));
            SetActorAngle(1001, ang);
            if (fire_cd <= 0)
            {{
                LineAttack(1001, ang, 0, {damage}, "BulletPuff", "None", {fire_range}.0);
                fire_cd = {refire_tics};
            }}
        }}
        if (fire_cd > 0) fire_cd--;
        SetActorAngle(1001, ang);
        Delay(1);
    }}
}}
"""

    def generate(self, output_path: str) -> None:
        cfg = self.config
        seed = cfg["seed"] if cfg["seed"] is not None else random.randint(0, 999999)
        random.seed(seed)
        print(f"seed: {seed}")

        start_room_size = int(cfg["start_room_size"])
        hub_size = int(cfg["hub_size"])
        num_branches = int(cfg["num_branches"])
        rooms_per_branch = int(cfg["rooms_per_branch"])
        branch_room_width = int(cfg["branch_room_width"])
        branch_room_length = int(cfg["branch_room_length"])
        start_center_y = int(cfg["start_center_y"])
        branch_offset = int(cfg["branch_offset"])
        turret_wall_margin = int(cfg["turret_wall_margin"])
        first_room_turret_wall_margin = int(cfg["first_room_turret_wall_margin"])
        corridor_width = int(cfg["corridor_width"])
        min_enemies_per_room = int(cfg["min_enemies_per_room"])
        max_enemies_per_room = int(cfg["max_enemies_per_room"])
        turret_health = int(cfg["turret_health"])
        first_room_turret_health = int(cfg["first_room_turret_health"])
        turret_damage = int(cfg["turret_damage"])
        turret_projectile_speed = int(cfg["turret_projectile_speed"])
        turret_burst_count = int(cfg["turret_burst_count"])
        turret_refire_tics = int(cfg["turret_refire_tics"])
        always_lit = bool(cfg.get("always_lit", False))
        scripted_scout = bool(cfg.get("scripted_scout", False))
        scripted_gunner = bool(cfg.get("scripted_gunner", False))

        # Light levels used when the scout is present ("on") and absent ("off"). always_lit makes "off" == "on",
        # so the ACS structure (and invulnerability gating) is unchanged but no sector ever goes dark.
        room_on, corr_on, start_on, hub_on, start_corr_on = 224, 176, 224, 160, 160
        room_off = room_on if always_lit else 0
        corr_off = corr_on if always_lit else 0
        start_off = start_on if always_lit else 0
        hub_off = hub_on if always_lit else 0
        start_corr_off = start_corr_on if always_lit else 0

        start_center = (0, start_center_y)
        hub_center = (0, 0)
        branch_templates = [
            {"label": "North", "axis": "y", "sign": 1},
            {"label": "West", "axis": "x", "sign": -1},
            {"label": "East", "axis": "x", "sign": 1},
        ]
        active_branches = branch_templates[:num_branches]
        room_step = branch_room_length + 128
        furthest_branch_extent = branch_offset + (rooms_per_branch - 1) * room_step + branch_room_length
        bounds_half = max(
            abs(start_center_y) + start_room_size,
            furthest_branch_extent,
            branch_offset + branch_room_width,
        ) + 512

        builder = ProceduralMapBuilder(
            bounds=(-bounds_half, -bounds_half, bounds_half, bounds_half),
            num_seeds=16000,
            seed=seed,
        )

        missile_burst_lines = []
        for _ in range(turret_burst_count):
            missile_burst_lines.append('    TROO F 0 A_CustomMissile("SLTurretBolt", 32, 0, 0, 0, 0)')
            missile_burst_lines.append("    TROO F 2 Bright")
        decorate_str = f"""
ACTOR SLTurretBolt 16001 {{
  Radius 6
  Height 8
  Speed {turret_projectile_speed}
  Damage {turret_damage}
  Projectile
  +NOGRAVITY
  +BLOODLESSIMPACT
  RenderStyle Add
  Alpha 0.85
  States {{
  Spawn:
    BAL1 AB 4 Bright
    Loop
  Death:
    BAL1 CDE 4 Bright
    Stop
  }}
}}

ACTOR SLTurretGuard : DoomImp 16000 {{
  Health {turret_health}
  Speed 0
  Radius 24
  Height 56
  PainChance 0
  DropItem "None"
  +DONTTHRUST
  +LOOKALLAROUND
  +MISSILEMORE
  +MISSILEEVENMORE
  States {{
  Spawn:
    TROO AB 5 A_Look
    Loop
  See:
    TROO A 2 A_Chase
    Loop
  Missile:
    TROO E 0 A_FaceTarget
{os.linesep.join(missile_burst_lines)}
    TROO G {turret_refire_tics}
    Goto See
  Pain:
    TROO H 2
    Goto See
  Death:
    TROO I 8
    TROO J 8 A_Scream
    TROO K 6
    TROO L 6 A_NoBlocking
    TROO M 4
    TROO N 1 A_FadeOut(0.10)
    Wait
  }}
}}
"""
        builder.wad_writer.add_lump("DECORATE", decorate_str.encode("utf-8"))

        room_specs: List[Tuple[str, Tuple[int, int], int, int, int, str, str, str]] = [
            ("Start", start_center, 100, start_room_size, start_room_size, "FLOOR4_8", "CEIL3_5", "STONE2"),
            ("Hub", hub_center, 101, hub_size, hub_size, "FLOOR4_8", "CEIL3_5", "STONE2"),
        ]
        branch_specs: List[Dict[str, Any]] = []
        next_room_tag = 102
        for branch_idx, branch in enumerate(active_branches):
            room_names: List[str] = []
            room_tags: List[int] = []
            room_centers: List[Tuple[int, int]] = []
            room_dims: List[Tuple[int, int]] = []
            room_enemy_counts: List[int] = []

            for room_idx in range(rooms_per_branch):
                offset = branch_offset + room_idx * room_step
                if branch["axis"] == "y":
                    center = (0, branch["sign"] * offset)
                    dims = (branch_room_width, branch_room_length)
                else:
                    center = (branch["sign"] * offset, 0)
                    dims = (branch_room_length, branch_room_width)

                room_name = f'{branch["label"]} Room {room_idx + 1}'
                room_tag = next_room_tag
                next_room_tag += 1
                room_specs.append(
                    (room_name, center, room_tag, dims[0], dims[1], "FLOOR5_2", "CEIL3_5", "METAL1")
                )
                room_names.append(room_name)
                room_tags.append(room_tag)
                room_centers.append(center)
                room_dims.append(dims)
                if rooms_per_branch == 1:
                    room_enemy_counts.append(1)
                else:
                    room_enemy_counts.append(random.randint(min_enemies_per_room, max_enemies_per_room))

            branch_specs.append(
                {
                    "label": branch["label"],
                    "axis": branch["axis"],
                    "sign": branch["sign"],
                    "room_names": room_names,
                    "room_tags": room_tags,
                    "room_centers": room_centers,
                    "room_dims": room_dims,
                    "room_enemy_counts": room_enemy_counts,
                    "corr_tags": [],
                }
            )

        area_centers: Dict[str, Tuple[int, int]] = {}
        initial_light = {100: start_off, 101: hub_off}
        for name, (cx, cy), tag, width, height, floor_tex, ceil_tex, wall_tex in room_specs:
            area_centers[name] = (cx, cy)
            builder.add_area(
                name=name,
                shape=(cx, cy, width, height),
                floor_height=0,
                ceiling_height=192,
                floor_texture=floor_tex,
                ceiling_texture=ceil_tex,
                wall_texture=wall_tex,
                light_level=initial_light.get(tag, room_off),
                tag=tag,
            )

        corridor_specs = [
            ("Start", "Hub", 200),
        ]
        next_corr_tag = 201
        for branch in branch_specs:
            corridor_specs.append(("Hub", branch["room_names"][0], next_corr_tag))
            branch["corr_tags"].append(next_corr_tag)
            next_corr_tag += 1

            for room_idx in range(1, rooms_per_branch):
                corridor_specs.append(
                    (branch["room_names"][room_idx - 1], branch["room_names"][room_idx], next_corr_tag)
                )
                branch["corr_tags"].append(next_corr_tag)
                next_corr_tag += 1

        corridor_centers: Dict[int, Tuple[int, int]] = {}
        for room_a, room_b, tag in corridor_specs:
            corridor_name = builder.add_corridor(
                room_a,
                room_b,
                width=corridor_width,
                floor_texture="FLOOR4_8",
                ceiling_height=192,
                wall_texture="METAL1",
                light_level=start_corr_off if tag == 200 else corr_off,
            )
            builder.graph.get_area(corridor_name).config.tag = tag
            if tag == 200:
                start_corr_area = corridor_name
            ax, ay = area_centers[room_a]
            bx, by = area_centers[room_b]
            corridor_centers[tag] = ((ax + bx) // 2, (ay + by) // 2)

        builder.add_thing(ThingType.PLAYER1_START, -40, start_center[1] - 32, angle=90)
        builder.add_thing(ThingType.PLAYER2_START, 40, start_center[1] - 32, angle=90)
        builder.add_thing(ThingType.PLAYER3_START, -40, start_center[1] + 32, angle=90)
        builder.add_thing(ThingType.PLAYER4_START, 40, start_center[1] + 32, angle=90)

        # enemy_specs: List[Dict[str, Any]] = []
        total_enemy_count = 0
        next_enemy_tid = 3000
        next_spot_tid = 5000
        for branch_idx, branch in enumerate(branch_specs):
            room_checks = []
            for (cx, cy), (width, height) in zip(branch["room_centers"], branch["room_dims"]):
                room_checks.append(
                    self._rect_check("p1x", "p1y", cx, cy, width // 2 + 64, height // 2 + 64)
                )

            corr_checks = []
            for corr_tag in branch["corr_tags"]:
                corr_center = corridor_centers[corr_tag]
                corr_checks.append(self._square_check("p1x", "p1y", corr_center[0], corr_center[1], 240))

            room_enemy_specs: List[List[Dict[str, Any]]] = []
            for room_idx, ((cx, cy), (width, height), enemy_count) in enumerate(
                zip(branch["room_centers"], branch["room_dims"], branch["room_enemy_counts"])
            ):
                room_enemies: List[Dict[str, Any]] = []
                wall_margin = turret_wall_margin
                if room_idx == 0:
                    wall_margin = first_room_turret_wall_margin
                if branch["axis"] == "y":
                    enemy_positions = [
                        (cx, cy + branch["sign"] * (height // 2 - wall_margin)),
                        (cx - width // 4, cy + branch["sign"] * (height // 2 - wall_margin - 96)),
                    ]
                else:
                    enemy_positions = [
                        (cx + branch["sign"] * (width // 2 - wall_margin), cy),
                        (cx + branch["sign"] * (width // 2 - wall_margin - 96), cy - height // 4),
                    ]

                for enemy_slot in range(enemy_count):
                    spot_tid = next_spot_tid
                    tid = next_enemy_tid
                    next_spot_tid += 1
                    next_enemy_tid += 1
                    total_enemy_count += 1
                    pos = enemy_positions[min(enemy_slot, len(enemy_positions) - 1)]
                    builder.add_thing("MAP_SPOT", pos[0], pos[1], tid=spot_tid)
                    room_enemies.append(
                        {
                            "spot_tid": spot_tid,
                            "tid": tid,
                            "aim": (cx << 16, cy << 16),
                            "room_idx": room_idx,
                            "spawn_health": first_room_turret_health if room_idx == 0 else turret_health,
                        }
                    )

                room_enemy_specs.append(room_enemies)

            branch["room_checks"] = room_checks
            branch["corr_checks"] = corr_checks
            branch["room_enemy_specs"] = room_enemy_specs

        builder.add_thing(ThingType.TECH_LAMP, start_center[0] - 96, start_center[1] - 32)
        builder.add_thing(ThingType.TECH_LAMP, start_center[0] + 96, start_center[1] - 32)
        builder.add_thing(ThingType.TALL_RED_PILLAR, hub_center[0], hub_center[1])
        for branch in branch_specs:
            for room_idx, ((cx, cy), (width, height)) in enumerate(zip(branch["room_centers"], branch["room_dims"])):
                if branch["axis"] == "y":
                    lamp_a = (cx - min(120, width // 4), cy - min(220, height // 4))
                    lamp_b = (cx + min(120, width // 4), cy + min(60, height // 6))
                else:
                    lamp_a = (cx + min(220, width // 4), cy - min(120, height // 4))
                    lamp_b = (cx - min(60, width // 6), cy + min(120, height // 4))
                builder.add_thing(ThingType.FLOOR_LAMP, lamp_a[0], lamp_a[1])
                builder.add_thing(ThingType.FLOOR_LAMP, lamp_b[0], lamp_b[1])

        acs = ACSBuilder()
        acs.add_include("zcommon.acs")
        acs.add_global_var("sl_started", 1, "int")
        acs.add_global_var("sl_p1_alive", 41, "int")
        acs.add_global_var("sl_p2_alive", 42, "int")
        acs.add_global_var("sl_team_dist", 43, "int")
        acs.add_global_var("sl_enemies_left", 44, "int")
        acs.add_global_var("sl_target_lit", 45, "int")
        acs.add_global_var("sl_enemies_dead", 46, "int")
        acs.add_global_var("sl_total_enemies", 47, "int")
        acs.add_global_var("sl_rooms_seen", 48, "int")
        acs.add_global_var("sl_total_rooms", 49, "int")
        acs.add_global_var("sl_team_hp", 50, "int")
        acs.add_global_var("sl_targets_lit", 51, "int")
        if scripted_scout:
            acs.add_global_var("sl_act_room", 52, "int")

        enemy_state_init_lines = []
        branch_light_lines = []
        branch_logic_lines = []
        room_seen_init_lines = []
        target_lit_init_lines = []
        for branch_idx, branch in enumerate(branch_specs):
            lit_var = f"branch_attack_lit_{branch_idx}"
            active_room_var = f"branch_active_room_{branch_idx}"
            room_light_lines = []
            room_seen_case_lines = []
            room_pending_lines = []
            room_spawn_lines = []
            for room_idx, (room_tag, room_check) in enumerate(zip(branch["room_tags"], branch["room_checks"])):
                seen_var = f"branch_room_seen_{branch_idx}_{room_idx}"
                lit_seen_var = f"lit_seen_{branch_idx}_{room_idx}"
                room_seen_init_lines.append(f"    int {seen_var} = 0;\n")
                target_lit_init_lines.append(f"    int {lit_seen_var} = 0;\n")
                room_seen_case_lines.append(
                    f"""
            if (!{seen_var} && {active_room_var} == {room_idx} && sl_p1_alive && sl_p2_alive && ({room_check})) {{
                {seen_var} = 1;
                rooms_seen_count++;
            }}"""
                )
                room_var = f"branch_room_lit_{branch_idx}_{room_idx}"
                room_light_lines.append(
                    f"""
            int {room_var} = 0;
            if (sl_p1_alive && ({room_check})) {{
                {room_var} = 1;
            }}
            if ({room_var}) {{
                Light_ChangeToValue({room_tag}, {room_on});
            }} else {{
                Light_ChangeToValue({room_tag}, {room_off});
            }}"""
                )
            corridor_light_lines = []
            for corr_idx, (corr_tag, corr_check) in enumerate(zip(branch["corr_tags"], branch["corr_checks"])):
                corr_var = f"branch_corr_lit_{branch_idx}_{corr_idx}"
                corridor_light_lines.append(
                    f"""
            int {corr_var} = 0;
            if (sl_p1_alive && ({corr_check})) {{
                {corr_var} = 1;
            }}
            if ({corr_var}) {{
                Light_ChangeToValue({corr_tag}, {corr_on});
            }} else {{
                    Light_ChangeToValue({corr_tag}, {corr_off});
            }}"""
                )
            for room_idx, room_enemies in enumerate(branch["room_enemy_specs"]):
                room_pending_lines.append(f"\n            int branch_room_pending_{branch_idx}_{room_idx} = 0;")
                for enemy in room_enemies:
                    temp_tid = 7000 + enemy["tid"]
                    spawned_var = f"enemy_spawned_{enemy['tid']}"
                    dead_var = f"enemy_dead_{enemy['tid']}"
                    enemy_state_init_lines.append(f"    int {spawned_var} = 0;\n")
                    enemy_state_init_lines.append(f"    int {dead_var} = 0;\n")
                    room_pending_lines.append(
                        f"""
            if ({spawned_var} && !{dead_var}) {{
                if (ThingCount(T_NONE, {enemy["tid"]}) <= 0) {{
                    {dead_var} = 1;
                }} else if (GetActorProperty({enemy["tid"]}, APROP_Health) <= 0) {{
                    {dead_var} = 1;
                }}
            }}
            if (!{dead_var}) {{
                branch_room_pending_{branch_idx}_{room_idx} = 1;
            }}"""
                    )
                    room_spawn_lines.append(
                        f"""
            if ({active_room_var} == {room_idx} && !{spawned_var} && !{dead_var}) {{
                int spawn_ok_{enemy["tid"]} = SpawnSpotFacing("SLTurretGuard", {enemy["spot_tid"]}, {temp_tid});
                if (spawn_ok_{enemy["tid"]} && ThingCount(T_NONE, {temp_tid}) > 0) {{
                    Thing_ChangeTID({temp_tid}, {enemy["tid"]});
                    SetActorProperty({enemy["tid"]}, APROP_Health, {enemy["spawn_health"]});
                    SetActorProperty({enemy["tid"]}, APROP_Invulnerable, 1);
                    {spawned_var} = 1;
                }}
            }}"""
                    )
            branch_light_lines.append(
                f"""
            {"".join(room_light_lines)}
            {"".join(corridor_light_lines)}
            {"".join(room_pending_lines)}
            int {active_room_var} = -1;
"""
            )
            for room_idx, _room_enemies in enumerate(branch["room_enemy_specs"]):
                branch_light_lines.append(
                    f"""
            if ({active_room_var} < 0 && branch_room_pending_{branch_idx}_{room_idx}) {{
                {active_room_var} = {room_idx};
            }}"""
                )
            lit_case_lines = []
            for room_idx, (room_check, corr_check) in enumerate(zip(branch["room_checks"], branch["corr_checks"])):
                lit_seen_var = f"lit_seen_{branch_idx}_{room_idx}"
                lit_case_lines.append(
                    f"""
            if ({active_room_var} == {room_idx} && sl_p1_alive && (({room_check}) || ({corr_check}))) {{
                {lit_var} = 1;
                if (!{lit_seen_var} && sl_p2_alive) {{
                    {lit_seen_var} = 1;
                    targets_lit_count++;
                }}
            }}"""
                )
            branch_light_lines.append(
                f"""
            int {lit_var} = 0;
            {"".join(room_spawn_lines)}
            {"".join(room_seen_case_lines)}
            {"".join(lit_case_lines)}"""
            )
            room_logic_lines = []
            for room_idx, room_enemies in enumerate(branch["room_enemy_specs"]):
                for enemy in room_enemies:
                    spawned_var = f"enemy_spawned_{enemy['tid']}"
                    dead_var = f"enemy_dead_{enemy['tid']}"
                    room_logic_lines.append(
                        f"""
            if (!{dead_var}) {{
                enemies_left++;
            }}
            if ({spawned_var} && !{dead_var}) {{
                if (ThingCount(T_NONE, {enemy["tid"]}) <= 0) {{
                    {dead_var} = 1;
                }} else if (GetActorProperty({enemy["tid"]}, APROP_Health) <= 0) {{
                    {dead_var} = 1;
                }}
            }}
            if ({spawned_var} && !{dead_var}) {{
                if ({active_room_var} == {room_idx} && {lit_var}) {{
                    SetActorProperty({enemy["tid"]}, APROP_Invulnerable, 0);
                    sl_target_lit = 1;

                    target_tid = 0;
                    if (sl_p1_alive && sl_p2_alive) {{
                        int d1_{enemy["tid"]} = SLDist({enemy["aim"][0]}, {enemy["aim"][1]}, p1x, p1y);
                        int d2_{enemy["tid"]} = SLDist({enemy["aim"][0]}, {enemy["aim"][1]}, p2x, p2y);
                        if (d2_{enemy["tid"]} < d1_{enemy["tid"]}) {{
                            target_tid = 1001;
                        }} else {{
                            target_tid = 1000;
                        }}
                    }} else if (sl_p1_alive) {{
                        target_tid = 1000;
                    }} else if (sl_p2_alive) {{
                        target_tid = 1001;
                    }}

                    if (target_tid > 0) {{
                        Thing_Hate({enemy["tid"]}, target_tid, 6);
                    }}
                }} else {{
                    SetActorProperty({enemy["tid"]}, APROP_Invulnerable, 1);
                }}
            }}"""
                    )
            branch_logic_lines.append("".join(room_logic_lines))

        start_room_check = self._square_check(
            "p1x", "p1y", start_center[0], start_center[1], start_room_size // 2 + 72
        )
        hub_room_check = self._square_check(
            "p1x", "p1y", hub_center[0], hub_center[1], hub_size // 2 + 72
        )
        start_corr_check = self._square_check(
            "p1x", "p1y", corridor_centers[200][0], corridor_centers[200][1], 220
        )

        stage_mirror_code = ""
        scout_code = ""
        if scripted_scout:
            # Keep the corridor midpoint when the straight start->hub line is walkable (canonical ACS); otherwise
            # follow the bent corridor's cells so the scout cannot stall against its wall.
            start_corr_points = self._walkable_route(
                builder,
                [builder.name_to_id.get("Start", "Start"), start_corr_area, builder.name_to_id.get("Hub", "Hub")],
                start_center,
                hub_center,
            ) or [corridor_centers[200]]
            stage_mirror_code, scout_code = self._scripted_scout_acs(
                branch_specs,
                rooms_per_branch,
                start_center,
                start_corr_points,
                hub_center,
                hold_depth=int(cfg["scout_hold_depth"]),
                speed=int(cfg["scout_speed"]),
                strafe=int(cfg["scout_strafe"]),
                strafe_period=int(cfg["scout_strafe_period"]),
            )

        script_setup = """
        int pn = PlayerNumber();
        int tid = 1000 + pn;
        Thing_ChangeTID(0, tid);

        SetActorProperty(0, APROP_Health, 100);
        SetActorProperty(0, APROP_SpawnHealth, 100);
        SetActorProperty(0, APROP_Speed, 1.0);
        ClearInventory();

        if (pn == 0) {
            sl_started = 0;
            sl_p1_alive = 0;
            sl_p2_alive = 0;
            sl_team_dist = 0;
            sl_enemies_left = __ENEMY_COUNT__;
            sl_target_lit = 0;
            sl_enemies_dead = 0;
            sl_total_enemies = __ENEMY_COUNT__;
            sl_rooms_seen = 0;
            sl_total_rooms = __ROOM_COUNT__;
            sl_team_hp = 200;
            sl_targets_lit = 0;

            TakeInventory("Fist", 999);
            TakeInventory("Chainsaw", 999);
            TakeInventory("Pistol", 999);
            TakeInventory("Shotgun", 999);
            TakeInventory("SuperShotgun", 999);
            TakeInventory("Chaingun", 999);
            TakeInventory("RocketLauncher", 999);
            TakeInventory("PlasmaRifle", 999);
            TakeInventory("BFG9000", 999);
            TakeInventory("Clip", 9999);
            TakeInventory("Shell", 9999);
            TakeInventory("RocketAmmo", 9999);
            TakeInventory("Cell", 9999);
        } else if (pn == 1) {
            SetAmmoCapacity("Clip", 10000);
            GiveInventory("Chaingun", 1);
            GiveInventory("Clip", 240);
            SetWeapon("Chaingun");
        }

        ACS_ExecuteAlways(10, 0);
        """
        script_setup = script_setup.replace("__ENEMY_COUNT__", str(total_enemy_count))
        script_setup = script_setup.replace("__ROOM_COUNT__", str(num_branches * rooms_per_branch))
        acs.add_script(ScriptType.ENTER, script_setup, number=1)
        acs.add_script(ScriptType.RESPAWN, script_setup, number=3)

        script_open = f"""
#define TOTAL_ENEMIES __ENEMY_COUNT__
#define TOTAL_ROOMS __ROOM_COUNT__

function int SLAlive(int tid)
{{
    if (tid <= 0) return 0;
    if (ThingCount(T_NONE, tid) <= 0) return 0;
    return GetActorProperty(tid, APROP_Health) > 0;
}}

function int SLDist(int ax, int ay, int bx, int by)
{{
    int dx = ax - bx;
    int dy = ay - by;
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;
    if (dy > dx) dx = dy;
    return dx >> 16;
}}

script 2 OPEN
{{
    int p1x, p1y, p2x, p2y;
    int p1hp, p2hp;
    int enemies_left;
    int target_tid;
    int rooms_seen_count = 0;
    int targets_lit_count = 0;
    int startup_ready_loops = 0;
{"".join(room_seen_init_lines)}
{"".join(target_lit_init_lines)}
{"".join(enemy_state_init_lines)}

    sl_started = 0;
    sl_total_enemies = TOTAL_ENEMIES;
    sl_total_rooms = TOTAL_ROOMS;
    sl_p1_alive = 0;
    sl_p2_alive = 0;
    sl_team_dist = 0;
    sl_enemies_left = TOTAL_ENEMIES;
    sl_target_lit = 0;
    sl_enemies_dead = 0;
    sl_rooms_seen = 0;
    sl_team_hp = 200;
    sl_targets_lit = 0;

    while (TRUE)
    {{
        sl_p1_alive = SLAlive(1000);
        sl_p2_alive = SLAlive(1001);

        if (sl_p1_alive)
        {{
            p1x = GetActorX(1000);
            p1y = GetActorY(1000);
            TakeActorInventory(1000, "Fist", 999);
            TakeActorInventory(1000, "Chainsaw", 999);
            TakeActorInventory(1000, "Pistol", 999);
            TakeActorInventory(1000, "Shotgun", 999);
            TakeActorInventory(1000, "SuperShotgun", 999);
            TakeActorInventory(1000, "Chaingun", 999);
            TakeActorInventory(1000, "RocketLauncher", 999);
            TakeActorInventory(1000, "PlasmaRifle", 999);
            TakeActorInventory(1000, "BFG9000", 999);
            TakeActorInventory(1000, "Clip", 9999);
            TakeActorInventory(1000, "Shell", 9999);
            TakeActorInventory(1000, "RocketAmmo", 9999);
            TakeActorInventory(1000, "Cell", 9999);
        }}
        else
        {{
            p1x = 0;
            p1y = 0;
        }}

        if (sl_p2_alive)
        {{
            p2x = GetActorX(1001);
            p2y = GetActorY(1001);
        }}
        else
        {{
            p2x = 0;
            p2y = 0;
        }}

        if (!sl_started)
        {{
            if (sl_p1_alive && sl_p2_alive)
            {{
                startup_ready_loops++;
                if (startup_ready_loops >= 4)
                {{
                    sl_started = 1;
                }}
            }}
            else
            {{
                startup_ready_loops = 0;
            }}
        }}

        if (sl_p1_alive && sl_p2_alive)
        {{
            sl_team_dist = SLDist(p1x, p1y, p2x, p2y);
        }}
        else
        {{
            sl_team_dist = 0;
        }}

        p1hp = 0;
        p2hp = 0;
        if (sl_p1_alive)
        {{
            p1hp = GetActorProperty(1000, APROP_Health);
            if (p1hp < 0) p1hp = 0;
        }}
        if (sl_p2_alive)
        {{
            p2hp = GetActorProperty(1001, APROP_Health);
            if (p2hp < 0) p2hp = 0;
        }}
        sl_team_hp = p1hp + p2hp;

        if (sl_p1_alive && (({start_room_check}) || ({start_corr_check})))
        {{
            Light_ChangeToValue(100, {start_on});
        }}
        else
        {{
            Light_ChangeToValue(100, {start_off});
        }}

        if (sl_p1_alive && ({hub_room_check}))
        {{
            Light_ChangeToValue(101, {hub_on});
        }}
        else
        {{
            Light_ChangeToValue(101, {hub_off});
        }}

        if (sl_p1_alive && ({start_corr_check}))
        {{
            Light_ChangeToValue(200, {start_corr_on});
        }}
        else
        {{
            Light_ChangeToValue(200, {start_corr_off});
        }}

        sl_target_lit = 0;
        {"".join(branch_light_lines)}
        {stage_mirror_code}
        sl_rooms_seen = rooms_seen_count;
        sl_targets_lit = targets_lit_count;

        enemies_left = 0;
        {"".join(branch_logic_lines)}
        sl_enemies_left = enemies_left;
        sl_enemies_dead = TOTAL_ENEMIES - enemies_left;

        if (sl_started && (!sl_p1_alive || !sl_p2_alive))
        {{
            Delay(1);
            Exit_Normal(0);
        }}
        if (sl_started && sl_enemies_dead >= TOTAL_ENEMIES)
        {{
            Delay(1);
            Exit_Normal(0);
        }}

        Delay(2);
    }}
}}
"""
        script_open = script_open.replace("__ENEMY_COUNT__", str(total_enemy_count))
        script_open = script_open.replace("__ROOM_COUNT__", str(num_branches * rooms_per_branch))
        acs.add_global_code(script_open)
        if scout_code:
            acs.add_global_code(scout_code)
        if scripted_gunner:
            enemy_tids = [
                enemy["tid"]
                for branch in branch_specs
                for room_enemies in branch["room_enemy_specs"]
                for enemy in room_enemies
            ]
            acs.add_global_code(
                self._scripted_gunner_acs(
                    enemy_tids,
                    speed=int(cfg["gunner_speed"]),
                    follow_dist=int(cfg["gunner_follow_dist"]),
                    trail_spacing=int(cfg["gunner_trail_spacing"]),
                    damage=int(cfg["gunner_damage"]),
                    refire_tics=int(cfg["gunner_refire_tics"]),
                    fire_range=int(cfg["gunner_range"]),
                )
            )

        script_hud = """
        int self_tid = ActivatorTID();
        while (TRUE) {
            if (self_tid > 0 && ThingCount(T_NONE, self_tid) <= 0) {
                break;
            }

            SetHudSize(320, 200, 1);
            HudMessage(
                d:sl_enemies_left;
                HUDMSG_PLAIN, 30, CR_WHITE, 160.0, 12.0, 0.0
            );
            if (sl_target_lit) {
                HudMessage(
                    s:"LIT";
                    HUDMSG_PLAIN, 31, CR_GOLD, 160.0, 24.0, 0.0
                );
            } else {
                HudMessage(
                    s:"DARK";
                    HUDMSG_PLAIN, 31, CR_WHITE, 160.0, 24.0, 0.0
                );
            }
            Delay(1);
        }
        """
        acs.add_script(ScriptType.VOID, script_hud, number=10)

        builder.map_data.scripts = acs.to_code()
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
    path = os.path.join(out_dir, "stealth_labyrinth.wad")
    print(f"Generating {path}...")
    StealthLabyrinthScenario().generate(path)
    print("Done.")
