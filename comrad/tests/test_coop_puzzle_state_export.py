"""Co-op Puzzle exports its hidden puzzle state (frontier plates, lane, sync timer, gates) as USER7-USER14."""

import random
import re
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOOMGEN_ROOT = REPO_ROOT / "DoomGen"
for _path in (DOOMGEN_ROOT, DOOMGEN_ROOT / "src", DOOMGEN_ROOT / "examples" / "benchmark"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from examples.benchmark.coop_puzzle import CoopPuzzleScenario  # noqa: E402

SCENARIOS_DIR = REPO_ROOT / "comrad" / "scenarios"
NEW_GLOBALS = {
    7: "frontier",
    8: "plate1_x",
    9: "plate1_y",
    10: "plate2_x",
    11: "plate2_y",
    12: "plate1_lane",
    13: "sync_time",
    14: "gate_mask",
}
HELDOUT_SEED = 2027


def _cfg_game_variables() -> list[str]:
    text = (SCENARIOS_DIR / "coop_puzzle.cfg").read_text()
    match = re.search(r"^available_game_variables\s*=\s*\{(.*)\}", text, re.MULTILINE)
    assert match is not None
    return match.group(1).split()


def _plate_function_entries(code: str, name: str) -> dict[tuple[int, int | None], int]:
    body = code.split(f"function int {name}(int tag, int lane)")[1].split("return 0;")[0]
    entries = re.findall(r"tag == (\d+)(?: && lane == (\d))?\) return (-?\d+);", body)
    return {(int(tag), int(lane) if lane else None): int(value) for tag, lane, value in entries}


def _wad_lump(path: Path, lump: str) -> bytes:
    data = path.read_bytes()
    count, offset = struct.unpack_from("<ii", data, 4)
    for i in range(count):
        pos, size, raw = struct.unpack_from("<ii8s", data, offset + 16 * i)
        if raw.rstrip(b"\0").decode() == lump:
            return data[pos : pos + size]
    raise KeyError(lump)


@pytest.fixture(scope="module")
def heldout_components():
    # same code path as scripts/heldout_wad_evaluation.py::generate_heldout_batch (scenario cls with {"seed": s})
    scenario = CoopPuzzleScenario({"seed": HELDOUT_SEED}, name=f"coop_puzzle_heldout_seed_{HELDOUT_SEED}")
    random.seed(HELDOUT_SEED)
    _, acs, metadata = scenario._build_components(HELDOUT_SEED)
    return scenario, acs.to_code(), metadata


def test_cfg_appends_new_user_variables_after_user6():
    variables = _cfg_game_variables()
    assert variables[:9] == ["POSITION_X", "POSITION_Y", "USER1", "USER2", "USER3", "USER4", "USER5", "USER6", "DEAD"]
    assert variables[9:] == [f"USER{i}" for i in NEW_GLOBALS]


def test_generated_acs_declares_and_updates_exports(heldout_components):
    _, code, _ = heldout_components
    for slot, name in NEW_GLOBALS.items():
        assert f"global int {slot}:{name};" in code
    assert "global int 6:zone_limit;" in code
    assert "function void UpdateStateExport(void)" in code
    # updated every tic in the main loop, after the gameplay state of this tic is final and before the exit check
    main = code.split("plate_state = plate;")[1]
    assert main.lstrip().startswith("UpdateStateExport();")
    export = code.split("function void UpdateStateExport(void)")[1].split("script ")[0]
    for forbidden in ("Random(", "Delay(", "Door_Open", "Door_Close", "ChangeFloor", "sync_timer[f]--", "zone ="):
        assert forbidden not in export


def test_plate_coordinate_functions_cover_every_candidate_plate(heldout_components):
    scenario, code, metadata = heldout_components
    plate_x = _plate_function_entries(code, "PlateX")
    plate_y = _plate_function_entries(code, "PlateY")
    assert plate_x.keys() == plate_y.keys()

    expected = []
    for zone in metadata["zones"].values():
        for lane in ("A", "B"):
            expected.extend((plate, None) for plate in zone["type1"][lane])
    for sync in metadata["sync_zones"].values():
        for lane_idx, lane in enumerate(("A", "B")):
            expected.append((sync["plates"][lane], lane_idx))
    num_zones = int(scenario.config["num_zones"])
    assert len(metadata["sync_zones"]) > 0
    assert len(expected) == len(plate_x)

    for plate, lane_idx in expected:
        key = (plate["tag"], lane_idx)
        assert key in plate_x, key
        half = plate["size"] / 2 + 1
        # exported centre comes from the sector's cells and must lie inside the requested plate square
        assert abs(plate_x[key] - plate["x"]) <= half, (key, plate_x[key], plate["x"])
        assert abs(plate_y[key] - plate["y"]) <= half, (key, plate_y[key], plate["y"])
        assert tuple(plate["center"]) == (plate_x[key], plate_y[key])
    assert all(0 < tag < 1000 * (num_zones + 1) for tag, _ in plate_x)


def test_installed_wad_ships_state_export_scripts():
    scripts = _wad_lump(SCENARIOS_DIR / "coop_puzzle.wad", "SCRIPTS").decode()
    for slot, name in NEW_GLOBALS.items():
        assert f"global int {slot}:{name};" in scripts
    assert "UpdateStateExport();" in scripts
    assert len(_wad_lump(SCENARIOS_DIR / "coop_puzzle.wad", "BEHAVIOR")) > 16
