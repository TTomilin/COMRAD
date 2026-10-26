import tomllib
from pathlib import Path

from comrad.utils.doom_utils import DOOM_ENVS

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_manifest_entries_point_to_existing_paths():
    # Regression: MANIFEST.in used to reference the non-existent comrad/docs and comrad/experiments.
    for line in (REPO_ROOT / "MANIFEST.in").read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "include":
            assert (REPO_ROOT / parts[1]).is_file(), line
        elif parts[0] == "recursive-include":
            assert (REPO_ROOT / parts[1]).is_dir(), line


def test_every_scenario_ships_cfg_and_wad():
    scenarios = REPO_ROOT / "comrad" / "scenarios"
    for spec in DOOM_ENVS:
        cfg = scenarios / spec.env_spec_file
        assert cfg.is_file(), cfg
        wad_lines = [line for line in cfg.read_text().splitlines() if line.strip().startswith("doom_scenario_path")]
        for line in wad_lines:
            wad = line.split("=", 1)[1].strip()
            assert (scenarios / wad).is_file(), f"{spec.name}: missing {wad}"


def test_comrad_package_is_not_tied_to_doomgen():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert "doomgen" not in pyproject["project"].get("optional-dependencies", {})
    assert "workspace" not in pyproject.get("tool", {}).get("uv", {})
    assert not any("doomgen" in dep for dep in pyproject["project"]["dependencies"])
