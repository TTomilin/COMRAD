"""
Build the Stealth Labyrinth learnability-reference WADs (pinned seed) so they differ from the deployed benchmark WAD
only in the intended ACS/lighting change.

Procedural geometry lumps (VERTEXES/LINEDEFS/NODES/...) are not bit-reproducible across rebuilds, while the ACS
(BEHAVIOR) and SECTORS of a seed-42 rebuild match the deployed WAD exactly. Variants are therefore produced by
transplanting only BEHAVIOR/SCRIPTS (and SECTORS light levels for `always_lit`) into a copy of the deployed WAD.

Run from DoomGen/:  uv run python examples/benchmark/build_stealth_labyrinth_variants.py [--out-dir DIR] [--deploy]
"""
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

import omg

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stealth_labyrinth import StealthLabyrinthScenario  # noqa: E402

COMRAD_SCENARIOS = Path(__file__).resolve().parents[3] / "comrad" / "scenarios"
VARIANTS = {
    "stealth_labyrinth_canonical_rebuild": {},
    "stealth_labyrinth_lit": {"always_lit": True},
    "stealth_labyrinth_scripted_scout": {"scripted_scout": True},
    "stealth_labyrinth_scripted_gunner": {"scripted_gunner": True},
}


def lump_digests(path):
    wad = omg.WAD(str(path))
    out = {}
    for name, lump in wad.maps["MAP01"].items():
        out[name] = hashlib.sha1(lump.data).hexdigest()[:12]
    return out


SECTOR_FIELDS = ("z_floor", "z_ceil", "tx_floor", "tx_ceil", "type", "tag")


def check_sectors_compatible(deployed_path, built_path, allow_light_change):
    ref = omg.MapEditor(omg.WAD(str(deployed_path)).maps["MAP01"]).sectors
    new = omg.MapEditor(omg.WAD(str(built_path)).maps["MAP01"]).sectors
    if len(ref) != len(new):
        raise SystemExit(f"sector count mismatch: {len(ref)} vs {len(new)}")
    light_changes = 0
    for a, b in zip(ref, new):
        for field in SECTOR_FIELDS:
            if getattr(a, field) != getattr(b, field):
                raise SystemExit(f"sector field {field} differs: {getattr(a, field)} vs {getattr(b, field)}")
        if a.light != b.light:
            light_changes += 1
    if light_changes and not allow_light_change:
        raise SystemExit("unexpected sector light change")
    return light_changes


def transplant(deployed_path, built_path, lumps, dst_path):
    wad = omg.WAD(str(deployed_path))
    built = omg.WAD(str(built_path))
    for lump in lumps:
        wad.maps["MAP01"][lump] = built.maps["MAP01"][lump]
    if Path(dst_path).exists():
        os.remove(dst_path)
    wad.to_file(str(dst_path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="/tmp/stealth_variants")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deploy", action="store_true", help="copy the variant WADs into comrad/scenarios/")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    built = {}
    for name, overrides in VARIANTS.items():
        path = out_dir / f"{name}.wad"
        if path.exists():
            os.remove(path)  # omg overwrite can raise PermissionError on existing targets
        StealthLabyrinthScenario({"seed": args.seed, **overrides}).generate(str(path))
        built[name] = path

    deployed = COMRAD_SCENARIOS / "stealth_labyrinth.wad"
    ref = lump_digests(deployed)
    for name, path in built.items():
        digests = lump_digests(path)
        diff = sorted(k for k in set(ref) | set(digests) if ref.get(k) != digests.get(k))
        print(f"{name}: lumps differing from deployed stealth_labyrinth.wad: {diff or 'none'}")

    check_sectors_compatible(deployed, built["stealth_labyrinth_canonical_rebuild"], allow_light_change=False)
    check_sectors_compatible(deployed, built["stealth_labyrinth_scripted_scout"], allow_light_change=False)
    check_sectors_compatible(deployed, built["stealth_labyrinth_scripted_gunner"], allow_light_change=False)
    lit_changes = check_sectors_compatible(deployed, built["stealth_labyrinth_lit"], allow_light_change=True)
    print(f"always_lit changes light of {lit_changes} sectors; all other sector fields identical")

    plan = {
        "stealth_labyrinth_lit": ["BEHAVIOR", "SCRIPTS", "SECTORS"],
        "stealth_labyrinth_scripted_scout": ["BEHAVIOR", "SCRIPTS"],
        "stealth_labyrinth_scripted_gunner": ["BEHAVIOR", "SCRIPTS"],
    }
    for name, lumps in plan.items():
        dst = out_dir / f"{name}_final.wad"
        transplant(deployed, built[name], lumps, dst)
        digests, ref_digests = lump_digests(dst), ref
        diff = sorted(k for k in set(ref_digests) | set(digests) if ref_digests.get(k) != digests.get(k))
        print(f"{name}_final: lumps differing from deployed: {diff}")
        if args.deploy:
            target = COMRAD_SCENARIOS / f"{name}.wad"
            shutil.copyfile(dst, target)
            print(f"deployed {target}")


if __name__ == "__main__":
    main()
