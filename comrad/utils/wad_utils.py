import hashlib
import os
import re
import shutil


def _staged_wad_name(abs_wad: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(abs_wad))
    digest = hashlib.sha1(abs_wad.encode("utf-8")).hexdigest()[:12]
    return f"{stem}__{digest}{ext}"


def _stage_wad_for_cfg(abs_wad: str, cfg_dir: str) -> str:
    staged_wad = os.path.join(cfg_dir, _staged_wad_name(abs_wad))

    if os.path.abspath(staged_wad) == abs_wad:
        return staged_wad

    if os.path.lexists(staged_wad):
        try:
            same_target = os.path.samefile(staged_wad, abs_wad)
        except OSError:
            same_target = False
        if not same_target:
            os.remove(staged_wad)

    if not os.path.exists(staged_wad):
        try:
            os.symlink(abs_wad, staged_wad)
        except OSError:
            shutil.copy2(abs_wad, staged_wad)

    return staged_wad


def patch_wad_path(base_cfg: str, wad_path: str, out_cfg: str) -> None:
    """Copy base_cfg to out_cfg, replacing the doom_scenario_path line."""
    abs_wad = os.path.abspath(wad_path)
    cfg_dir = os.path.dirname(os.path.abspath(out_cfg))
    staged_wad = _stage_wad_for_cfg(abs_wad, cfg_dir)
    wad_ref = os.path.basename(staged_wad)

    with open(base_cfg) as f:
        lines = f.readlines()
    patched, found = [], False
    for line in lines:
        if re.match(r"\s*doom_scenario_path\s*=", line):
            patched.append(f"doom_scenario_path = {wad_ref}\n")
            found = True
        else:
            patched.append(line)
    if not found:
        patched.append(f"doom_scenario_path = {wad_ref}\n")
    with open(out_cfg, "w") as f:
        f.writelines(patched)
