import os
import re


def patch_wad_path(base_cfg: str, wad_path: str, out_cfg: str) -> None:
    """Copy base_cfg to out_cfg, replacing the doom_scenario_path line."""
    abs_wad = os.path.abspath(wad_path)
    cfg_dir = os.path.dirname(os.path.abspath(out_cfg))
    rel_wad = os.path.relpath(abs_wad, cfg_dir)

    with open(base_cfg) as f:
        lines = f.readlines()
    patched, found = [], False
    for line in lines:
        if re.match(r"\s*doom_scenario_path\s*=", line):
            patched.append(f"doom_scenario_path = {rel_wad}\n")
            found = True
        else:
            patched.append(line)
    if not found:
        patched.append(f"doom_scenario_path = {rel_wad}\n")
    with open(out_cfg, "w") as f:
        f.writelines(patched)
