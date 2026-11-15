"""Version of the kd6J checkpoint-pack evaluation protocol (`eval_kd6j_checkpoint_pack.py`).

v2: every pass (normal, partner occlusion, zero-shot blind) starts from its own `env.reset()` after its intervention is
set, episodes hit by a ViZDoom crash recovery are discarded, and the blind clock uses the policy's observation clock.
Held-out JSONs without this version were produced by the earlier protocol and are not reused or aggregated.
Kept free of torch/ViZDoom imports so SLURM scripts can check results cheaply.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

KD6J_PACK_PROTOCOL_VERSION = "kd6j-pack-v2"


def is_current_result(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("protocol_version") == KD6J_PACK_PROTOCOL_VERSION


def is_current_result_file(path: Path) -> bool:
    """True iff `path` is a complete JSON written by the current protocol."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return is_current_result(payload)


if __name__ == "__main__":
    sys.exit(0 if is_current_result_file(Path(sys.argv[1])) else 1)
