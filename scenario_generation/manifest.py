from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WadManifestEntry:
    scenario: str
    wad_id: str
    seed: int
    filename: str
    sha256: str
    config: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(batch_dir: Path) -> list[WadManifestEntry]:
    manifest_path = batch_dir / "batch_registry.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [WadManifestEntry(**item) for item in payload]
    for entry in entries:
        wad_path = batch_dir / entry.filename
        actual_sha = sha256_file(wad_path)
        if actual_sha != entry.sha256:
            raise RuntimeError(f"WAD hash mismatch for {wad_path}: {actual_sha} != {entry.sha256}")
    return entries
