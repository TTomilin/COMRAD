import glob
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WadInfo:
    name: str
    wad_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


BATCH_INDEX = "batch_registry.json"


class WadBatch:
    def __init__(self, entries: List[WadInfo], name: str = ""):
        self.entries = entries
        self.name = name

    @classmethod
    def from_dir(cls, batch_dir: str) -> "WadBatch":
        idx = os.path.join(batch_dir, BATCH_INDEX)
        if os.path.isfile(idx):
            with open(idx) as f:
                data = json.load(f)
            entries = []
            for e in data:
                wad_path = os.path.abspath(os.path.join(batch_dir, e["filename"]))
                metadata = {key: value for key, value in e.items() if key not in {"id", "filename"}}
                entries.append(WadInfo(e["id"], wad_path, metadata))
            return cls(entries)
        wads = sorted(glob.glob(os.path.join(batch_dir, "*.wad")))
        if not wads:
            raise FileNotFoundError(f"No WADs found in {batch_dir}")
        entries = [
            WadInfo(os.path.splitext(os.path.basename(w))[0], os.path.abspath(w))
            for w in wads
        ]
        return cls(entries)

    def __len__(self) -> int:
        return len(self.entries)
