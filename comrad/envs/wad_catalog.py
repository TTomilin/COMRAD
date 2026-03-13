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
        self._rr = 0
        self._weights: Optional[List[float]] = None

    @classmethod
    def from_dir(cls, batch_dir: str) -> "WadBatch":
        idx = os.path.join(batch_dir, BATCH_INDEX)
        if os.path.isfile(idx):
            with open(idx) as f:
                data = json.load(f)
            entries = []
            for e in data:
                wad_path = os.path.abspath(os.path.join(batch_dir, e["filename"]))
                entries.append(WadInfo(e["id"], wad_path, e.get("config", {})))
            return cls(entries)
        wads = sorted(glob.glob(os.path.join(batch_dir, "*.wad")))
        if not wads:
            raise FileNotFoundError(f"No WADs found in {batch_dir}")
        entries = [
            WadInfo(os.path.splitext(os.path.basename(w))[0], os.path.abspath(w))
            for w in wads
        ]
        return cls(entries)

    def sample(self, strategy: str = "round_robin", rng=None) -> WadInfo:
        if not self.entries:
            raise RuntimeError("WadBatch is empty")
        if strategy == "round_robin":
            e = self.entries[self._rr % len(self.entries)]
            self._rr += 1
            return e
        if strategy == "random":
            return (rng or random).choice(self.entries)
        if strategy == "weighted":
            w = self._weights or [1.0] * len(self.entries)
            return (rng or random).choices(self.entries, weights=w, k=1)[0]
        raise ValueError(f"Unknown sampling strategy: {strategy!r}")

    def set_weights(self, weights: Dict[str, float]) -> None:
        self._weights = [weights.get(e.name, 1.0) for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)
