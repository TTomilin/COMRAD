"""Paths shared by the COMRAD scenario-generation scripts."""

from __future__ import annotations

from pathlib import Path

SCENARIO_GENERATION_ROOT = Path(__file__).resolve().parent
COMRAD_ROOT = SCENARIO_GENERATION_ROOT.parent
RUNTIME_SCENARIOS = COMRAD_ROOT / "comrad" / "scenarios"

__all__ = ["COMRAD_ROOT", "RUNTIME_SCENARIOS", "SCENARIO_GENERATION_ROOT"]
