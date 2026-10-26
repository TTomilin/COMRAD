# `comrad/tests/`

Tests for COMRAD-specific behaviour: public API and PettingZoo adapter, explicit reset, packaging, launcher profiles, recorders, and wrapper invariants.

```bash
uv run pytest comrad/tests -v
uv run pytest sample_factory/tests -v   # engine tests
```

Run DoomGen tests separately from `DoomGen/`, because running both suites from the repository root causes a `tests.*` import collision.
