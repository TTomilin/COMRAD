# Contributing

```bash
git clone https://github.com/TTomilin/COMRAD.git && cd COMRAD
uv sync
uv run pre-commit install
uv run pytest comrad/tests sample_factory/tests
```

## Repository layout

| Path | Contents |
|---|---|
| `comrad/` | The `comrad-bench` package: public API (`api.py`), environments, scenario wrappers, models, training entry points |
| `comrad/scenarios/` | Shipped `.cfg`/`.wad` files and `batch_*` map pools |
| `sample_factory/` | Modified Sample Factory with the MARL algorithms (shipped in the package) |
| `examples/` | Runnable quickstart scripts used in the docs |
| `scripts/` | Figure, table and analysis scripts |
| `results/` | Committed figures, tables and result summaries |
| `docs/`, `mkdocs.yml` | This website |
| `DoomGen/`, `DoomGen2/` | Separate map-generation tools. They are **not** part of the COMRAD package and have their own environment and tests. |

## Adding a scenario

1. Put the `.cfg` and `.wad` files in `comrad/scenarios/`.
2. Add a `DoomSpec` to `DOOM_ENVS` and a one-line entry to `SCENARIO_METADATA` in `comrad/utils/doom_utils.py`.
3. Put reward shaping in a wrapper under `comrad/wrappers/scenario_wrappers/`, and set `info["true_objective"]` at episode end.
4. Add a page under `docs/scenarios/` and link it in `mkdocs.yml`.

## Documentation

```bash
uv run --group docs mkdocs serve          # live preview
uv run --group docs mkdocs build --strict # what CI runs
./scripts/build_docs_assets.sh            # regenerate thumbnails, figures and the GIF
```

Pushing changes under `docs/`, `mkdocs.yml` or `README.md` to `main` redeploys the site.
