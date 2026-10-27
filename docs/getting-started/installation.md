# Installation

```bash
pip install comrad-bench
```

COMRAD needs **Python 3.11 or 3.12** on Linux or macOS. The ViZDoom engine comes with the `vizdoom` wheel, so nothing else has to be installed on common platforms. If `pip` has to build ViZDoom from source, follow the [ViZDoom build instructions](https://vizdoom.farama.org/dev/building_from_source/) first.

Check the install:

```bash
python -c "import comrad; print([s.name for s in comrad.list_scenarios()])"
```

!!! note "Network ports"
    Every multi-agent scenario runs one Doom process per agent, and the processes talk over local UDP. Each environment reserves ports starting at `40300`; set `DOOM_DEFAULT_UDP_PORT` to move that base if it clashes with something else. Sandboxes that block UDP binding will fail at `reset()`.

## Development install

To change the code, run the tests, or train with the bundled Sample Factory, clone the repository and use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/TTomilin/COMRAD.git
cd COMRAD
uv sync
uv run pytest comrad/tests
```

See [Contributing](../contributing.md) for the repository layout.
