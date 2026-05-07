# `comrad/play/`

This directory contains local standalone play assets for ViZDoom-based manual inspection.

## Contents

- `play.sh`: Unix launcher helper
- `play.exe`: Windows launcher helper
- bundled `freedoom2.wad`, `vizdoom`, `vizdoom.pk3`, and config files needed for quick local play

## Usage

Windows:

```bash
play.exe
```

macOS or Linux:

```bash
./play.sh
```

To run ViZDoom directly:

```bash
./vizdoom -windowed 1 -iwad freedoom2.wad
```

To join a multiplayer session manually, pass the appropriate `-file` and `-join` arguments to the ViZDoom binary.
