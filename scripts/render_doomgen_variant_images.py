from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from omg import MapEditor, WAD


REPO_ROOT = Path(__file__).resolve().parents[1]
WAD2IMAGE_SCRIPT = REPO_ROOT / "scripts" / "wad2image" / "bin" / "wad2image.py"
DEFAULT_IWAD = REPO_ROOT / "DoomGen" / "examples" / "play" / "freedoom2.wad"
DEFAULT_VARIANTS_DIR = REPO_ROOT / "results" / "variants"
DEFAULT_OUTPUT_DIR = DEFAULT_VARIANTS_DIR / "images"
DEFAULT_REGISTRY_PATH = DEFAULT_VARIANTS_DIR / "variants_registry.json"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render DoomGen variant WADs into readable top-down images using wad2image plus preview overlays."
    )
    parser.add_argument(
        "--variants-dir",
        type=Path,
        default=DEFAULT_VARIANTS_DIR,
        help="Directory containing variant WAD files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where rendered PNG files will be written.",
    )
    parser.add_argument(
        "--iwad",
        type=Path,
        default=DEFAULT_IWAD,
        help="IWAD used by wad2image and for texture/sprite extraction.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="JSON registry mapping variant filenames to generation configs.",
    )
    parser.add_argument("--width", type=int, default=1800, help="Image width in pixels.")
    parser.add_argument("--thickness", type=int, default=3, help="Line thickness.")
    parser.add_argument("--margin", type=int, default=24, help="Image margin.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def render_variant_base(wad_path: Path, tmp_root: Path, iwad_path: Path, width: int, thickness: int, margin: int) -> Path:
    cmd = [
        sys.executable,
        str(WAD2IMAGE_SCRIPT),
        "-c",
        "yadex",
        "--game",
        "doom2",
        "--iwad",
        str(iwad_path),
        "--out-dir",
        str(tmp_root),
        "--width",
        str(width),
        "--background-color",
        "#0b1020",
        "--line-default-color",
        "#f8fafc",
        "--line-colors",
        "none",
        "--line-colors",
        "two_sided=#475569",
        "--thickness",
        str(thickness),
        "--thing-type",
        "sprite",
        "--thing-scale",
        "1.8",
        "--grid-step",
        "0",
        "--margin",
        str(margin),
        str(wad_path),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    pngs = sorted(tmp_root.glob("*.png"))
    if len(pngs) != 1:
        raise RuntimeError(f"Expected exactly one PNG for {wad_path.name}, found {len(pngs)}")
    return pngs[0]


def load_registry(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing variants registry: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["filename"]: entry["config"] for entry in data}


def compute_transform(wad_path: Path, image_width: int, margin: int) -> tuple[float, int, int, int, int]:
    wad = WAD(str(wad_path))
    edit = MapEditor(wad.maps["MAP01"])
    xs = [vertex.x for vertex in edit.vertexes]
    ys = [vertex.y for vertex in edit.vertexes]
    vxmin, vxmax = min(xs), max(xs)
    _, vymax = min(ys), max(ys)
    scale = (image_width - 2 * margin) / float(vxmax - vxmin)
    return scale, margin, margin, vxmin, vymax


def doom_to_pixel(x: float, y: float, *, scale: float, pxmin: int, pymin: int, vxmin: int, vymax: int) -> tuple[int, int]:
    px = int(scale * (x - vxmin) + 0.5) + pxmin
    py = int(scale * (vymax - y) + 0.5) + pymin
    return px, py


def tile_flat(flat_image: Image.Image, size: tuple[int, int], scale: int = 2) -> Image.Image:
    tile = flat_image.convert("RGBA").resize(
        (flat_image.width * scale, flat_image.height * scale),
        Image.Resampling.NEAREST,
    )
    tiled = Image.new("RGBA", size)
    for x in range(0, size[0], tile.width):
        for y in range(0, size[1], tile.height):
            tiled.paste(tile, (x, y))
    return tiled


def apply_masked_texture(base: Image.Image, mask: Image.Image, flat_image: Image.Image, opacity: int) -> None:
    texture = tile_flat(flat_image, base.size)
    alpha = mask.point(lambda p: opacity if p else 0)
    texture.putalpha(alpha)
    base.alpha_composite(texture)


def load_flat_rgba(iwad: WAD, name: str) -> Image.Image:
    return iwad.flats[name].to_Image().convert("RGBA")


def load_sprite_rgba(iwad: WAD, sprite_name: str) -> Image.Image:
    sprite = iwad.sprites[sprite_name].to_Image()
    if sprite.mode == "P":
        # Doom sprite index 247 is the transparent color. Build the alpha
        # channel explicitly so modern Pillow versions do not leave magenta
        # palette squares behind in the overlay preview.
        sprite.info["transparency"] = 247
        alpha = sprite.point(lambda idx: 0 if idx == 247 else 255, mode="L")
        rgba = sprite.convert("RGBA")
        rgba.putalpha(alpha)
        return rgba
    return sprite.convert("RGBA")


def paste_scaled_sprite(base: Image.Image, sprite: Image.Image, center: tuple[int, int], scale: float) -> None:
    resized = sprite.resize(
        (max(1, int(sprite.width * scale)), max(1, int(sprite.height * scale))),
        Image.Resampling.NEAREST,
    )
    x = center[0] - resized.width // 2
    y = center[1] - resized.height // 2
    base.alpha_composite(resized, (x, y))


def area_centers(distance: int) -> dict[str, tuple[int, int]]:
    room_b = (int(distance * 0.866), int(distance * -0.5))
    room_c = (int(distance * -0.866), int(distance * -0.5))
    return {
        "armory": (0, 0),
        "room_a": (0, distance),
        "room_b": room_b,
        "room_c": room_c,
        "corr_a": (0, int(distance * 0.5)),
        "corr_b": (int(distance * 0.433), int(distance * -0.25)),
        "corr_c": (int(distance * -0.433), int(distance * -0.25)),
    }


def sector_index_for_sidedef(edit: MapEditor, sidedef_index: int) -> int | None:
    if sidedef_index in (-1, 0xFFFF):
        return None
    return edit.sidedefs[sidedef_index].sector


def build_sector_loops(edit: MapEditor, sector_index: int) -> list[list[int]]:
    outgoing: dict[int, list[int]] = defaultdict(list)
    for linedef in edit.linedefs:
        front_sector = sector_index_for_sidedef(edit, linedef.front)
        back_sector = sector_index_for_sidedef(edit, linedef.back)
        if front_sector == sector_index:
            outgoing[linedef.vx_a].append(linedef.vx_b)
        if back_sector == sector_index:
            outgoing[linedef.vx_b].append(linedef.vx_a)

    loops: list[list[int]] = []
    visited_edges: set[tuple[int, int]] = set()

    for start, ends in outgoing.items():
        for end in ends:
            if (start, end) in visited_edges:
                continue

            loop = [start]
            current = start
            nxt = end

            while True:
                visited_edges.add((current, nxt))
                loop.append(nxt)
                if nxt == start:
                    break

                next_vertex = next(
                    (candidate for candidate in outgoing.get(nxt, []) if (nxt, candidate) not in visited_edges),
                    None,
                )
                if next_vertex is None:
                    raise RuntimeError(f"Could not close sector loop for sector {sector_index}")
                current, nxt = nxt, next_vertex

            loops.append(loop[:-1])

    if not loops:
        raise RuntimeError(f"No polygon loops found for sector {sector_index}")
    return loops


def rasterize_sector_mask(
    edit: MapEditor,
    sector_index: int,
    image_size: tuple[int, int],
    *,
    scale: float,
    pxmin: int,
    pymin: int,
    vxmin: int,
    vymax: int,
) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for loop in build_sector_loops(edit, sector_index):
        points = [
            doom_to_pixel(
                edit.vertexes[vertex_index].x,
                edit.vertexes[vertex_index].y,
                scale=scale,
                pxmin=pxmin,
                pymin=pymin,
                vxmin=vxmin,
                vymax=vymax,
            )
            for vertex_index in loop
        ]
        draw.polygon(points, fill=255)
    return mask


def texture_opacity_for_sector(floor_texture: str) -> int:
    if floor_texture == "RROCK19":
        return 140
    if floor_texture == "GATE1":
        return 185
    return 165


def add_floor_textures(
    base: Image.Image,
    iwad: WAD,
    edit: MapEditor,
    *,
    scale: float,
    pxmin: int,
    pymin: int,
    vxmin: int,
    vymax: int,
) -> None:
    flat_cache: dict[str, Image.Image] = {}
    for sector_index, sector in enumerate(edit.sectors):
        floor_texture = sector.tx_floor
        if floor_texture in ("-", "F_SKY1"):
            continue

        try:
            flat_image = flat_cache.setdefault(floor_texture, load_flat_rgba(iwad, floor_texture))
        except KeyError:
            continue

        mask = rasterize_sector_mask(
            edit,
            sector_index,
            base.size,
            scale=scale,
            pxmin=pxmin,
            pymin=pymin,
            vxmin=vxmin,
            vymax=vymax,
        )
        apply_masked_texture(base, mask, flat_image, opacity=texture_opacity_for_sector(floor_texture))


def add_preview_sprites(
    base: Image.Image,
    iwad: WAD,
    distance: int,
    *,
    scale: float,
    pxmin: int,
    pymin: int,
    vxmin: int,
    vymax: int,
) -> None:
    core_sprite = load_sprite_rgba(iwad, "TLMPA0")
    zombie_sprite = load_sprite_rgba(iwad, "POSSA1")
    imp_sprite = load_sprite_rgba(iwad, "TROOA1")
    demon_sprite = load_sprite_rgba(iwad, "SARGA1")
    soul_sprite = load_sprite_rgba(iwad, "SKULA1")

    positions = {
        "core": (0, 0),
        "north_monster": (0, int(distance * 0.52)),
        "east_monster": (int(distance * 0.50), int(distance * -0.28)),
        "west_monster": (int(distance * -0.50), int(distance * -0.28)),
        "armory_monster": (90, 96),
    }

    sprite_specs = [
        (core_sprite, positions["core"], 1.9),
        (zombie_sprite, positions["north_monster"], 1.35),
        (imp_sprite, positions["east_monster"], 1.35),
        (demon_sprite, positions["west_monster"], 1.3),
        (soul_sprite, positions["armory_monster"], 1.2),
    ]

    for sprite, pos, sprite_scale in sprite_specs:
        pixel = doom_to_pixel(pos[0], pos[1], scale=scale, pxmin=pxmin, pymin=pymin, vxmin=vxmin, vymax=vymax)
        paste_scaled_sprite(base, sprite, pixel, sprite_scale)


def postprocess_armory_preview(base_png: Path, output_path: Path, wad_path: Path, iwad_path: Path, config: dict, margin: int) -> None:
    base = Image.open(base_png).convert("RGBA")
    iwad = WAD(str(iwad_path))
    wad = WAD(str(wad_path))
    edit = MapEditor(wad.maps["MAP01"])

    scale, pxmin, pymin, vxmin, vymax = compute_transform(wad_path, base.width, margin)
    add_floor_textures(base, iwad, edit, scale=scale, pxmin=pxmin, pymin=pymin, vxmin=vxmin, vymax=vymax)
    add_preview_sprites(base, iwad, int(config["distance"]), scale=scale, pxmin=pxmin, pymin=pymin, vxmin=vxmin, vymax=vymax)
    base.save(output_path)


def main() -> None:
    args = parse_args()
    variants_dir = args.variants_dir.resolve()
    output_dir = args.output_dir.resolve()
    iwad_path = args.iwad.resolve()
    registry_path = args.registry.resolve()

    if not WAD2IMAGE_SCRIPT.exists():
        raise FileNotFoundError(f"Missing wad2image script: {WAD2IMAGE_SCRIPT}")
    if not variants_dir.exists():
        raise FileNotFoundError(f"Missing variants directory: {variants_dir}")
    if not iwad_path.exists():
        raise FileNotFoundError(f"Missing IWAD: {iwad_path}")

    registry = load_registry(registry_path)
    ensure_dir(output_dir)
    wad_paths = sorted(variants_dir.glob("*.wad"))
    if not wad_paths:
        raise FileNotFoundError(f"No WAD variants found in {variants_dir}")

    for wad_path in wad_paths:
        output_path = output_dir / f"{wad_path.stem}.png"
        if wad_path.name not in registry:
            raise KeyError(f"No config entry found in registry for {wad_path.name}")

        print(f"Rendering {wad_path.name} -> {output_path.name}...", flush=True)
        with tempfile.TemporaryDirectory(prefix="wad2image_variant_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            base_png = render_variant_base(
                wad_path=wad_path,
                tmp_root=tmp_root,
                iwad_path=iwad_path,
                width=args.width,
                thickness=args.thickness,
                margin=args.margin,
            )
            postprocess_armory_preview(
                base_png=base_png,
                output_path=output_path,
                wad_path=wad_path,
                iwad_path=iwad_path,
                config=registry[wad_path.name],
                margin=args.margin,
            )

    print(f"Wrote {len(wad_paths)} images to {output_dir}")


if __name__ == "__main__":
    main()
