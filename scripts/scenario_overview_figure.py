from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "paper" / "figure" / "scenarios"
OUTPUT_PATH = REPO_ROOT / "results" / "scenario_overview.png"
SYMLINK_PATH = REPO_ROOT / "paper" / "figure" / "scenario_overview.png"

SCENARIOS = [
    ("1.png", "Stag Hunt Arena"),
    ("2.png", "Rhythm Sync"),
    ("3.png", "Foraging Commons"),
    ("4.png", "Co-op Puzzle"),
    ("5.png", "Platform Chain"),
    ("6.png", "Armory Siege"),
    ("7.png", "Co-op Health Gathering"),
    ("8.png", "Lava Pit"),
    ("9.png", "Smart/Dumb Enemies"),
    ("10.png", "Stealth Labyrinth"),
    ("11.png", "Ammo Carrier"),
    ("12.png", "Lava Maze"),
]

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    src_w, src_h = image.size
    dst_w, dst_h = size
    src_ratio = src_w / src_h
    dst_ratio = dst_w / dst_h

    if src_ratio > dst_ratio:
        crop_w = int(src_h * dst_ratio)
        left = (src_w - crop_w) // 2
        box = (left, 0, left + crop_w, src_h)
    else:
        crop_h = int(src_w / dst_ratio)
        top = (src_h - crop_h) // 2
        box = (0, top, src_w, top + crop_h)

    cropped = image.crop(box)
    return cropped.resize(size, Image.Resampling.LANCZOS)


def make_figure() -> None:
    cols = 4
    rows = 3
    panel_size = (680, 382)
    label_height = 56
    gutter_x = 28
    gutter_y = 34
    outer_pad = 34
    bg = (255, 255, 255)
    panel_border = (215, 223, 232)
    text_color = (17, 24, 39)

    width = outer_pad * 2 + cols * panel_size[0] + (cols - 1) * gutter_x
    height = outer_pad * 2 + rows * (panel_size[1] + label_height) + (rows - 1) * gutter_y
    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)

    label_font = load_font(26)

    for idx, (filename, title) in enumerate(SCENARIOS):
        image_path = SOURCE_DIR / filename
        image = Image.open(image_path).convert("RGB")
        panel = fit_panel(image, panel_size)

        row = idx // cols
        col = idx % cols
        x = outer_pad + col * (panel_size[0] + gutter_x)
        y = outer_pad + row * (panel_size[1] + label_height + gutter_y)

        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + panel_size[0], y + panel_size[1]), outline=panel_border, width=2)

        label = f"({LETTERS[idx]}) {title}"
        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_w = bbox[2] - bbox[0]
        text_x = x + (panel_size[0] - text_w) / 2
        text_y = y + panel_size[1] + 12
        draw.text((text_x, text_y), label, fill=text_color, font=label_font)

    ensure_parent(OUTPUT_PATH)
    canvas.save(OUTPUT_PATH, quality=95)


def update_symlink() -> None:
    ensure_parent(SYMLINK_PATH)
    if SYMLINK_PATH.exists() or SYMLINK_PATH.is_symlink():
        SYMLINK_PATH.unlink()
    SYMLINK_PATH.symlink_to(Path("../../results/scenario_overview.png"))


def main() -> None:
    make_figure()
    update_symlink()
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Linked {SYMLINK_PATH} -> ../../results/scenario_overview.png")


if __name__ == "__main__":
    main()
