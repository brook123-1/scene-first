from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SCALE = 4

PALETTES = {
    "bald-bearded": ("#17324D", "#74D9C3", "#EAFBF7"),
    "graduate": ("#283457", "#8AB4F8", "#EEF4FF"),
    "architect": ("#3B2E50", "#C4A7E7", "#F7F0FF"),
    "office": ("#283B36", "#8BD5CA", "#ECFAF7"),
    "programmer": ("#243044", "#A6DA95", "#F1FAED"),
    "blue-collar": ("#3A3348", "#F5BDE6", "#FFF1FA"),
}


def _scaled(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(x * SCALE, y * SCALE) for x, y in points]


def _placeholder_cover(path: Path, palette: tuple[str, str, str], variant: int) -> None:
    size = 512
    image = Image.new("RGBA", (size * SCALE, size * SCALE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    dark, accent, light = palette
    shift = (variant - 2) * 5

    draw.ellipse((82 * SCALE, 310 * SCALE, 430 * SCALE, 620 * SCALE), fill=dark)
    draw.rounded_rectangle(
        ((154 + shift) * SCALE, 246 * SCALE, (358 + shift) * SCALE, 410 * SCALE),
        radius=38 * SCALE,
        fill=accent,
    )
    draw.ellipse(
        ((126 + shift) * SCALE, 50 * SCALE, (386 + shift) * SCALE, 330 * SCALE),
        fill=light,
        outline=dark,
        width=14 * SCALE,
    )
    draw.ellipse(
        ((189 + shift) * SCALE, 157 * SCALE, (215 + shift) * SCALE, 183 * SCALE),
        fill=dark,
    )
    draw.ellipse(
        ((297 + shift) * SCALE, 157 * SCALE, (323 + shift) * SCALE, 183 * SCALE),
        fill=dark,
    )
    draw.rounded_rectangle(
        ((170 + shift) * SCALE, 202 * SCALE, (342 + shift) * SCALE, 266 * SCALE),
        radius=28 * SCALE,
        fill=accent,
        outline=dark,
        width=8 * SCALE,
    )
    draw.line(
        _scaled([(170 + shift, 220), (126 + shift, 194)]),
        fill=dark,
        width=8 * SCALE,
    )
    draw.line(
        _scaled([(342 + shift, 220), (386 + shift, 194)]),
        fill=dark,
        width=8 * SCALE,
    )

    output = image.resize((size, size), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path, "PNG", optimize=True)


def _icon_png(path: Path, size: int) -> None:
    canvas = size * SCALE
    image = Image.new("RGBA", (canvas, canvas), "#102A45")
    draw = ImageDraw.Draw(image)
    margin = round(size * 0.17) * SCALE
    center = size // 2
    shield = _scaled(
        [
            (center, round(size * 0.14)),
            (size - margin, round(size * 0.27)),
            (size - margin, round(size * 0.57)),
            (center, round(size * 0.86)),
            (margin, round(size * 0.57)),
            (margin, round(size * 0.27)),
        ]
    )
    draw.polygon(shield, fill="#EAFBF7", outline="#74D9C3", width=max(SCALE, round(size * 0.035) * SCALE))
    radius = round(size * 0.12) * SCALE
    cx, cy = center * SCALE, round(size * 0.39) * SCALE
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill="#102A45")
    band = (
        round(size * 0.31) * SCALE,
        round(size * 0.49) * SCALE,
        round(size * 0.69) * SCALE,
        round(size * 0.62) * SCALE,
    )
    draw.rounded_rectangle(band, radius=round(size * 0.055) * SCALE, fill="#74D9C3")

    output = image.resize((size, size), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path, "PNG", optimize=True)


def _icon_svg(path: Path) -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" role="img" aria-label="Scene First placeholder icon">
  <rect width="512" height="512" rx="96" fill="#102a45"/>
  <path d="M256 72 425 138v153c0 70-69 127-169 165C156 418 87 361 87 291V138Z" fill="#eafbf7" stroke="#74d9c3" stroke-width="18" stroke-linejoin="round"/>
  <circle cx="256" cy="200" r="61" fill="#102a45"/>
  <rect x="158" y="251" width="196" height="70" rx="28" fill="#74d9c3"/>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8", newline="\n")


def generate(root: Path) -> list[Path]:
    generated: list[Path] = []
    cover_root = root / "static" / "assets" / "safe-covers"
    for variant, (name, palette) in enumerate(sorted(PALETTES.items())):
        destination = cover_root / f"{name}.png"
        _placeholder_cover(destination, palette, variant)
        generated.append(destination)

    asset_root = root / "static" / "assets"
    svg_path = asset_root / "app-icon.svg"
    _icon_svg(svg_path)
    generated.append(svg_path)
    for size in (192, 512):
        destination = asset_root / f"app-icon-{size}.png"
        _icon_png(destination, size)
        generated.append(destination)
    return generated


def _matches_committed(committed_path: Path, generated_path: Path) -> bool:
    if not committed_path.is_file():
        return False
    if generated_path.suffix.lower() == ".svg":
        return committed_path.read_text(encoding="utf-8") == generated_path.read_text(encoding="utf-8")
    return committed_path.read_bytes() == generated_path.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic license-clear public placeholder art.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to populate")
    parser.add_argument("--check", action="store_true", help="Verify committed outputs without modifying them")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.check:
        with TemporaryDirectory(prefix="scene-first-placeholders-") as temporary_directory:
            generated = generate(Path(temporary_directory))
            for temporary_path in generated:
                relative_path = temporary_path.relative_to(temporary_directory)
                committed_path = root / relative_path
                if not _matches_committed(committed_path, temporary_path):
                    raise SystemExit(f"placeholder mismatch: {relative_path.as_posix()}")
                print(f"OK {relative_path.as_posix()}")
        return
    for path in generate(root):
        digest = sha256(path.read_bytes()).hexdigest()
        print(f"{path.relative_to(root).as_posix()}  sha256:{digest}")


if __name__ == "__main__":
    main()
