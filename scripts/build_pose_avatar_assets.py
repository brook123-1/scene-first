from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image, ImageDraw


def color(value: str | None):
    if not value or value == "none":
        return None
    value = value.lstrip("#")
    if len(value) == 6:
        return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4)) + (255,)
    raise ValueError(f"unsupported SVG color: {value}")


def number(element, key, default=0.0) -> float:
    return float(element.attrib.get(key, default))


def render_svg(source: Path, destination: Path) -> None:
    root = ElementTree.parse(source).getroot()
    width, height = int(float(root.attrib["width"])), int(float(root.attrib["height"]))
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for element in root:
        tag = element.tag.rsplit("}", 1)[-1]
        fill = color(element.attrib.get("fill"))
        stroke = color(element.attrib.get("stroke"))
        stroke_width = max(1, round(number(element, "stroke-width", 1)))
        if tag == "rect":
            box = (number(element, "x"), number(element, "y"), number(element, "x") + number(element, "width"), number(element, "y") + number(element, "height"))
            radius = number(element, "rx")
            draw.rounded_rectangle(box, radius=radius, fill=fill, outline=stroke, width=stroke_width)
        elif tag == "ellipse":
            cx, cy, rx, ry = (number(element, key) for key in ("cx", "cy", "rx", "ry"))
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fill, outline=stroke, width=stroke_width)
        elif tag == "circle":
            cx, cy, radius = (number(element, key) for key in ("cx", "cy", "r"))
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill, outline=stroke, width=stroke_width)
        elif tag == "line":
            draw.line(tuple(number(element, key) for key in ("x1", "y1", "x2", "y2")), fill=stroke, width=stroke_width)
        elif tag == "polygon":
            points = [tuple(float(value) for value in pair.split(",")) for pair in element.attrib["points"].split()]
            draw.polygon(points, fill=fill)
            if stroke:
                draw.line(points + [points[0]], fill=stroke, width=stroke_width, joint="curve")
        else:
            raise ValueError(f"unsupported SVG element: {tag}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, "PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic P0 avatar PNG assets from the restricted SVG subset.")
    parser.add_argument("--family", default="assets/avatar_families/generic")
    args = parser.parse_args()
    family = Path(args.family)
    source = family / "source_svg"
    output = family / "runtime_png"
    for path in sorted(source.glob("*.svg")):
        render_svg(path, output / f"{path.stem}.png")
        print(output / f"{path.stem}.png")


if __name__ == "__main__":
    main()
