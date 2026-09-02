from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import AVATAR_ASSET_DIR
from app.pose_avatar.debug import write_debug_bundle
from app.pose_avatar.models import DetectedHead, PoseEstimate, Scene, View
from app.pose_avatar.registry import AssetRegistry
from app.pose_avatar.renderer import PoseAvatarRenderer
from app.pose_avatar.transform import apply_matrix, solve_transform


def background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (205 + x * 20 // width, 216 + y * 12 // height, 225 + (x + y) * 12 // (width + height))
    return image


def draw_person(image: Image.Image, bbox, color=(83, 108, 132)) -> None:
    draw = ImageDraw.Draw(image)
    x, y, width, height = bbox
    draw.ellipse((x + width * 0.08, y, x + width * 0.92, y + height * 0.78), fill=(197, 151, 126), outline=(62, 67, 76), width=3)
    draw.ellipse((x - width * 0.45, y + height * 0.76, x + width * 1.45, y + height * 2.2), fill=color, outline=(62, 67, 76), width=3)


def aligned_head(registry: AssetRegistry, scene: Scene, head_id: str, bbox, roll=0.0, depth=0.0) -> DetectedHead:
    yaw = {Scene.S01_FRONT_NEUTRAL: 0.0, Scene.S04_L34_NEUTRAL: -30.0, Scene.S07_R34_NEUTRAL: 30.0, Scene.S12_BACK: 0.0}[scene]
    x, y, width, height = bbox
    value = DetectedHead(
        head_id=head_id,
        bbox=bbox,
        body_landmarks={
            "left_shoulder": (x + width * 0.04, y + height * 1.02),
            "right_shoulder": (x + width * 0.96, y + height * 1.02),
            "neck_center": (x + width * 0.5, y + height * 0.88),
        },
        pose=PoseEstimate(yaw_deg=yaw, pitch_deg=0, roll_deg=roll, confidence=0.97),
        view_hint=View.BACK if scene == Scene.S12_BACK else None,
        visibility=0.96,
        occlusion_score=0.02,
        confidence=0.97,
        depth_order=depth,
    )
    record = registry.select(scene)
    transform = solve_transform(value, scene, record.anchors)
    mapping = {"left_eye_center": "left_eye", "right_eye_center": "right_eye", "nose_tip": "nose", "chin": "chin"}
    face = {
        target: apply_matrix(transform.matrix, source)
        for source_name, target in mapping.items()
        if (source := record.anchors.anchors.get(source_name)) is not None
    }
    return value.model_copy(update={"face_landmarks": face})


def build_cases(registry: AssetRegistry):
    first = background((640, 420))
    first_box = (255.0, 75.0, 130.0, 160.0)
    draw_person(first, first_box)
    front_roll = [aligned_head(registry, Scene.S01_FRONT_NEUTRAL, "front-roll", first_box, roll=18.0)]

    multi = background((820, 460))
    left_box, right_box = (125.0, 95.0, 130.0, 155.0), (540.0, 80.0, 150.0, 180.0)
    draw_person(multi, left_box, (95, 122, 142))
    draw_person(multi, right_box, (131, 105, 126))
    multi_heads = [
        aligned_head(registry, Scene.S04_L34_NEUTRAL, "multi-left", left_box, roll=-7.0, depth=1),
        aligned_head(registry, Scene.S07_R34_NEUTRAL, "multi-right", right_box, roll=9.0, depth=2),
    ]

    back = background((640, 420))
    back_box = (250.0, 70.0, 140.0, 170.0)
    draw_person(back, back_box, (73, 102, 126))
    back_heads = [aligned_head(registry, Scene.S12_BACK, "back-head", back_box, roll=-11.0)]

    fallback = background((900, 430))
    specs = [
        ("tiny", (70.0, 100.0, 42.0, 46.0), {"confidence": 0.96}),
        ("low-confidence", (250.0, 75.0, 105.0, 130.0), {"confidence": 0.50}),
        ("occluded", (485.0, 70.0, 115.0, 140.0), {"occlusion_score": 0.72}),
        ("unsupported-profile", (720.0, 70.0, 115.0, 140.0), {"yaw": 62.0}),
    ]
    fallback_heads = []
    for index, (name, box, changes) in enumerate(specs):
        draw_person(fallback, box, (88 + index * 15, 108, 132))
        x, y, width, height = box
        fallback_heads.append(DetectedHead(
            head_id=name,
            bbox=box,
            body_landmarks={"neck_center": (x + width / 2, y + height * 0.88)},
            pose=PoseEstimate(yaw_deg=changes.get("yaw", 0), pitch_deg=0, roll_deg=0, confidence=0.95),
            confidence=changes.get("confidence", 0.96),
            visibility=0.96,
            occlusion_score=changes.get("occlusion_score", 0.02),
            depth_order=index,
        ))
    return {
        "front-roll": (first, front_roll),
        "multi-left-right": (multi, multi_heads),
        "back": (back, back_heads),
        "fallbacks": (fallback, fallback_heads),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="debug/pose-avatar-p0")
    args = parser.parse_args()
    output = Path(args.output)
    registry = AssetRegistry(AVATAR_ASSET_DIR)
    renderer = PoseAvatarRenderer(registry)
    summaries = []
    for name, (image, heads) in build_cases(registry).items():
        result = renderer.render(image, heads, image_id=f"fixture-{name}")
        write_debug_bundle(output / name, image, heads, result)
        summaries.append((name, [f"{value.head_id}: {value.route_type.value} / {value.scene_id.value if value.scene_id else '-'} / {value.fallback_reason or '-'}" for value in result.decisions]))
    lines = ["# Pose-aware Avatar Overlay P0 debug fixtures", "", "这些图片是确定性合成 fixture，不是真实人物照片。每个目录均包含原图、bbox/landmark/pose 标注、变换后 overlay、最终 composite 和完整 Decision Trace。", ""]
    for name, decisions in summaries:
        lines += [f"## {name}", ""] + [f"- {value}" for value in decisions] + [""]
    (output / "README.md").write_text("\n".join(lines), "utf8")
    print(output)


if __name__ == "__main__":
    main()
