from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field, model_validator

from .models import Expression, Scene


class AssetValidationError(ValueError):
    pass


class SceneAssetSpec(BaseModel):
    asset: str
    anchor_key: str
    constraint_key: str


class FamilyManifest(BaseModel):
    schema_version: Literal["1.0", "1.1"]
    family_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    style_group: str
    privacy_mode: Literal["non_identity_preserving"]
    supported_scenes: list[Scene]
    supported_expressions: list[Expression]
    supported_accessories: list[str] = Field(default_factory=list)
    default_priority: int = 100
    enabled: bool = True
    playground_only: bool = False
    anchors_file: str
    constraints_file: str
    scenes: dict[Scene, SceneAssetSpec]

    @model_validator(mode="after")
    def scenes_match_support(self) -> "FamilyManifest":
        if set(self.scenes) != set(self.supported_scenes):
            raise ValueError("manifest scenes must exactly match supported_scenes")
        return self


class SceneAnchors(BaseModel):
    canvas_size: tuple[int, int]
    anchors: dict[str, tuple[float, float] | None]
    overlay_bbox: tuple[float, float, float, float]
    safe_mask_polygon: list[tuple[float, float]]
    silhouette_bbox: tuple[float, float, float, float] | None = None
    silhouette_polygon: list[tuple[float, float]] | None = None
    neck_overlap_zone: list[tuple[float, float]] | None = None
    transparent_canvas_bounds: tuple[float, float, float, float] | None = None


class AnchorsDocument(BaseModel):
    schema_version: Literal["1.0", "1.1"]
    scenes: dict[str, SceneAnchors]


class SceneConstraint(BaseModel):
    scene_id: Scene
    yaw_range_deg: tuple[float, float]
    pitch_range_deg: tuple[float, float]
    roll_range_deg: tuple[float, float]
    min_head_size_px: float = Field(gt=0)
    max_head_size_px: float = Field(gt=0)
    visibility_min: float = Field(ge=0, le=1)
    crop_tolerance: float = Field(ge=0, le=1)
    max_occlusion_score: float = Field(ge=0, le=1)


class ConstraintsDocument(BaseModel):
    schema_version: Literal["1.0", "1.1"]
    scenes: dict[str, SceneConstraint]


@dataclass(frozen=True)
class AssetRecord:
    family: FamilyManifest
    scene: Scene
    path: Path
    anchors: SceneAnchors
    constraints: SceneConstraint


def _orientation(a, b, c) -> float:
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])


def _segments_cross(a, b, c, d) -> bool:
    return (_orientation(a, b, c) * _orientation(a, b, d) < 0) and (
        _orientation(c, d, a) * _orientation(c, d, b) < 0
    )


def _polygon_self_intersects(points: list[tuple[float, float]]) -> bool:
    if len(points) < 3:
        return True
    edges = list(zip(points, points[1:] + points[:1]))
    for first, (a, b) in enumerate(edges):
        for second, (c, d) in enumerate(edges):
            if abs(first - second) <= 1 or {first, second} == {0, len(edges) - 1}:
                continue
            if _segments_cross(a, b, c, d):
                return True
    return False


class AssetRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._records: dict[tuple[str, Scene], AssetRecord] = {}
        self._image_cache: dict[Path, Image.Image] = {}
        self._load()

    def _load_json(self, path: Path) -> dict:
        try:
            return json.loads(path.read_text("utf8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AssetValidationError(f"invalid JSON: {path}") from exc

    def _load(self) -> None:
        if not self.root.is_dir():
            raise AssetValidationError(f"asset root is missing: {self.root}")
        identities: set[tuple[str, str]] = set()
        errors: list[str] = []
        for manifest_path in sorted(self.root.glob("*/manifest.json")):
            family_dir = manifest_path.parent
            try:
                manifest = FamilyManifest.model_validate(self._load_json(manifest_path))
                identity = (manifest.family_id, manifest.version)
                if identity in identities:
                    raise AssetValidationError(f"duplicate family version: {identity}")
                identities.add(identity)
                anchors = AnchorsDocument.model_validate(self._load_json(family_dir / manifest.anchors_file))
                constraints = ConstraintsDocument.model_validate(self._load_json(family_dir / manifest.constraints_file))
                for scene, spec in manifest.scenes.items():
                    if spec.anchor_key not in anchors.scenes:
                        raise AssetValidationError(f"missing anchors for {scene}")
                    if spec.constraint_key not in constraints.scenes:
                        raise AssetValidationError(f"missing constraints for {scene}")
                    scene_anchors = anchors.scenes[spec.anchor_key]
                    constraint = constraints.scenes[spec.constraint_key]
                    if constraint.scene_id != scene:
                        raise AssetValidationError(f"constraint scene mismatch for {scene}")
                    asset_path = family_dir / spec.asset
                    self._validate_asset(
                        scene,
                        asset_path,
                        scene_anchors,
                        require_real_geometry=manifest.schema_version == "1.1",
                    )
                    self._records[(manifest.family_id, scene)] = AssetRecord(
                        family=manifest, scene=scene, path=asset_path,
                        anchors=scene_anchors, constraints=constraint,
                    )
            except Exception as exc:
                errors.append(f"{manifest_path}: {exc}")
        if errors:
            raise AssetValidationError("; ".join(errors))
        if not self._records:
            raise AssetValidationError("no valid avatar families found")

    @staticmethod
    def _open_asset(path: Path) -> Image.Image:
        if path.suffix.lower() == ".svg":
            try:
                import cairosvg
            except ImportError as exc:
                raise AssetValidationError("SVG assets require cairosvg; run the project setup") from exc
            try:
                return Image.open(BytesIO(cairosvg.svg2png(url=str(path)))).convert("RGBA")
            except Exception as exc:
                raise AssetValidationError(f"unreadable SVG asset: {path}") from exc
        try:
            with Image.open(path) as image:
                image.load()
                return image.convert("RGBA") if image.mode == "RGBA" else image.copy()
        except OSError as exc:
            raise AssetValidationError(f"unreadable asset: {path}") from exc

    @classmethod
    def _validate_asset(
        cls,
        scene: Scene,
        path: Path,
        anchors: SceneAnchors,
        *,
        require_real_geometry: bool = False,
    ) -> None:
        if not path.is_file():
            raise AssetValidationError(f"asset file missing: {path}")
        if path.suffix.lower() not in {".png", ".svg"}:
            raise AssetValidationError(f"asset must be transparent PNG or SVG: {path}")
        image = cls._open_asset(path)
        if image.mode != "RGBA" or "A" not in image.getbands():
            raise AssetValidationError(f"asset must have alpha transparency: {path}")
        if image.size != anchors.canvas_size:
            raise AssetValidationError(f"canvas size mismatch: {path}")
        if not image.getchannel("A").getbbox():
            raise AssetValidationError(f"asset alpha is empty: {path}")

        width, height = anchors.canvas_size
        required = {"head_top", "chin", "neck_left", "neck_right"}
        if scene != Scene.S12_BACK:
            required |= {"nose_tip"}
        missing = sorted(name for name in required if anchors.anchors.get(name) is None)
        if missing:
            raise AssetValidationError(f"missing required anchors for {scene}: {missing}")
        if scene != Scene.S12_BACK and not (anchors.anchors.get("left_eye_center") or anchors.anchors.get("right_eye_center")):
            raise AssetValidationError(f"at least one eye anchor is required for {scene}")
        if require_real_geometry:
            geometry = {
                "silhouette_bbox": anchors.silhouette_bbox,
                "silhouette_polygon": anchors.silhouette_polygon,
                "neck_overlap_zone": anchors.neck_overlap_zone,
                "transparent_canvas_bounds": anchors.transparent_canvas_bounds,
            }
            missing_geometry = sorted(name for name, value in geometry.items() if not value)
            if missing_geometry:
                raise AssetValidationError(f"schema 1.1 scene is missing geometry: {missing_geometry}")
        points = [value for value in anchors.anchors.values() if value is not None]
        points += list(anchors.safe_mask_polygon)
        points += list(anchors.silhouette_polygon or [])
        points += list(anchors.neck_overlap_zone or [])
        x, y, box_width, box_height = anchors.overlay_bbox
        if x < 0 or y < 0 or box_width <= 0 or box_height <= 0 or x + box_width > width or y + box_height > height:
            raise AssetValidationError(f"overlay_bbox outside canvas for {scene}")
        if any(px < 0 or py < 0 or px > width or py > height for px, py in points):
            raise AssetValidationError(f"anchor outside canvas for {scene}")
        if _polygon_self_intersects(anchors.safe_mask_polygon):
            raise AssetValidationError(f"safe_mask_polygon self-intersects for {scene}")
        if anchors.silhouette_polygon and _polygon_self_intersects(anchors.silhouette_polygon):
            raise AssetValidationError(f"silhouette_polygon self-intersects for {scene}")
        if anchors.neck_overlap_zone and _polygon_self_intersects(anchors.neck_overlap_zone):
            raise AssetValidationError(f"neck_overlap_zone self-intersects for {scene}")
        for name, box in (("silhouette_bbox", anchors.silhouette_bbox), ("transparent_canvas_bounds", anchors.transparent_canvas_bounds)):
            if box is None:
                continue
            bx, by, bw, bh = box
            if bx < 0 or by < 0 or bw <= 0 or bh <= 0 or bx + bw > width or by + bh > height:
                raise AssetValidationError(f"{name} outside canvas for {scene}")

    def select(
        self,
        scene: Scene,
        family_id: str | None = None,
        *,
        include_playground: bool = False,
    ) -> AssetRecord | None:
        candidates = [
            value for (candidate_family, candidate_scene), value in self._records.items()
            if candidate_scene == scene
            and value.family.enabled
            and (include_playground or not value.family.playground_only)
            and (family_id is None or candidate_family == family_id)
        ]
        candidates.sort(key=lambda value: (-value.family.default_priority, value.family.family_id))
        return candidates[0] if candidates else None

    def image(self, record: AssetRecord) -> Image.Image:
        if record.path not in self._image_cache:
            self._image_cache[record.path] = self._open_asset(record.path).copy()
        return self._image_cache[record.path].copy()

    @property
    def records(self) -> list[AssetRecord]:
        return list(self._records.values())
