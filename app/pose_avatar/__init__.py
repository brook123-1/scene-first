"""Pose-aware local avatar overlay P0.

The package is intentionally independent from provider adapters.  It consumes
confirmed head regions in full-image pixel coordinates and returns a local
composite plus an explainable trace.
"""

from .adapters import detected_head_from_region, enrich_detection
from .models import (
    DetectedHead,
    Expression,
    OverlayDecision,
    RenderRoute,
    RenderTransform,
    Scene,
)
from .registry import AssetRegistry
from .renderer import PoseAvatarRenderer

__all__ = [
    "AssetRegistry",
    "DetectedHead",
    "Expression",
    "OverlayDecision",
    "PoseAvatarRenderer",
    "RenderRoute",
    "RenderTransform",
    "Scene",
    "detected_head_from_region",
    "enrich_detection",
]
