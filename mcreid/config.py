from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PipelineConfig:
    detector_model: str = "yolo11n.pt"
    reid_model: str = "vit_small_patch14_dinov2.lvd142m"
    detection_confidence: float = 0.35
    detection_iou: float = 0.55
    reid_similarity_threshold: float = 0.72
    reid_min_margin: float = 0.04
    embedding_interval: int = 8
    min_crop_height: int = 80
    min_crop_width: int = 28
    embedding_history: int = 12
    max_idle_seconds: float = 300.0
    same_camera_reentry_seconds: float = 12.0
    overlap_tolerance_seconds: float = 0.75
    strict_topology: bool = False
    device: str = "auto"
    camera_time_offsets: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if not 0.0 < self.detection_confidence <= 1.0:
            raise ValueError("detection_confidence must be in (0, 1].")
        if not 0.0 < self.reid_similarity_threshold <= 1.0:
            raise ValueError("reid_similarity_threshold must be in (0, 1].")
        if self.embedding_interval < 1:
            raise ValueError("embedding_interval must be >= 1.")
        if self.min_crop_height < 1 or self.min_crop_width < 1:
            raise ValueError("minimum crop dimensions must be positive.")
        if self.reid_min_margin < 0:
            raise ValueError("reid_min_margin must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
