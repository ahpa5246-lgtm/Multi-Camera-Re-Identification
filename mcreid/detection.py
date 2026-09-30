from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TrackedPerson:
    local_track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float


class DetectorTracker:
    """YOLO person detector with one ByteTrack state per camera."""

    def __init__(self, model_name: str, confidence: float, iou: float, device: str = "auto") -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.iou = iou
        self.device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def track(self, frame: np.ndarray) -> list[TrackedPerson]:
        height, width = frame.shape[:2]
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        ids = boxes.id.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()

        tracked: list[TrackedPerson] = []
        for box, track_id, confidence in zip(xyxy, ids, confidences, strict=True):
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            tracked.append(TrackedPerson(int(track_id), (x1, y1, x2, y2), float(confidence)))
        return tracked
