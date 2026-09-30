from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import PipelineConfig
from .detection import DetectorTracker
from .embedding import AppearanceEmbedder, mean_embedding
from .identity import GlobalIdentityManager, Tracklet
from .storage import EventStore
from .topology import CameraTopology
from .visualization import render_annotated_video


ProgressCallback = Callable[[str, int, int], None]


@dataclass(slots=True)
class RunResult:
    output_dir: Path
    database_path: Path
    run_id: int
    annotated_videos: dict[str, Path]
    tracklets: list[Tracklet]
    observations: list[dict]
    elapsed_seconds: float

    @property
    def global_identity_count(self) -> int:
        return len({t.global_id for t in self.tracklets if t.global_id is not None})


@dataclass(slots=True)
class _Accumulator:
    start_time: float
    end_time: float
    frame_count: int = 0
    confidence_sum: float = 0.0
    embeddings: deque[np.ndarray] = field(default_factory=deque)
    best_crop: np.ndarray | None = None
    best_area: int = 0

    def update_seen(self, timestamp: float, confidence: float) -> None:
        self.end_time = timestamp
        self.frame_count += 1
        self.confidence_sum += confidence


class MultiCameraPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        topology: CameraTopology | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.config.validate()
        self.topology = topology or CameraTopology(strict=self.config.strict_topology)

    def _process_camera(
        self,
        camera_id: str,
        video_path: Path,
        embedder: AppearanceEmbedder,
        progress_callback: ProgressCallback | None,
    ) -> tuple[list[Tracklet], list[dict], dict[int, np.ndarray]]:
        tracker = DetectorTracker(
            model_name=self.config.detector_model,
            confidence=self.config.detection_confidence,
            iou=self.config.detection_iou,
            device=self.config.device,
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for {camera_id}: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        offset = float(self.config.camera_time_offsets.get(camera_id, 0.0))

        accumulators: dict[int, _Accumulator] = {}
        observations: list[dict] = []
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_index / fps + offset
            for detection in tracker.track(frame):
                local_id = detection.local_track_id
                x1, y1, x2, y2 = detection.bbox
                accumulator = accumulators.get(local_id)
                if accumulator is None:
                    accumulator = _Accumulator(
                        start_time=timestamp,
                        end_time=timestamp,
                        embeddings=deque(maxlen=self.config.embedding_history),
                    )
                    accumulators[local_id] = accumulator

                accumulator.update_seen(timestamp, detection.confidence)
                crop = frame[y1:y2, x1:x2]
                crop_h, crop_w = crop.shape[:2]
                area = crop_h * crop_w

                if area > accumulator.best_area:
                    accumulator.best_area = area
                    accumulator.best_crop = crop.copy()

                should_embed = (
                    len(accumulator.embeddings) == 0
                    or frame_index % self.config.embedding_interval == 0
                )
                if (
                    should_embed
                    and crop_h >= self.config.min_crop_height
                    and crop_w >= self.config.min_crop_width
                ):
                    accumulator.embeddings.append(embedder.extract(crop))

                observations.append(
                    {
                        "camera_id": camera_id,
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "local_track_id": local_id,
                        "bbox": detection.bbox,
                        "confidence": detection.confidence,
                    }
                )

            frame_index += 1
            if progress_callback and (frame_index % 20 == 0 or frame_index == total_frames):
                progress_callback(camera_id, frame_index, total_frames)

        cap.release()

        tracklets: list[Tracklet] = []
        thumbnails: dict[int, np.ndarray] = {}
        for local_id, accumulator in accumulators.items():
            embedding = mean_embedding(accumulator.embeddings)
            mean_conf = (
                accumulator.confidence_sum / accumulator.frame_count
                if accumulator.frame_count else 0.0
            )
            tracklets.append(
                Tracklet(
                    camera_id=camera_id,
                    local_track_id=local_id,
                    start_time=accumulator.start_time,
                    end_time=accumulator.end_time,
                    embedding=embedding,
                    frame_count=accumulator.frame_count,
                    mean_confidence=mean_conf,
                )
            )
            if accumulator.best_crop is not None:
                thumbnails[local_id] = accumulator.best_crop

        return tracklets, observations, thumbnails

    def run(
        self,
        cameras: dict[str, str | Path],
        output_dir: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> RunResult:
        if not cameras:
            raise ValueError("At least one camera/video input is required.")

        started = time.perf_counter()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "videos").mkdir(exist_ok=True)
        (output_dir / "thumbnails").mkdir(exist_ok=True)

        embedder = AppearanceEmbedder(self.config.reid_model, self.config.device)

        all_tracklets: list[Tracklet] = []
        all_observations: list[dict] = []
        thumbnails: dict[tuple[str, int], np.ndarray] = {}

        for camera_id, source in cameras.items():
            camera_tracklets, camera_observations, camera_thumbnails = self._process_camera(
                camera_id, Path(source), embedder, progress_callback
            )
            all_tracklets.extend(camera_tracklets)
            all_observations.extend(camera_observations)
            for local_id, image in camera_thumbnails.items():
                thumbnails[(camera_id, local_id)] = image

        identity_manager = GlobalIdentityManager(
            similarity_threshold=self.config.reid_similarity_threshold,
            min_margin=self.config.reid_min_margin,
            max_idle_seconds=self.config.max_idle_seconds,
            same_camera_reentry_seconds=self.config.same_camera_reentry_seconds,
            overlap_tolerance_seconds=self.config.overlap_tolerance_seconds,
            topology=self.topology,
        )
        mapping = identity_manager.associate(all_tracklets)

        for observation in all_observations:
            observation["global_id"] = mapping[
                (observation["camera_id"], observation["local_track_id"])
            ]

        for tracklet in all_tracklets:
            crop = thumbnails.get(tracklet.key)
            if crop is None:
                continue
            filename = (
                f"G{int(tracklet.global_id or 0):04d}_"
                f"{tracklet.camera_id}_L{tracklet.local_track_id}.jpg"
            )
            thumb_path = output_dir / "thumbnails" / filename
            cv2.imwrite(str(thumb_path), crop)
            tracklet.thumbnail_path = str(thumb_path)

        annotated_videos: dict[str, Path] = {}
        for camera_id, source in cameras.items():
            camera_observations = [o for o in all_observations if o["camera_id"] == camera_id]
            output_path = output_dir / "videos" / f"{camera_id}_annotated.mp4"
            render_annotated_video(source, output_path, camera_observations)
            annotated_videos[camera_id] = output_path

        database_path = output_dir / "events.sqlite3"
        store = EventStore(database_path)
        run_id = store.create_run(self.config.to_dict())
        store.save_tracklets(run_id, all_tracklets)
        store.save_observations(run_id, all_observations)
        store.close()

        return RunResult(
            output_dir=output_dir,
            database_path=database_path,
            run_id=run_id,
            annotated_videos=annotated_videos,
            tracklets=all_tracklets,
            observations=all_observations,
            elapsed_seconds=time.perf_counter() - started,
        )
