from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .embedding import l2_normalize
from .topology import CameraTopology


@dataclass(slots=True)
class Tracklet:
    camera_id: str
    local_track_id: int
    start_time: float
    end_time: float
    embedding: np.ndarray | None
    frame_count: int
    mean_confidence: float
    thumbnail_path: str | None = None
    global_id: int | None = None

    @property
    def key(self) -> tuple[str, int]:
        return self.camera_id, self.local_track_id


@dataclass(slots=True)
class GlobalProfile:
    global_id: int
    centroid: np.ndarray | None
    last_camera: str
    last_end: float
    history: list[tuple[str, int]] = field(default_factory=list)


class GlobalIdentityManager:
    """Tracklet-level association with appearance and spatio-temporal gates."""

    def __init__(
        self,
        similarity_threshold: float,
        min_margin: float,
        max_idle_seconds: float,
        same_camera_reentry_seconds: float,
        overlap_tolerance_seconds: float,
        topology: CameraTopology | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.min_margin = min_margin
        self.max_idle_seconds = max_idle_seconds
        self.same_camera_reentry_seconds = same_camera_reentry_seconds
        self.overlap_tolerance_seconds = overlap_tolerance_seconds
        self.topology = topology or CameraTopology()
        self.profiles: dict[int, GlobalProfile] = {}
        self.mapping: dict[tuple[str, int], int] = {}
        self._next_id = 1

    @staticmethod
    def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return -1.0
        return float(np.dot(l2_normalize(a), l2_normalize(b)))

    def _candidate_score(self, tracklet: Tracklet, profile: GlobalProfile) -> float | None:
        gap = tracklet.start_time - profile.last_end
        if gap < -self.overlap_tolerance_seconds or gap > self.max_idle_seconds:
            return None

        if tracklet.camera_id == profile.last_camera:
            if gap < 0 or gap > self.same_camera_reentry_seconds:
                return None
        elif not self.topology.allows(profile.last_camera, tracklet.camera_id, max(gap, 0.0)):
            return None

        score = self.similarity(tracklet.embedding, profile.centroid)
        return score if score >= self.similarity_threshold else None

    def _new_profile(self, tracklet: Tracklet) -> int:
        global_id = self._next_id
        self._next_id += 1
        self.profiles[global_id] = GlobalProfile(
            global_id,
            tracklet.embedding.copy() if tracklet.embedding is not None else None,
            tracklet.camera_id,
            tracklet.end_time,
            [tracklet.key],
        )
        return global_id

    def _update_profile(self, profile: GlobalProfile, tracklet: Tracklet) -> None:
        if tracklet.embedding is not None:
            if profile.centroid is None:
                profile.centroid = tracklet.embedding.copy()
            else:
                profile.centroid = l2_normalize(0.8 * profile.centroid + 0.2 * tracklet.embedding)
        profile.last_camera = tracklet.camera_id
        profile.last_end = max(profile.last_end, tracklet.end_time)
        profile.history.append(tracklet.key)

    def associate(self, tracklets: list[Tracklet]) -> dict[tuple[str, int], int]:
        for tracklet in sorted(tracklets, key=lambda t: (t.start_time, t.end_time, t.camera_id)):
            candidates: list[tuple[float, int]] = []
            for global_id, profile in self.profiles.items():
                score = self._candidate_score(tracklet, profile)
                if score is not None:
                    candidates.append((score, global_id))

            candidates.sort(reverse=True)
            chosen: int | None = None
            if candidates:
                best_score, best_id = candidates[0]
                second_score = candidates[1][0] if len(candidates) > 1 else -1.0
                if best_score - second_score >= self.min_margin or second_score < self.similarity_threshold:
                    chosen = best_id

            if chosen is None:
                chosen = self._new_profile(tracklet)
            else:
                self._update_profile(self.profiles[chosen], tracklet)

            tracklet.global_id = chosen
            self.mapping[tracklet.key] = chosen

        return dict(self.mapping)
