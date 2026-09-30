import numpy as np

from mcreid.identity import GlobalIdentityManager, Tracklet
from mcreid.topology import CameraTopology, TransitionWindow


def vec(*values):
    x = np.asarray(values, dtype=np.float32)
    return x / np.linalg.norm(x)


def track(camera, local_id, start, end, embedding):
    return Tracklet(
        camera_id=camera,
        local_track_id=local_id,
        start_time=start,
        end_time=end,
        embedding=embedding,
        frame_count=20,
        mean_confidence=0.9,
    )


def manager(topology=None, threshold=0.8):
    return GlobalIdentityManager(
        similarity_threshold=threshold,
        min_margin=0.03,
        max_idle_seconds=300,
        same_camera_reentry_seconds=15,
        overlap_tolerance_seconds=0.5,
        topology=topology,
    )


def test_same_appearance_across_cameras_links_to_one_global_id():
    m = manager()
    a = track("cam1", 1, 0, 5, vec(1.0, 0.0, 0.0))
    b = track("cam2", 8, 8, 12, vec(0.99, 0.08, 0.0))
    mapping = m.associate([a, b])
    assert mapping[a.key] == mapping[b.key]


def test_different_appearance_creates_new_identity():
    m = manager()
    a = track("cam1", 1, 0, 5, vec(1.0, 0.0, 0.0))
    b = track("cam2", 8, 8, 12, vec(0.0, 1.0, 0.0))
    mapping = m.associate([a, b])
    assert mapping[a.key] != mapping[b.key]


def test_topology_rejects_impossible_transition():
    topology = CameraTopology(
        {"cam1": {"cam2": TransitionWindow(min_seconds=10, max_seconds=30)}},
        strict=True,
    )
    m = manager(topology=topology)
    a = track("cam1", 1, 0, 5, vec(1.0, 0.0))
    b = track("cam2", 2, 7, 10, vec(1.0, 0.0))
    mapping = m.associate([a, b])
    assert mapping[a.key] != mapping[b.key]


def test_large_overlap_does_not_merge_simultaneous_people():
    m = manager()
    a = track("cam1", 1, 0, 10, vec(1.0, 0.0))
    b = track("cam2", 2, 2, 7, vec(1.0, 0.0))
    mapping = m.associate([a, b])
    assert mapping[a.key] != mapping[b.key]
