from mcreid.identity import Tracklet
from mcreid.storage import EventStore


def test_event_store_round_trip_counts(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    run_id = store.create_run({"hello": "world"})

    t = Tracklet(
        camera_id="cam1",
        local_track_id=3,
        start_time=0.0,
        end_time=1.0,
        embedding=None,
        frame_count=2,
        mean_confidence=0.91,
        global_id=7,
    )
    store.save_tracklets(run_id, [t])
    store.save_observations(
        run_id,
        [{
            "camera_id": "cam1",
            "frame_index": 1,
            "timestamp": 0.04,
            "local_track_id": 3,
            "global_id": 7,
            "bbox": (1, 2, 30, 90),
            "confidence": 0.91,
        }],
    )

    tracklet_count = store.connection.execute(
        "SELECT COUNT(*) FROM tracklets WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    event_count = store.connection.execute(
        "SELECT COUNT(*) FROM observations WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    store.close()

    assert tracklet_count == 1
    assert event_count == 1
