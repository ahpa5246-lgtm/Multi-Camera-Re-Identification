from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .identity import Tracklet


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tracklets (
    run_id INTEGER NOT NULL,
    camera_id TEXT NOT NULL,
    local_track_id INTEGER NOT NULL,
    global_id INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    frame_count INTEGER NOT NULL,
    mean_confidence REAL NOT NULL,
    thumbnail_path TEXT,
    PRIMARY KEY (run_id, camera_id, local_track_id)
);

CREATE TABLE IF NOT EXISTS observations (
    run_id INTEGER NOT NULL,
    camera_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    local_track_id INTEGER NOT NULL,
    global_id INTEGER NOT NULL,
    x1 INTEGER NOT NULL,
    y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL,
    y2 INTEGER NOT NULL,
    confidence REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_observations_global
ON observations(run_id, global_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_observations_camera
ON observations(run_id, camera_id, frame_index);
"""


class EventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        self.connection.close()

    def create_run(self, config: dict) -> int:
        cursor = self.connection.execute(
            "INSERT INTO runs(config_json) VALUES (?)",
            (json.dumps(config, sort_keys=True),),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def save_tracklets(self, run_id: int, tracklets: Iterable[Tracklet]) -> None:
        rows = [
            (
                run_id, t.camera_id, t.local_track_id, int(t.global_id or -1),
                t.start_time, t.end_time, t.frame_count, t.mean_confidence, t.thumbnail_path,
            )
            for t in tracklets
        ]
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO tracklets(
                run_id, camera_id, local_track_id, global_id,
                start_time, end_time, frame_count, mean_confidence, thumbnail_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.connection.commit()

    def save_observations(self, run_id: int, observations: Iterable[dict]) -> None:
        rows = [
            (
                run_id, o["camera_id"], o["frame_index"], o["timestamp"],
                o["local_track_id"], o["global_id"],
                o["bbox"][0], o["bbox"][1], o["bbox"][2], o["bbox"][3],
                o["confidence"],
            )
            for o in observations
        ]
        self.connection.executemany(
            """
            INSERT INTO observations(
                run_id, camera_id, frame_index, timestamp,
                local_track_id, global_id, x1, y1, x2, y2, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.connection.commit()
