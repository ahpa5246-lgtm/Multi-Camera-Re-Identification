from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from mcreid.config import PipelineConfig
from mcreid.pipeline import MultiCameraPipeline, RunResult
from mcreid.topology import CameraTopology


st.set_page_config(page_title="ARGUS", page_icon="◉", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.08);
    padding: 14px 16px;
    border-radius: 12px;
}
.argus-sub {opacity:.68;font-size:.95rem;margin-top:-.6rem;margin-bottom:1.3rem;}
</style>
""", unsafe_allow_html=True)

st.title("ARGUS")
st.markdown(
    '<div class="argus-sub">Multi-camera visual tracking and temporary re-identification for authorized footage.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Pipeline")
    detector_model = st.text_input("Detector", "yolo11n.pt")
    reid_model = st.text_input("Appearance model", "vit_small_patch14_dinov2.lvd142m")
    det_conf = st.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
    reid_threshold = st.slider("Re-ID similarity", 0.40, 0.95, 0.72, 0.01)
    min_margin = st.slider("Ambiguity margin", 0.0, 0.20, 0.04, 0.01)
    embedding_interval = st.slider("Embedding every N frames", 1, 30, 8)
    device = st.selectbox("Compute device", ["auto", "cpu", "0"])

uploads = st.file_uploader(
    "Upload synchronized camera recordings",
    type=["mp4", "mov", "mkv", "avi", "m4v"],
    accept_multiple_files=True,
)

camera_specs: list[tuple[str, object, float]] = []
if uploads:
    st.subheader("Cameras")
    cols = st.columns(min(3, len(uploads)))
    seen: set[str] = set()
    for index, upload in enumerate(uploads):
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(upload.name).stem).strip("_")
        default_id = stem or f"cam{index + 1:02d}"
        with cols[index % len(cols)]:
            camera_id = st.text_input(
                f"Camera ID — {upload.name}",
                value=default_id,
                key=f"camera_name_{index}_{upload.name}",
            ).strip()
            offset = st.number_input(
                f"Time offset (s) — {upload.name}",
                value=0.0,
                step=0.5,
                key=f"offset_{index}_{upload.name}",
            )
        if camera_id and camera_id not in seen:
            seen.add(camera_id)
            camera_specs.append((camera_id, upload, float(offset)))
        elif camera_id in seen:
            st.error(f"Duplicate camera ID: {camera_id}")

can_process = len(camera_specs) == len(uploads or []) and len(camera_specs) > 0

if st.button("Process recordings", type="primary", disabled=not can_process):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"streamlit_{stamp}"
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    camera_paths: dict[str, Path] = {}
    offsets: dict[str, float] = {}
    for camera_id, upload, offset in camera_specs:
        suffix = Path(upload.name).suffix or ".mp4"
        path = input_dir / f"{camera_id}{suffix}"
        path.write_bytes(upload.getbuffer())
        camera_paths[camera_id] = path
        offsets[camera_id] = offset

    config = PipelineConfig(
        detector_model=detector_model,
        reid_model=reid_model,
        detection_confidence=det_conf,
        reid_similarity_threshold=reid_threshold,
        reid_min_margin=min_margin,
        embedding_interval=embedding_interval,
        device=device,
        camera_time_offsets=offsets,
    )

    status = st.status("Initializing models…", expanded=True)
    progress = st.progress(0.0)
    camera_order = list(camera_paths)
    camera_index = {camera: i for i, camera in enumerate(camera_order)}

    def report(camera: str, frame: int, total: int) -> None:
        local = (frame / total) if total else 0.0
        overall = (camera_index[camera] + local) / max(1, len(camera_order))
        progress.progress(min(1.0, overall))
        status.write(f"{camera}: processed {frame}" + (f" / {total} frames" if total else " frames"))

    try:
        result = MultiCameraPipeline(config, CameraTopology()).run(
            camera_paths, run_dir, progress_callback=report
        )
        progress.progress(1.0)
        status.update(label="Processing complete", state="complete", expanded=False)
        st.session_state["argus_result"] = result
    except Exception as exc:
        status.update(label="Processing failed", state="error")
        st.exception(exc)

result: RunResult | None = st.session_state.get("argus_result")
if result:
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cameras", len(result.annotated_videos))
    m2.metric("Tracklets", len(result.tracklets))
    m3.metric("Global IDs", result.global_identity_count)
    m4.metric("Elapsed", f"{result.elapsed_seconds:.1f}s")

    tab_live, tab_ids, tab_events, tab_run = st.tabs(
        ["Camera outputs", "Identity explorer", "Event timeline", "Run files"]
    )

    with tab_live:
        for camera_id, path in result.annotated_videos.items():
            st.subheader(camera_id)
            st.video(str(path))

    tracklet_df = pd.DataFrame([{
        "global_id": int(t.global_id or -1),
        "camera": t.camera_id,
        "local_track": t.local_track_id,
        "start_s": round(t.start_time, 2),
        "end_s": round(t.end_time, 2),
        "frames": t.frame_count,
        "mean_confidence": round(t.mean_confidence, 3),
        "thumbnail": t.thumbnail_path,
    } for t in result.tracklets])

    with tab_ids:
        if tracklet_df.empty:
            st.info("No person tracklets were produced.")
        else:
            ids = sorted(tracklet_df["global_id"].unique().tolist())
            selected = st.selectbox("Global identity", ids, format_func=lambda x: f"G{x:04d}")
            subset = tracklet_df[tracklet_df["global_id"] == selected]
            st.dataframe(subset.drop(columns=["thumbnail"]), use_container_width=True, hide_index=True)
            thumbs = [Path(p) for p in subset["thumbnail"].dropna().tolist() if Path(p).exists()]
            if thumbs:
                columns = st.columns(min(5, len(thumbs)))
                for i, thumb in enumerate(thumbs):
                    columns[i % len(columns)].image(str(thumb), caption=thumb.stem)

    with tab_events:
        events = pd.DataFrame([{
            "camera": o["camera_id"],
            "frame": o["frame_index"],
            "time_s": round(o["timestamp"], 3),
            "global_id": o["global_id"],
            "local_track": o["local_track_id"],
            "confidence": round(o["confidence"], 3),
        } for o in result.observations])
        if events.empty:
            st.info("No events recorded.")
        else:
            selected_ids = st.multiselect(
                "Filter global IDs", sorted(events["global_id"].unique().tolist())
            )
            if selected_ids:
                events = events[events["global_id"].isin(selected_ids)]
            st.dataframe(events.head(5000), use_container_width=True, hide_index=True)
            if len(events) > 5000:
                st.caption("Showing 5,000 rows. The complete event stream is in SQLite.")

    with tab_run:
        st.code(str(result.output_dir))
        st.write("SQLite database:", str(result.database_path))
        st.caption(
            "Global IDs are temporary appearance-based associations, not claims about real identity."
        )
