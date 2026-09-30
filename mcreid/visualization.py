from __future__ import annotations

import colorsys
from collections import defaultdict
from pathlib import Path

import cv2


def _color_for_id(global_id: int) -> tuple[int, int, int]:
    hue = (global_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_identity(frame, bbox, global_id: int, confidence: float):
    x1, y1, x2, y2 = bbox
    color = _color_for_id(global_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"G{global_id:04d}  {confidence:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    )
    top = max(0, y1 - text_h - baseline - 7)
    cv2.rectangle(frame, (x1, top), (x1 + text_w + 10, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 5, y1 - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA,
    )
    return frame


def render_annotated_video(
    input_path: str | Path,
    output_path: str | Path,
    observations: list[dict],
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for observation in observations:
        by_frame[int(observation["frame_index"])].append(observation)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for observation in by_frame.get(frame_index, []):
            draw_identity(
                frame,
                observation["bbox"],
                int(observation["global_id"]),
                float(observation["confidence"]),
            )
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    return output_path
