from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .pipeline import MultiCameraPipeline
from .topology import CameraTopology


def _pairs(values: list[str], cast=str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {value}")
        key, raw = value.split("=", 1)
        if not key:
            raise argparse.ArgumentTypeError("Camera ID cannot be empty.")
        parsed[key] = cast(raw)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARGUS multi-camera person re-identification")
    parser.add_argument("--input", action="append", required=True, metavar="CAMERA=VIDEO")
    parser.add_argument("--offset", action="append", default=[], metavar="CAMERA=SECONDS")
    parser.add_argument("--output", default="runs/latest")
    parser.add_argument("--topology", default=None)
    parser.add_argument("--strict-topology", action="store_true")
    parser.add_argument("--detector-model", default="yolo11n.pt")
    parser.add_argument("--reid-model", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--det-confidence", type=float, default=0.35)
    parser.add_argument("--reid-threshold", type=float, default=0.72)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cameras = {k: Path(v) for k, v in _pairs(args.input).items()}
    offsets = {k: float(v) for k, v in _pairs(args.offset, float).items()}

    missing = [str(path) for path in cameras.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing input video(s): {', '.join(missing)}")

    config = PipelineConfig(
        detector_model=args.detector_model,
        reid_model=args.reid_model,
        detection_confidence=args.det_confidence,
        reid_similarity_threshold=args.reid_threshold,
        strict_topology=args.strict_topology,
        device=args.device,
        camera_time_offsets=offsets,
    )
    topology = (
        CameraTopology.from_yaml(args.topology, strict=args.strict_topology)
        if args.topology
        else CameraTopology(strict=args.strict_topology)
    )

    def progress(camera: str, frame: int, total: int) -> None:
        suffix = f"/{total}" if total else ""
        print(f"\r[{camera}] frame {frame}{suffix}", end="", flush=True)

    result = MultiCameraPipeline(config, topology).run(
        cameras, args.output, progress_callback=progress
    )
    print()
    print(json.dumps({
        "run_id": result.run_id,
        "output_dir": str(result.output_dir),
        "database": str(result.database_path),
        "cameras": {k: str(v) for k, v in result.annotated_videos.items()},
        "tracklets": len(result.tracklets),
        "global_identities": result.global_identity_count,
        "elapsed_seconds": round(result.elapsed_seconds, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
