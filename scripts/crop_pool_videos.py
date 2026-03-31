from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pool_geometry import (
    apply_pool_crop,
    detect_pool_geometry,
    load_scene_config,
    relative_polygon,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_videos" / "swim_vids"
UNDISTORTED_DIR = ROOT / "dataset" / "videos_undistorted"
OUT_DIR = ROOT / "dataset" / "videos_cropped"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop swimming pool videos to the detected blue water area.")
    parser.add_argument(
        "--input",
        dest="inputs",
        type=Path,
        action="append",
        help="Specific input video path. Repeat the flag to process multiple videos.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory with source videos. Defaults to dataset/videos_undistorted when available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory where cropped videos will be written.",
    )
    return parser.parse_args()


def resolve_input_dir(candidate: Path | None) -> Path:
    if candidate is not None:
        return candidate
    if any(UNDISTORTED_DIR.glob("*.mp4")):
        return UNDISTORTED_DIR
    return RAW_DIR


def iter_video_paths(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        return [path.resolve() for path in args.inputs]
    return sorted(resolve_input_dir(args.input_dir).glob("*.mp4"))


def save_geometry_debug(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_scene_config()
    video_paths = iter_video_paths(args)
    if not video_paths:
        raise FileNotFoundError("No input videos found for pool cropping.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video_path in video_paths:
        geometry = detect_pool_geometry(video_path, config)
        output_size = (geometry.bbox_xywh[2], geometry.bbox_xywh[3])
        output_path = args.output_dir / f"{video_path.stem}_pool.mp4"
        geometry_path = args.output_dir / f"{video_path.stem}_pool_geometry.json"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            output_size,
        )

        debug_payload = {
            "input_video": video_path.name,
            "binary_mask_bbox_xywh": list(geometry.bbox_xywh),
            "detected_bbox_xywh": list(geometry.bbox_xywh),
            "source_size": list(geometry.source_size),
            "source_polygon": geometry.polygon,
            "output_polygon": relative_polygon(geometry, output_size),
            "output_size": list(output_size),
        }
        save_geometry_debug(geometry_path, debug_payload)

        print(f"[crop] {video_path.name} -> {output_path.name}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                cropped = apply_pool_crop(frame, geometry, output_size=output_size, mask_outside_pool=True)
                writer.write(cropped)
        finally:
            cap.release()
            writer.release()


if __name__ == "__main__":
    main()
