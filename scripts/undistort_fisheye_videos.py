from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pool_geometry import apply_undistort, build_undistort_plan, load_scene_config


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_videos" / "swim_vids"
OUT_DIR = ROOT / "dataset" / "videos_undistorted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Undistort swimming pool videos with OpenCV fisheye remapping.")
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
        default=RAW_DIR,
        help="Directory with source videos when --input is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory where undistorted videos will be written.",
    )
    return parser.parse_args()


def iter_video_paths(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        return [path.resolve() for path in args.inputs]
    return sorted(args.input_dir.glob("*.mp4"))


def main() -> None:
    args = parse_args()
    config = load_scene_config()
    video_paths = iter_video_paths(args)
    if not video_paths:
        raise FileNotFoundError("No input videos found for fisheye correction.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        plan = build_undistort_plan((width, height), config)

        output_path = args.output_dir / f"{video_path.stem}_undistorted.mp4"
        debug_path = args.output_dir / f"{video_path.stem}_undistorted_plan.json"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            plan.output_size,
        )
        debug_path.write_text(
            json.dumps(
                {
                    "input_video": video_path.name,
                    "input_size": [width, height],
                    "output_size": list(plan.output_size),
                    "crop_xywh": [int(value) for value in plan.crop_xywh],
                    "camera_matrix": [[float(value) for value in row] for row in plan.camera_matrix.tolist()],
                    "new_camera_matrix": [[float(value) for value in row] for row in plan.new_camera_matrix.tolist()],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"[undistort] {video_path.name} -> {output_path.name}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                corrected = apply_undistort(frame, plan)
                writer.write(corrected)
        finally:
            cap.release()
            writer.release()


if __name__ == "__main__":
    main()
