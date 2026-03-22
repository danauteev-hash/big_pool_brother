from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "pool_scene.json").read_text(encoding="utf-8"))
RAW_DIR = ROOT / "data" / "raw_videos" / "swim_vids"
OUT_DIR = ROOT / "dataset" / "videos_undistorted"


def build_maps(dim: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    reference_w, reference_h = CONFIG["fisheye"]["reference_dim"]
    K = np.array(CONFIG["fisheye"]["K"], dtype=np.float64)
    D = np.array(CONFIG["fisheye"]["D"], dtype=np.float64).reshape(4, 1)
    balance = float(CONFIG["fisheye"]["balance"])

    scaled_K = K.copy()
    scaled_K[0, :] *= dim[0] / reference_w
    scaled_K[1, :] *= dim[1] / reference_h
    scaled_K[2, 2] = 1.0

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        scaled_K,
        D,
        dim,
        np.eye(3),
        balance=balance,
    )
    return cv2.fisheye.initUndistortRectifyMap(
        scaled_K,
        D,
        np.eye(3),
        new_K,
        dim,
        cv2.CV_16SC2,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for video_path in sorted(RAW_DIR.glob("*.mp4")):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        map1, map2 = build_maps((width, height))

        output_path = OUT_DIR / f"{video_path.stem}_undistorted.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        print(f"[undistort] {video_path.name} -> {output_path.name}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                corrected = cv2.remap(
                    frame,
                    map1,
                    map2,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                writer.write(corrected)
        finally:
            cap.release()
            writer.release()


if __name__ == "__main__":
    main()
