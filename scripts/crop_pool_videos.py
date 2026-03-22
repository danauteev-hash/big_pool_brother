from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "pool_scene.json").read_text(encoding="utf-8"))
FFMPEG = ROOT / "tools" / "ffmpeg" / "ffmpeg.exe"
RAW_DIR = ROOT / "data" / "raw_videos" / "swim_vids"
UNDISTORTED_DIR = ROOT / "dataset" / "videos_undistorted"
OUT_DIR = ROOT / "dataset" / "videos_cropped"


def main() -> None:
    crop = CONFIG["crop"]
    crop_filter = f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_dir = UNDISTORTED_DIR if any(UNDISTORTED_DIR.glob("*.mp4")) else RAW_DIR
    for video_path in sorted(input_dir.glob("*.mp4")):
        output_path = OUT_DIR / f"{video_path.stem}_pool.mp4"
        cmd = [
            str(FFMPEG),
            "-y",
            "-i",
            str(video_path),
            "-vf",
            crop_filter,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "copy",
            str(output_path),
        ]
        print(f"[crop] {video_path.name} -> {output_path.name}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
