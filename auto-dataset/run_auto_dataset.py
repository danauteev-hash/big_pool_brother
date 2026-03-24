from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().with_name("config.json")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def run_step(command: list[str]) -> None:
    print(f"[auto-dataset] running: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    config = load_config()
    python_exe = sys.executable

    run_step(
        [
            python_exe,
            "scripts/extend_dataset_from_video.py",
            "--video",
            config["source_video"],
            "--video-id",
            config["video_id"],
            "--sample-fps",
            str(config["sample_fps"]),
        ]
    )
    run_step([python_exe, "scripts/build_yolov26_pose_dataset.py"])

    summary = {
        "source_video": config["source_video"],
        "sample_fps": config["sample_fps"],
        "pose_dataset_dir": str(ROOT / config["pose_dataset_dir"]),
        "config_path": str(CONFIG_PATH),
    }
    (Path(__file__).resolve().with_name("last_run_summary.json")).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
