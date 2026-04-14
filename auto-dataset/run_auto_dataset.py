from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().with_name("config.json")
SCENE_CONFIG_PATH = ROOT / "config" / "pool_scene.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the repo-local auto-dataset pipeline.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to the auto-dataset JSON config.")
    parser.add_argument("--video", type=Path, default=None, help="Optional override for the source video path.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def repo_relative_str(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def run_step(command: list[str]) -> None:
    print(f"[auto-dataset] running: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def sync_scene_config(config: dict) -> None:
    scene_config = json.loads(SCENE_CONFIG_PATH.read_text(encoding="utf-8"))
    for key in (
        "segmentation_backend",
        "sam3_checkpoint",
        "allow_box_fallback",
        "sam3_load_from_hf",
        "sam3_device",
        "sam3_compile",
    ):
        if key in config:
            scene_config[key] = config[key]
    if "sample_fps" in config:
        scene_config["sample_fps"] = config["sample_fps"]
    for key in ("balance", "fov_scale", "auto_crop", "crop_margin_px", "scale"):
        if key in config:
            scene_config.setdefault("fisheye", {})
            scene_config["fisheye"][key] = config[key]
    SCENE_CONFIG_PATH.write_text(json.dumps(scene_config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    python_exe = sys.executable
    configured_video = args.video if args.video is not None else str(config.get("source_video", "")).strip()
    if not configured_video:
        raise ValueError(
            "Source video is not configured. Put it under auto-dataset/input_videos/ or pass --video <path>."
        )
    video_path = configured_video if isinstance(configured_video, Path) else resolve_repo_path(configured_video)
    if not video_path.exists():
        raise FileNotFoundError(
            f"Source video not found: {video_path}. Put it under auto-dataset/input_videos/ or pass --video <path>."
        )
    sync_scene_config(config)

    if bool(config.get("rebuild_base_dataset", True)):
        raw_video_dir = resolve_repo_path(config.get("raw_video_dir", "data/raw_videos/swim_vids"))
        if not raw_video_dir.exists():
            raise FileNotFoundError(f"Configured raw_video_dir does not exist: {raw_video_dir}")

        run_step(
            [
                python_exe,
                "scripts/undistort_fisheye_videos.py",
                "--input-dir",
                str(raw_video_dir),
            ]
        )
        run_step([python_exe, "scripts/crop_pool_videos.py"])
        run_step([python_exe, "scripts/build_autolabeled_dataset.py"])

        if bool(config.get("rebuild_yolov8_exports", True)):
            run_step([python_exe, "scripts/export_yolov8_labels.py"])
            run_step([python_exe, "scripts/build_yolov8_augmented_dataset.py"])

    run_step(
        [
            python_exe,
            "scripts/extend_dataset_from_video.py",
            "--video",
            str(video_path),
            "--video-id",
            config["video_id"],
            "--sample-fps",
            str(config["sample_fps"]),
        ]
    )
    run_step([python_exe, "scripts/build_yolov26_pose_dataset.py"])

    summary = {
        "source_video": repo_relative_str(video_path),
        "sample_fps": config["sample_fps"],
        "rebuild_base_dataset": bool(config.get("rebuild_base_dataset", True)),
        "raw_video_dir": repo_relative_str(resolve_repo_path(config.get("raw_video_dir", "data/raw_videos/swim_vids"))),
        "pose_dataset_dir": repo_relative_str(resolve_repo_path(config["pose_dataset_dir"])),
        "config_path": repo_relative_str(config_path),
    }
    (Path(__file__).resolve().with_name("last_run_summary.json")).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
