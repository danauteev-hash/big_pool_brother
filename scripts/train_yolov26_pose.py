from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))

from ultralytics import YOLO
DEFAULT_DATA_PATH = ROOT / "dataset" / "yolov26_pose" / "data.yaml"
DEFAULT_PROJECT_DIR = ROOT / "artifacts" / "train"
DEFAULT_MODEL_CANDIDATES = [
    ROOT / "models" / "yolo26n-pose.pt",
    ROOT / "models" / "yolo26s-pose.pt",
]


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def resolve_model_path(path_like: str | None) -> Path:
    if path_like:
        path = resolve_repo_path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        return path

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "YOLOv26 pose checkpoint not found. Expected one of: "
        + ", ".join(str(path) for path in DEFAULT_MODEL_CANDIDATES)
    )


def repo_relative_str(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv26 pose model on the local swimmer pose dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to dataset/yolov26_pose/data.yaml")
    parser.add_argument("--model", default=None, help="Optional checkpoint override.")
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="cpu", help="Ultralytics device string, for example cpu or 0.")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers. Keep 0 on Windows if unstable.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT_DIR, help="Directory for training runs.")
    parser.add_argument("--name", default="yolov26_pose", help="Run name inside the project directory.")
    parser.add_argument(
        "--cache",
        default="false",
        choices=("false", "disk", "ram"),
        help="Ultralytics cache mode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_repo_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model_path = resolve_model_path(args.model)
    project_dir = resolve_repo_path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    cache_arg: bool | str
    if args.cache == "false":
        cache_arg = False
    else:
        cache_arg = args.cache

    model = YOLO(str(model_path))
    train_kwargs = {
        "data": str(data_path),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": args.device,
        "workers": int(args.workers),
        "patience": int(args.patience),
        "project": str(project_dir),
        "name": args.name,
        "cache": cache_arg,
        "seed": int(args.seed),
    }

    model.train(**train_kwargs)

    run_dir = project_dir / args.name
    (run_dir / "train_args.json").write_text(
        json.dumps(
            {
                "model": repo_relative_str(model_path),
                "data": repo_relative_str(data_path),
                "project": repo_relative_str(project_dir),
                "train_kwargs": train_kwargs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model": repo_relative_str(model_path),
                "data": repo_relative_str(data_path),
                "run_dir": repo_relative_str(run_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
