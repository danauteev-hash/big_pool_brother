from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
YOLO_DIR = DATASET_DIR / "yolov8"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def coco_bbox_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    center_x = (x + w / 2.0) / width
    center_y = (y + h / 2.0) / height
    norm_w = w / width
    norm_h = h / height
    return center_x, center_y, norm_w, norm_h


def export_split(split: str) -> None:
    coco = json.loads((ANNOTATIONS_DIR / f"instances_{split}.json").read_text(encoding="utf-8"))
    labels_dir = YOLO_DIR / "labels" / split
    ensure_dir(labels_dir)

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    for image in coco["images"]:
        label_path = labels_dir / f"{Path(image['file_name']).stem}.txt"
        lines: list[str] = []
        for annotation in annotations_by_image.get(image["id"], []):
            cx, cy, w, h = coco_bbox_to_yolo(annotation["bbox"], image["width"], image["height"])
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        label_path.write_text("\n".join(lines), encoding="utf-8")


def write_data_yaml() -> None:
    yaml_path = YOLO_DIR / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "path: ..",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                "  0: swimmer",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    export_split("train")
    export_split("val")
    write_data_yaml()
    print(f"YOLOv8 labels exported to {YOLO_DIR}")


if __name__ == "__main__":
    main()
