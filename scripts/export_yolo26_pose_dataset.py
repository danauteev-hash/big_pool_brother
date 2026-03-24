from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".cache" / "ultralytics"))

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
SOURCE_IMAGES_DIR = DATASET_DIR / "images"
POSE_DIR = DATASET_DIR / "yolo26_pose"
POSE_IMAGES_DIR = POSE_DIR / "images"
POSE_LABELS_DIR = POSE_DIR / "labels"
POSE_VIZ_DIR = POSE_DIR / "visualizations"

POSE_MODEL_CANDIDATES = [
    ROOT / "models" / "yolo26n-pose.pt",
    ROOT / "yolo26n-pose.pt",
]

POSE_IMGSZ = 640
POSE_CONF = 0.05
POSE_RETRY_CONF = 0.02
PRIMARY_PAD = 0.75
RETRY_PAD = 1.2
MIN_VISIBLE_KEYPOINTS = 4
MIN_POSE_SCORE = 0.05
MIN_IOU = 0.01
KEYPOINT_CONF = 0.3
POSE_CLASS_NAME = "swimmer"
COCO17_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
COCO17_SKELETON = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]


@dataclass
class CropInfo:
    target_box: list[float]
    origin_x: int
    origin_y: int


@dataclass
class PoseInstance:
    bbox_xyxy: list[float]
    keypoints: list[tuple[float, float, int]]


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_pose_model_path() -> Path:
    for candidate in POSE_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "YOLO26 pose checkpoint not found. Expected one of: "
        + ", ".join(str(path) for path in POSE_MODEL_CANDIDATES)
    )


def coco_bbox_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def bbox_xyxy_to_yolo(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return center_x, center_y, box_width, box_height


def iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def clip_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        float(np.clip(x1, 0, max(0, width - 1))),
        float(np.clip(y1, 0, max(0, height - 1))),
        float(np.clip(x2, 1, width)),
        float(np.clip(y2, 1, height)),
    ]


def make_crop(image_rgb: np.ndarray, box_xyxy: list[float], pad: float) -> tuple[np.ndarray, CropInfo]:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    crop_x1 = max(int(round(x1 - box_w * pad)), 0)
    crop_y1 = max(int(round(y1 - box_h * pad)), 0)
    crop_x2 = min(int(round(x2 + box_w * pad)), width)
    crop_y2 = min(int(round(y2 + box_h * pad)), height)
    crop = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop, CropInfo(target_box=box_xyxy, origin_x=crop_x1, origin_y=crop_y1)


def select_pose(result, crop_info: CropInfo, image_width: int, image_height: int) -> PoseInstance | None:
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None or len(result.keypoints) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    keypoints = result.keypoints.data.cpu().numpy()

    target_box = crop_info.target_box
    target_center_x = (target_box[0] + target_box[2]) / 2.0
    target_center_y = (target_box[1] + target_box[3]) / 2.0
    target_diag = max(np.hypot(target_box[2] - target_box[0], target_box[3] - target_box[1]), 1.0)

    best_instance: PoseInstance | None = None
    best_score = -1.0

    for box, pose_score, pose_keypoints in zip(boxes, scores, keypoints):
        global_box = [
            float(box[0] + crop_info.origin_x),
            float(box[1] + crop_info.origin_y),
            float(box[2] + crop_info.origin_x),
            float(box[3] + crop_info.origin_y),
        ]
        global_box = clip_box(global_box, image_width, image_height)
        overlap = iou_xyxy(global_box, target_box)

        pred_center_x = (global_box[0] + global_box[2]) / 2.0
        pred_center_y = (global_box[1] + global_box[3]) / 2.0
        center_distance = np.hypot(pred_center_x - target_center_x, pred_center_y - target_center_y) / target_diag

        visible_keypoints: list[tuple[float, float, int]] = []
        visible_count = 0
        for kp_x, kp_y, kp_conf in pose_keypoints:
            global_x = float(np.clip(kp_x + crop_info.origin_x, 0, image_width - 1))
            global_y = float(np.clip(kp_y + crop_info.origin_y, 0, image_height - 1))
            visible = 2 if float(kp_conf) >= KEYPOINT_CONF else 0
            if visible:
                visible_count += 1
                visible_keypoints.append((global_x, global_y, visible))
            else:
                visible_keypoints.append((0.0, 0.0, 0))

        rank_score = overlap + (0.35 / (1.0 + center_distance)) + float(pose_score) * 0.1
        if visible_count < MIN_VISIBLE_KEYPOINTS:
            continue
        if float(pose_score) < MIN_POSE_SCORE and overlap < MIN_IOU:
            continue
        if rank_score > best_score:
            best_score = rank_score
            best_instance = PoseInstance(bbox_xyxy=global_box, keypoints=visible_keypoints)

    return best_instance


def format_pose_label(instance: PoseInstance, image_width: int, image_height: int) -> str:
    cx, cy, w, h = bbox_xyxy_to_yolo(instance.bbox_xyxy, image_width, image_height)
    values = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"]
    for kp_x, kp_y, visible in instance.keypoints:
        if visible:
            values.append(f"{kp_x / image_width:.6f} {kp_y / image_height:.6f} {visible}")
        else:
            values.append("0.000000 0.000000 0")
    return " ".join(values)


def draw_pose_visualization(image_rgb: np.ndarray, poses: list[PoseInstance]) -> Image.Image:
    image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image)

    for pose in poses:
        x1, y1, x2, y2 = pose.bbox_xyxy
        draw.rectangle((x1, y1, x2, y2), outline=(255, 196, 0), width=2)

        for start_idx, end_idx in COCO17_SKELETON:
            x_start, y_start, v_start = pose.keypoints[start_idx]
            x_end, y_end, v_end = pose.keypoints[end_idx]
            if v_start and v_end:
                draw.line((x_start, y_start, x_end, y_end), fill=(48, 223, 164), width=3)

        for kp_x, kp_y, visible in pose.keypoints:
            if not visible:
                continue
            radius = 3
            draw.ellipse((kp_x - radius, kp_y - radius, kp_x + radius, kp_y + radius), fill=(255, 90, 90))

    return image


def write_data_yaml() -> None:
    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "",
        "kpt_shape: [17, 3]",
        f"flip_idx: {COCO17_FLIP_IDX}",
        "",
        "names:",
        f"  0: {POSE_CLASS_NAME}",
        "",
    ]
    (POSE_DIR / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def update_metadata(summary: dict) -> None:
    metadata_path = DATASET_DIR / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {}
    metadata["yolo26_pose"] = summary
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_split(split: str, model: YOLO) -> dict[str, int]:
    coco = json.loads((ANNOTATIONS_DIR / f"instances_{split}.json").read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    split_image_dir = POSE_IMAGES_DIR / split
    split_label_dir = POSE_LABELS_DIR / split
    split_viz_dir = POSE_VIZ_DIR / split
    split_image_dir.mkdir(parents=True, exist_ok=True)
    split_label_dir.mkdir(parents=True, exist_ok=True)
    split_viz_dir.mkdir(parents=True, exist_ok=True)

    images_with_pose = 0
    pose_annotations = 0

    for image_meta in coco["images"]:
        source_path = SOURCE_IMAGES_DIR / split / image_meta["file_name"]
        target_path = split_image_dir / image_meta["file_name"]
        shutil.copy2(source_path, target_path)

        image_rgb = np.array(Image.open(source_path).convert("RGB"))
        image_height, image_width = image_rgb.shape[:2]

        annotations = annotations_by_image.get(image_meta["id"], [])
        crops: list[np.ndarray] = []
        crop_infos: list[CropInfo] = []
        for annotation in annotations:
            crop, crop_info = make_crop(image_rgb, coco_bbox_to_xyxy(annotation["bbox"]), PRIMARY_PAD)
            crops.append(crop)
            crop_infos.append(crop_info)

        results = []
        if crops:
            results = model(crops, verbose=False, imgsz=POSE_IMGSZ, conf=POSE_CONF, device="cpu")

        pose_instances: list[PoseInstance] = []
        for annotation, crop_info, result in zip(annotations, crop_infos, results):
            pose = select_pose(result, crop_info, image_width, image_height)
            if pose is None:
                retry_crop, retry_crop_info = make_crop(image_rgb, coco_bbox_to_xyxy(annotation["bbox"]), RETRY_PAD)
                retry_result = model(
                    retry_crop,
                    verbose=False,
                    imgsz=POSE_IMGSZ,
                    conf=POSE_RETRY_CONF,
                    device="cpu",
                )[0]
                pose = select_pose(retry_result, retry_crop_info, image_width, image_height)
            if pose is not None:
                pose_instances.append(pose)

        label_path = split_label_dir / f"{Path(image_meta['file_name']).stem}.txt"
        label_lines = [format_pose_label(pose, image_width, image_height) for pose in pose_instances]
        label_path.write_text("\n".join(label_lines), encoding="utf-8")

        if pose_instances:
            images_with_pose += 1
            pose_annotations += len(pose_instances)

        viz_image = draw_pose_visualization(image_rgb, pose_instances)
        viz_image.save(split_viz_dir / image_meta["file_name"], quality=90)

    return {
        "images": len(coco["images"]),
        "images_with_pose": images_with_pose,
        "pose_annotations": pose_annotations,
        "label_files": len(coco["images"]),
    }


def main() -> None:
    ensure_clean_dir(POSE_DIR)
    ensure_clean_dir(POSE_IMAGES_DIR)
    ensure_clean_dir(POSE_LABELS_DIR)
    ensure_clean_dir(POSE_VIZ_DIR)

    model_path = resolve_pose_model_path()
    model = YOLO(model_path)

    train_summary = build_split("train", model)
    val_summary = build_split("val", model)
    write_data_yaml()

    summary = {
        "model_checkpoint": str(model_path),
        "dataset_dir": str(POSE_DIR),
        "visualization_dir": str(POSE_VIZ_DIR),
        "train_images": train_summary["images"],
        "train_images_with_pose": train_summary["images_with_pose"],
        "train_pose_annotations": train_summary["pose_annotations"],
        "train_label_files": train_summary["label_files"],
        "val_images": val_summary["images"],
        "val_images_with_pose": val_summary["images_with_pose"],
        "val_pose_annotations": val_summary["pose_annotations"],
        "val_label_files": val_summary["label_files"],
    }
    update_metadata(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
