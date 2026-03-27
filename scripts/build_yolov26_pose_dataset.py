from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".cache" / "ultralytics"))

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
SOURCE_IMAGES_DIR = DATASET_DIR / "images"
POSE_DIR = DATASET_DIR / "yolov26_pose"
POSE_IMAGES_DIR = POSE_DIR / "images"
POSE_LABELS_DIR = POSE_DIR / "labels"
POSE_VIZ_DIR = POSE_DIR / "visualizations"
LABEL_STUDIO_DIR = POSE_DIR / "label_studio"

POSE_MODEL_CANDIDATES = [
    ROOT / "models" / "yolo26n-pose.pt",
    ROOT / "models" / "yolo26s-pose.pt",
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


def repo_relative_str(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


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
        "YOLOv26 pose checkpoint not found. Expected one of: "
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


def yolo_bbox_to_xyxy(cx: float, cy: float, w: float, h: float, width: int, height: int) -> list[float]:
    box_w = w * width
    box_h = h * height
    center_x = cx * width
    center_y = cy * height
    return [
        center_x - box_w / 2.0,
        center_y - box_h / 2.0,
        center_x + box_w / 2.0,
        center_y + box_h / 2.0,
    ]


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


def parse_pose_label(path: Path) -> list[dict]:
    if not path.exists():
        return []
    annotations: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        entry = {
            "cls": int(float(parts[0])),
            "cx": float(parts[1]),
            "cy": float(parts[2]),
            "w": float(parts[3]),
            "h": float(parts[4]),
            "keypoints": [],
        }
        for idx in range(5, len(parts), 3):
            entry["keypoints"].append(
                {
                    "x": float(parts[idx]),
                    "y": float(parts[idx + 1]),
                    "v": int(float(parts[idx + 2])),
                }
            )
        annotations.append(entry)
    return annotations


def save_pose_labels(path: Path, annotations: list[dict]) -> None:
    lines: list[str] = []
    for annotation in annotations:
        values = [
            str(int(annotation["cls"])),
            f"{annotation['cx']:.6f}",
            f"{annotation['cy']:.6f}",
            f"{annotation['w']:.6f}",
            f"{annotation['h']:.6f}",
        ]
        for keypoint in annotation["keypoints"]:
            values.extend(
                [
                    f"{keypoint['x']:.6f}",
                    f"{keypoint['y']:.6f}",
                    str(int(keypoint["v"])),
                ]
            )
        lines.append(" ".join(values))
    path.write_text("\n".join(lines), encoding="utf-8")


def draw_pose_visualization(image_rgb: np.ndarray, annotations: list[dict]) -> Image.Image:
    image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size

    for annotation in annotations:
        x1, y1, x2, y2 = yolo_bbox_to_xyxy(
            annotation["cx"],
            annotation["cy"],
            annotation["w"],
            annotation["h"],
            image_width,
            image_height,
        )
        draw.rectangle((x1, y1, x2, y2), outline=(255, 196, 0), width=2)

        keypoints = annotation["keypoints"]
        for start_idx, end_idx in COCO17_SKELETON:
            start = keypoints[start_idx]
            end = keypoints[end_idx]
            if start["v"] and end["v"]:
                draw.line(
                    (
                        start["x"] * image_width,
                        start["y"] * image_height,
                        end["x"] * image_width,
                        end["y"] * image_height,
                    ),
                    fill=(48, 223, 164),
                    width=3,
                )

        for keypoint in keypoints:
            if not keypoint["v"]:
                continue
            px = keypoint["x"] * image_width
            py = keypoint["y"] * image_height
            radius = 3
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(255, 90, 90))

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


def write_label_studio_config() -> None:
    LABEL_STUDIO_DIR.mkdir(parents=True, exist_ok=True)
    (LABEL_STUDIO_DIR / "label_config.xml").write_text(
        "\n".join(
            [
                "<View>",
                '  <Image name="image" value="$image"/>',
                '  <RectangleLabels name="bbox" toName="image">',
                '    <Label value="swimmer" background="#F4B400"/>',
                "  </RectangleLabels>",
                '  <KeyPointLabels name="kp" toName="image" strokeColor="#30DFA4">',
                '    <Label value="swimmer" background="#30DFA4"/>',
                "  </KeyPointLabels>",
                "</View>",
            ]
        ),
        encoding="utf-8",
    )


def update_metadata(summary: dict) -> None:
    metadata_path = DATASET_DIR / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {}
    metadata.pop("yolo26_pose", None)
    metadata["yolov26_pose"] = summary
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def rng_for_name(name: str) -> np.random.Generator:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(seed)


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image_bgr: np.ndarray, quality: int = 95) -> None:
    success, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(path)


def augment_pose_labels(annotations: list[dict], flip_horizontal: bool) -> list[dict]:
    augmented: list[dict] = []
    for annotation in annotations:
        updated = {
            "cls": annotation["cls"],
            "cx": annotation["cx"],
            "cy": annotation["cy"],
            "w": annotation["w"],
            "h": annotation["h"],
            "keypoints": [dict(keypoint) for keypoint in annotation["keypoints"]],
        }

        if flip_horizontal:
            updated["cx"] = 1.0 - updated["cx"]
            remapped = [updated["keypoints"][index].copy() for index in COCO17_FLIP_IDX]
            for keypoint in remapped:
                if keypoint["v"]:
                    keypoint["x"] = 1.0 - keypoint["x"]
            updated["keypoints"] = remapped

        augmented.append(updated)
    return augmented


def augment_image(image_bgr: np.ndarray, labels: list[dict], rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    image = image_bgr.copy()
    flip_horizontal = rng.random() < 0.45
    annotations = augment_pose_labels(labels, flip_horizontal)

    if flip_horizontal:
        image = cv2.flip(image, 1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6.0, 6.0)) % 180.0
    hsv[..., 1] *= rng.uniform(0.82, 1.18)
    hsv[..., 2] *= rng.uniform(0.80, 1.20)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    alpha = rng.uniform(0.88, 1.14)
    beta = rng.uniform(-10.0, 12.0)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if rng.random() < 0.35:
        image = cv2.GaussianBlur(image, (5, 5), sigmaX=rng.uniform(0.3, 1.1))

    if rng.random() < 0.40:
        noise = rng.normal(0.0, rng.uniform(3.0, 8.0), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.25:
        sharpen = np.array([[0, -1, 0], [-1, 5.2, -1], [0, -1, 0]], dtype=np.float32)
        image = cv2.filter2D(image, -1, sharpen)

    return image, annotations


def save_visualization(path: Path, image_bgr: np.ndarray, annotations: list[dict]) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    draw_pose_visualization(image_rgb, annotations).save(path, quality=90)


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

        parsed_annotations = parse_pose_label(label_path)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        save_visualization(split_viz_dir / image_meta["file_name"], image_bgr, parsed_annotations)

        if parsed_annotations:
            images_with_pose += 1
            pose_annotations += len(parsed_annotations)

    return {
        "images": len(coco["images"]),
        "images_with_pose": images_with_pose,
        "pose_annotations": pose_annotations,
        "label_files": len(coco["images"]),
    }


def build_augmented_train() -> int:
    out_train_images = POSE_IMAGES_DIR / "train"
    out_train_labels = POSE_LABELS_DIR / "train"
    out_train_viz = POSE_VIZ_DIR / "train"
    generated = 0

    for image_path in sorted(out_train_images.glob("*.jpg")):
        if "__aug" in image_path.stem:
            continue
        label_path = out_train_labels / f"{image_path.stem}.txt"
        labels = parse_pose_label(label_path)
        if not labels:
            continue

        image = read_image(image_path)
        if image is None:
            continue

        rng = rng_for_name(image_path.stem)
        augmented_image, augmented_labels = augment_image(image, labels, rng)

        out_image_path = out_train_images / f"{image_path.stem}__aug01.jpg"
        out_label_path = out_train_labels / f"{image_path.stem}__aug01.txt"
        out_viz_path = out_train_viz / f"{image_path.stem}__aug01.jpg"

        write_image(out_image_path, augmented_image)
        save_pose_labels(out_label_path, augmented_labels)
        save_visualization(out_viz_path, augmented_image, augmented_labels)
        generated += 1

    return generated


def build_label_studio_bundle() -> None:
    write_label_studio_config()
    for split in ("train", "val"):
        images_dir = POSE_IMAGES_DIR / split
        labels_dir = POSE_LABELS_DIR / split
        coco = {
            "info": {"description": f"YOLOv26 pose Label Studio bundle ({split})"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": POSE_CLASS_NAME,
                    "supercategory": "person",
                    "keypoints": [f"kp_{idx}" for idx in range(17)],
                    "skeleton": [[a + 1, b + 1] for a, b in COCO17_SKELETON],
                }
            ],
        }
        image_id = 1
        annotation_id = 1

        for image_path in sorted(images_dir.glob("*.jpg")):
            if "__aug" in image_path.stem:
                continue
            image = Image.open(image_path)
            image_width, image_height = image.size
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": image_width,
                    "height": image_height,
                }
            )

            labels = parse_pose_label(labels_dir / f"{image_path.stem}.txt")
            for label in labels:
                bbox = yolo_bbox_to_xyxy(label["cx"], label["cy"], label["w"], label["h"], image_width, image_height)
                keypoints: list[float] = []
                visible = 0
                for keypoint in label["keypoints"]:
                    keypoints.extend(
                        [
                            round(keypoint["x"] * image_width, 2),
                            round(keypoint["y"] * image_height, 2),
                            keypoint["v"],
                        ]
                    )
                    if keypoint["v"]:
                        visible += 1

                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [
                            round(bbox[0], 2),
                            round(bbox[1], 2),
                            round(bbox[2] - bbox[0], 2),
                            round(bbox[3] - bbox[1], 2),
                        ],
                        "area": round(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]), 2),
                        "iscrowd": 0,
                        "num_keypoints": visible,
                        "keypoints": keypoints,
                    }
                )
                annotation_id += 1

            image_id += 1

        (LABEL_STUDIO_DIR / f"coco_keypoints_{split}.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")


def main() -> None:
    ensure_clean_dir(POSE_DIR)
    ensure_clean_dir(POSE_IMAGES_DIR)
    ensure_clean_dir(POSE_LABELS_DIR)
    ensure_clean_dir(POSE_VIZ_DIR)
    ensure_clean_dir(LABEL_STUDIO_DIR)

    model_path = resolve_pose_model_path()
    model = YOLO(model_path)

    train_summary = build_split("train", model)
    val_summary = build_split("val", model)
    generated_augmented = build_augmented_train()
    build_label_studio_bundle()
    write_data_yaml()

    summary = {
        "model_checkpoint": repo_relative_str(model_path),
        "dataset_dir": repo_relative_str(POSE_DIR),
        "visualization_dir": repo_relative_str(POSE_VIZ_DIR),
        "label_studio_dir": repo_relative_str(LABEL_STUDIO_DIR),
        "original_train_images": train_summary["images"],
        "augmented_train_images": generated_augmented,
        "total_train_images": train_summary["images"] + generated_augmented,
        "train_images_with_pose": train_summary["images_with_pose"],
        "train_pose_annotations": train_summary["pose_annotations"],
        "train_label_files": len(list((POSE_LABELS_DIR / "train").glob("*.txt"))),
        "val_images": val_summary["images"],
        "val_images_with_pose": val_summary["images_with_pose"],
        "val_pose_annotations": val_summary["pose_annotations"],
        "val_label_files": val_summary["label_files"],
        "augmentation_policy": "offline_hsv_brightness_blur_noise_sharpen_optional_horizontal_flip",
    }
    update_metadata(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
