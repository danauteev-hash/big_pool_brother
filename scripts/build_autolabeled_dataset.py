from __future__ import annotations

import json
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

warnings.filterwarnings("ignore", message="A NumPy version")

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

import sys

sys.path.insert(0, str(ROOT / "vendor" / "FastSAM"))
from fastsam import FastSAM, FastSAMPrompt


CONFIG = json.loads((ROOT / "config" / "pool_scene.json").read_text(encoding="utf-8"))
VIDEOS_DIR = ROOT / "dataset" / "videos_cropped"
DATASET_DIR = ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
RTDETR_DIR = ROOT / "models" / "rtdetr_r18vd_coco_o365"
FASTSAM_WEIGHTS = ROOT / "models" / "FastSAM-s.pt"
CATEGORY = {"id": 1, "name": "swimmer", "supercategory": "person"}


@dataclass
class Detection:
    bbox_xyxy: list[float]
    score: float


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def polygon_mask(shape: tuple[int, int], polygon: list[list[int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def contour_to_coco(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < CONFIG["min_mask_area"]:
            continue
        epsilon = 0.0025 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        flat = approx.reshape(-1, 2).astype(float).flatten().tolist()
        if len(flat) >= 6:
            segments.append(flat)
    return segments


def bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max())
    y2 = float(ys.max())
    return [x1, y1, x2, y2]


def bbox_area_xyxy(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_xyxy_to_coco(box: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)]


def iou_xyxy(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area_xyxy(box_a) + bbox_area_xyxy(box_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def load_models() -> tuple[AutoImageProcessor, AutoModelForObjectDetection, FastSAM]:
    processor = AutoImageProcessor.from_pretrained(RTDETR_DIR, use_fast=False)
    model = AutoModelForObjectDetection.from_pretrained(RTDETR_DIR)
    model.eval()
    fastsam_model = FastSAM(str(FASTSAM_WEIGHTS))
    return processor, model, fastsam_model


def detect_people(
    image: Image.Image,
    processor: AutoImageProcessor,
    model: AutoModelForObjectDetection,
    polygon: np.ndarray,
) -> list[Detection]:
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([[image.height, image.width]]),
        threshold=CONFIG["rtdetr_threshold"],
    )[0]

    detections: list[Detection] = []
    labels = model.config.id2label
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        if labels[int(label)] != "person":
            continue
        box_xyxy = [float(x) for x in box.tolist()]
        if bbox_area_xyxy(box_xyxy) < CONFIG["min_box_area"]:
            continue
        center = ((box_xyxy[0] + box_xyxy[2]) / 2.0, (box_xyxy[1] + box_xyxy[3]) / 2.0)
        if not point_in_polygon(center, polygon):
            continue
        detections.append(Detection(bbox_xyxy=box_xyxy, score=float(score)))
    return detections


def refine_with_fastsam(
    image_rgb: np.ndarray,
    detections: list[Detection],
    fastsam_model: FastSAM,
) -> list[tuple[list[float], list[list[float]], float, float]]:
    if not detections:
        return []

    everything = fastsam_model(
        image_rgb,
        device="cpu",
        retina_masks=True,
        imgsz=1024,
        conf=CONFIG["fastsam_conf"],
        iou=CONFIG["fastsam_iou"],
        verbose=False,
    )
    prompt = FastSAMPrompt(image_rgb, everything, device="cpu")
    refined: list[tuple[list[float], list[list[float]], float, float]] = []

    for detection in detections:
        box = [int(round(v)) for v in detection.bbox_xyxy]
        masks = prompt.box_prompt(bboxes=[box])
        if len(masks) > 0:
            mask = masks[0].astype(np.uint8)
            if mask.sum() >= CONFIG["min_mask_area"]:
                refined_box = bbox_from_mask(mask)
                segments = contour_to_coco(mask)
                if refined_box and segments:
                    refined.append((refined_box, segments, float(mask.sum()), detection.score))
                    continue

        fallback_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = box
        fallback_mask[y1:y2, x1:x2] = 1
        refined.append(
            (
                detection.bbox_xyxy,
                contour_to_coco(fallback_mask),
                bbox_area_xyxy(detection.bbox_xyxy),
                detection.score,
            )
        )
    return refined


def deduplicate_annotations(
    refined: list[tuple[list[float], list[list[float]], float, float]]
) -> list[tuple[list[float], list[list[float]], float, float]]:
    kept: list[tuple[list[float], list[list[float]], float, float]] = []
    for candidate in sorted(refined, key=lambda item: item[3], reverse=True):
        candidate_box = candidate[0]
        if any(iou_xyxy(candidate_box, existing[0]) >= CONFIG["dedupe_iou_threshold"] for existing in kept):
            continue
        kept.append(candidate)
    return kept


def save_rgb(image_rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(image_rgb).save(path, quality=95)


def sampled_frames(video_path: Path, sample_fps: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps / sample_fps)))
    frame_idx = 0
    sample_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_interval == 0:
                yield sample_idx, frame
                sample_idx += 1
            frame_idx += 1
    finally:
        cap.release()


def build_dataset() -> None:
    ensure_clean_dir(IMAGES_DIR / "train")
    ensure_clean_dir(IMAGES_DIR / "val")
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    processor, detector, fastsam_model = load_models()
    polygon = np.array(CONFIG["pool_polygon"], dtype=np.int32)

    splits = {
        "train": {"images": [], "annotations": [], "count": 0},
        "val": {"images": [], "annotations": [], "count": 0},
    }
    image_id = 1
    annotation_id = 1

    video_paths = sorted(VIDEOS_DIR.glob("*_undistorted_pool.mp4")) or sorted(VIDEOS_DIR.glob("*.mp4"))
    for video_path in video_paths:
        stem = video_path.stem.replace("_pool", "").replace("_undistorted", "")
        split = "val" if stem in CONFIG["val_videos"] else "train"
        print(f"[dataset] {video_path.name} -> {split}")

        for sample_idx, frame_bgr in sampled_frames(video_path, CONFIG["sample_fps"]):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            filename = f"{video_path.stem}_{sample_idx:04d}.jpg"
            save_rgb(frame_rgb, IMAGES_DIR / split / filename)

            detections = detect_people(image, processor, detector, polygon)
            refined = deduplicate_annotations(refine_with_fastsam(frame_rgb, detections, fastsam_model))

            splits[split]["images"].append(
                {
                    "id": image_id,
                    "file_name": filename,
                    "width": int(frame_rgb.shape[1]),
                    "height": int(frame_rgb.shape[0]),
                    "video_id": video_path.stem,
                    "frame_index": sample_idx,
                }
            )

            for bbox_xyxy, segments, area, _score in refined:
                if not segments:
                    continue
                splits[split]["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": CATEGORY["id"],
                        "bbox": bbox_xyxy_to_coco(bbox_xyxy),
                        "area": round(float(area), 2),
                        "iscrowd": 0,
                        "segmentation": segments,
                    }
                )
                annotation_id += 1

            splits[split]["count"] += len(refined)
            image_id += 1

    for split_name, payload in splits.items():
        coco = {
            "info": {
                "description": "Auto-labeled swimmer dataset generated from cropped pool videos with RT-DETR + FastSAM",
            },
            "licenses": [],
            "images": payload["images"],
            "annotations": payload["annotations"],
            "categories": [CATEGORY],
        }
        (ANNOTATIONS_DIR / f"instances_{split_name}.json").write_text(
            json.dumps(coco, indent=2),
            encoding="utf-8",
        )

    stats = {
        "train_images": len(splits["train"]["images"]),
        "val_images": len(splits["val"]["images"]),
        "train_annotations": len(splits["train"]["annotations"]),
        "val_annotations": len(splits["val"]["annotations"]),
        "sample_fps": CONFIG["sample_fps"],
        "val_videos": CONFIG["val_videos"],
    }
    (DATASET_DIR / "metadata.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    build_dataset()
