from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "pool_scene.json"


@dataclass
class PoolGeometry:
    bbox_xywh: tuple[int, int, int, int]
    polygon: list[list[int]]
    source_size: tuple[int, int]


def load_scene_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def target_crop_size(config: dict) -> tuple[int, int]:
    crop = config["crop"]
    return int(crop["width"]), int(crop["height"])


def reference_video_dim(config: dict) -> tuple[int, int]:
    fisheye = config["fisheye"]
    if "reference_dim" in fisheye:
        width, height = fisheye["reference_dim"]
        return int(width), int(height)
    return int(fisheye["width"]), int(fisheye["height"])


def fisheye_balance(config: dict) -> float:
    fisheye = config["fisheye"]
    if "balance" in fisheye:
        return float(fisheye["balance"])
    return float(fisheye.get("scale", 0.0))


def fisheye_camera_matrix(config: dict) -> np.ndarray:
    fisheye = config["fisheye"]
    if "K" in fisheye:
        return np.array(fisheye["K"], dtype=np.float64)
    return np.array(
        [
            [float(fisheye["fx"]), 0.0, float(fisheye["cx"])],
            [0.0, float(fisheye["fy"]), float(fisheye["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def fisheye_distortion(config: dict) -> np.ndarray:
    fisheye = config["fisheye"]
    if "D" in fisheye:
        return np.array(fisheye["D"], dtype=np.float64).reshape(4, 1)
    return np.array(
        [
            float(fisheye.get("k1", 0.0)),
            float(fisheye.get("k2", 0.0)),
            float(fisheye.get("k3", 0.0)),
            float(fisheye.get("k4", 0.0)),
        ],
        dtype=np.float64,
    ).reshape(4, 1)


def build_undistort_maps(dim: tuple[int, int], config: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    config = config or load_scene_config()
    reference_w, reference_h = reference_video_dim(config)
    K = fisheye_camera_matrix(config)
    D = fisheye_distortion(config)
    balance = fisheye_balance(config)

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


def odd_kernel_size(size: int) -> int:
    size = max(1, int(size))
    return size if size % 2 == 1 else size + 1


def detect_water_component(frame_bgr: np.ndarray, config: dict) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(config["water_hsv"]["lower"], dtype=np.uint8)
    upper = np.array(config["water_hsv"]["upper"], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    crop_detection = config.get("crop_detection", {})
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            odd_kernel_size(crop_detection.get("open_kernel", 9)),
            odd_kernel_size(crop_detection.get("open_kernel", 9)),
        ),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            odd_kernel_size(crop_detection.get("close_kernel", 31)),
            odd_kernel_size(crop_detection.get("close_kernel", 31)),
        ),
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return np.zeros(mask.shape, dtype=np.uint8)

    frame_area = float(frame_bgr.shape[0] * frame_bgr.shape[1])
    min_ratio = float(crop_detection.get("min_component_area_ratio", 0.05))
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    largest_idx = int(np.argmax(areas))
    largest_area = float(areas[largest_idx])
    if largest_area < frame_area * min_ratio:
        return np.zeros(mask.shape, dtype=np.uint8)

    label_idx = largest_idx + 1
    return np.uint8(labels == label_idx) * 255


def sample_video_frames(
    video_path: Path,
    sample_count: int,
    preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sample_count = max(1, int(sample_count))
    frame_positions = np.linspace(0, max(0, total_frames - 1), num=sample_count, dtype=np.int32)

    frames: list[np.ndarray] = []
    try:
        for frame_index in sorted(set(int(value) for value in frame_positions.tolist())):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(preprocess(frame) if preprocess else frame)
    finally:
        cap.release()
    return frames


def mask_bounding_box(mask: np.ndarray, padding: int = 0) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Binary mask does not contain any foreground pixels.")

    x, y, width, height = cv2.boundingRect(points)
    height_limit, width_limit = mask.shape
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width_limit, x + width + padding)
    y2 = min(height_limit, y + height + padding)
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def geometry_from_consensus_masks(masks: list[np.ndarray], config: dict) -> PoolGeometry:
    if not masks:
        raise ValueError("Cannot build pool geometry without masks.")

    crop_detection = config.get("crop_detection", {})
    vote_threshold = float(crop_detection.get("vote_threshold", 0.6))
    padding = int(crop_detection.get("padding_px", 0))
    epsilon_ratio = float(crop_detection.get("polygon_epsilon_ratio", 0.015))

    stack = np.stack([(mask > 0).astype(np.float32) for mask in masks], axis=0)
    consensus = (stack.mean(axis=0) >= vote_threshold).astype(np.uint8) * 255
    if not np.any(consensus):
        consensus = masks[0]

    contours, _ = cv2.findContours(consensus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No pool contour found in consensus mask.")
    contour = max(contours, key=cv2.contourArea)

    filled_mask = np.zeros_like(consensus)
    cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)
    x1, y1, width, height = mask_bounding_box(filled_mask, padding=padding)

    epsilon = max(1.0, epsilon_ratio * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    polygon = [[int(point[0]), int(point[1])] for point in approx.tolist()]

    return PoolGeometry(
        bbox_xywh=(int(x1), int(y1), int(width), int(height)),
        polygon=polygon,
        source_size=(int(consensus.shape[1]), int(consensus.shape[0])),
    )


def fallback_geometry(config: dict, source_size: tuple[int, int] | None = None) -> PoolGeometry:
    crop = config["crop"]
    polygon = [[int(x), int(y)] for x, y in config["pool_polygon"]]
    if source_size is None:
        return PoolGeometry(
            bbox_xywh=(int(crop["x"]), int(crop["y"]), int(crop["width"]), int(crop["height"])),
            polygon=polygon,
            source_size=(int(crop["width"]), int(crop["height"])),
        )

    ref_width, ref_height = reference_video_dim(config)
    scale_x = source_size[0] / ref_width
    scale_y = source_size[1] / ref_height
    x = int(round(crop["x"] * scale_x))
    y = int(round(crop["y"] * scale_y))
    width = int(round(crop["width"] * scale_x))
    height = int(round(crop["height"] * scale_y))
    scaled_polygon = []
    for px, py in polygon:
        scaled_polygon.append(
            [
                int(round((crop["x"] + px) * scale_x)),
                int(round((crop["y"] + py) * scale_y)),
            ]
        )
    return PoolGeometry(
        bbox_xywh=(x, y, width, height),
        polygon=scaled_polygon,
        source_size=source_size,
    )


def detect_pool_geometry(
    video_path: Path,
    config: dict | None = None,
    preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
) -> PoolGeometry:
    config = config or load_scene_config()
    crop_detection = config.get("crop_detection", {})
    frames = sample_video_frames(video_path, crop_detection.get("sample_frames", 7), preprocess=preprocess)
    if not frames:
        return fallback_geometry(config)

    masks = [detect_water_component(frame, config) for frame in frames]
    masks = [mask for mask in masks if np.any(mask)]
    if not masks:
        return fallback_geometry(config, source_size=(frames[0].shape[1], frames[0].shape[0]))
    return geometry_from_consensus_masks(masks, config)


def relative_polygon(geometry: PoolGeometry, output_size: tuple[int, int] | None = None) -> list[list[int]]:
    x, y, width, height = geometry.bbox_xywh
    polygon = np.array(geometry.polygon, dtype=np.float32)
    polygon[:, 0] -= x
    polygon[:, 1] -= y

    if output_size is not None and (width, height) != output_size:
        scale_x = output_size[0] / width
        scale_y = output_size[1] / height
        polygon[:, 0] *= scale_x
        polygon[:, 1] *= scale_y

    return [[int(round(px)), int(round(py))] for px, py in polygon.tolist()]


def apply_pool_crop(
    frame_bgr: np.ndarray,
    geometry: PoolGeometry,
    output_size: tuple[int, int] | None = None,
    mask_outside_pool: bool = True,
) -> np.ndarray:
    x, y, width, height = geometry.bbox_xywh
    x = max(0, min(x, frame_bgr.shape[1] - 1))
    y = max(0, min(y, frame_bgr.shape[0] - 1))
    width = max(1, min(width, frame_bgr.shape[1] - x))
    height = max(1, min(height, frame_bgr.shape[0] - y))

    if mask_outside_pool:
        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(geometry.polygon, dtype=np.int32)], 255)
        frame_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

    cropped = frame_bgr[y : y + height, x : x + width]
    if output_size is not None and (cropped.shape[1], cropped.shape[0]) != output_size:
        cropped = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
    return cropped
