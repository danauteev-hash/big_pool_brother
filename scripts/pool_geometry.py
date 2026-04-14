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


@dataclass
class UndistortPlan:
    input_size: tuple[int, int]
    crop_xywh: tuple[int, int, int, int]
    map1: np.ndarray
    map2: np.ndarray
    camera_matrix: np.ndarray
    new_camera_matrix: np.ndarray

    @property
    def output_size(self) -> tuple[int, int]:
        return int(self.crop_xywh[2]), int(self.crop_xywh[3])


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


def fisheye_fov_scale(config: dict) -> float:
    return float(config["fisheye"].get("fov_scale", 1.0))


def fisheye_scale(config: dict) -> float:
    return float(config["fisheye"].get("scale", 1.0))


def fisheye_auto_crop(config: dict) -> bool:
    return bool(config["fisheye"].get("auto_crop", False))


def fisheye_crop_margin_px(config: dict) -> int:
    return max(0, int(config["fisheye"].get("crop_margin_px", 0)))


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


def largest_valid_rect(binary_mask: np.ndarray) -> tuple[int, int, int, int]:
    height, width = binary_mask.shape
    heights = np.zeros(width, dtype=np.int32)

    best_x, best_y, best_width, best_height = 0, 0, width, height
    best_area = 0

    for y_coord in range(height):
        row = binary_mask[y_coord] > 0
        heights[row] += 1
        heights[~row] = 0

        stack: list[int] = []
        for x_coord in range(width + 1):
            current_height = heights[x_coord] if x_coord < width else 0
            while stack and current_height < heights[stack[-1]]:
                top = stack.pop()
                rect_height = heights[top]
                if rect_height == 0:
                    continue
                left = stack[-1] + 1 if stack else 0
                rect_width = x_coord - left
                area = rect_width * rect_height
                if area > best_area:
                    best_area = area
                    best_x = left
                    best_y = y_coord - rect_height + 1
                    best_width = rect_width
                    best_height = rect_height
            stack.append(x_coord)

    return best_x, best_y, best_width, best_height


def compute_auto_crop_roi(
    map1: np.ndarray,
    map2: np.ndarray,
    dim: tuple[int, int],
    margin_px: int = 0,
) -> tuple[int, int, int, int]:
    width, height = dim
    white_mask = np.full((height, width), 255, dtype=np.uint8)
    valid = cv2.remap(
        white_mask,
        map1,
        map2,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    x_coord, y_coord, roi_width, roi_height = largest_valid_rect(valid)
    margin_px = max(0, int(margin_px))
    if margin_px > 0:
        x_coord += margin_px
        y_coord += margin_px
        roi_width -= margin_px * 2
        roi_height -= margin_px * 2

    roi_width = max(2, roi_width - (roi_width % 2))
    roi_height = max(2, roi_height - (roi_height % 2))
    x_coord = int(np.clip(x_coord, 0, max(0, width - roi_width)))
    y_coord = int(np.clip(y_coord, 0, max(0, height - roi_height)))
    return x_coord, y_coord, roi_width, roi_height


def build_undistort_plan(dim: tuple[int, int], config: dict | None = None) -> UndistortPlan:
    config = config or load_scene_config()
    reference_w, reference_h = reference_video_dim(config)
    K = fisheye_camera_matrix(config)
    D = fisheye_distortion(config)
    fisheye = config["fisheye"]

    scaled_K = K.copy()
    scaled_K[0, :] *= dim[0] / reference_w
    scaled_K[1, :] *= dim[1] / reference_h
    scaled_K[2, 2] = 1.0

    if "balance" in fisheye:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            scaled_K,
            D,
            dim,
            np.eye(3),
            balance=fisheye_balance(config),
            new_size=dim,
            fov_scale=fisheye_fov_scale(config),
        )
    else:
        new_K = scaled_K.copy()
        new_K[0, 0] *= fisheye_scale(config)
        new_K[1, 1] *= fisheye_scale(config)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_K,
        D,
        np.eye(3),
        new_K,
        dim,
        cv2.CV_16SC2,
    )

    if fisheye_auto_crop(config):
        crop_xywh = compute_auto_crop_roi(
            map1,
            map2,
            dim,
            margin_px=fisheye_crop_margin_px(config),
        )
    else:
        crop_xywh = (0, 0, int(dim[0]), int(dim[1]))

    return UndistortPlan(
        input_size=(int(dim[0]), int(dim[1])),
        crop_xywh=crop_xywh,
        map1=map1,
        map2=map2,
        camera_matrix=scaled_K,
        new_camera_matrix=new_K,
    )


def build_undistort_maps(dim: tuple[int, int], config: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    plan = build_undistort_plan(dim, config)
    return plan.map1, plan.map2


def apply_undistort(
    frame_bgr: np.ndarray,
    plan: UndistortPlan,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
) -> np.ndarray:
    corrected = cv2.remap(
        frame_bgr,
        plan.map1,
        plan.map2,
        interpolation=interpolation,
        borderMode=border_mode,
    )
    x_coord, y_coord, width, height = plan.crop_xywh
    return corrected[y_coord : y_coord + height, x_coord : x_coord + width]


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


def scale_polygon(
    polygon: list[list[int]],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[list[int]]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Polygon source size must be positive.")

    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return [
        [
            int(round(point_x * scale_x)),
            int(round(point_y * scale_y)),
        ]
        for point_x, point_y in polygon
    ]


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
