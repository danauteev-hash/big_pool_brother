from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pool_geometry import apply_pool_crop, build_undistort_maps, detect_pool_geometry, load_scene_config, target_crop_size


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
TRAIN_JSON = ANNOTATIONS_DIR / "instances_train.json"
VAL_JSON = ANNOTATIONS_DIR / "instances_val.json"
CONFIG = load_scene_config()


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = load_module("pool_builder", ROOT / "scripts" / "build_autolabeled_dataset.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend the swimmer dataset from a single full-length video.")
    parser.add_argument("--video", type=Path, required=True, help="Path to the source video.")
    parser.add_argument("--video-id", default="full_pool_video", help="Stable video id used in the dataset.")
    parser.add_argument("--split", default="train", choices=["train"], help="Dataset split to extend.")
    parser.add_argument("--sample-fps", type=float, default=4.0, help="Target frame sampling rate.")
    return parser.parse_args()


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_coco(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def preprocess_frame(
    frame_bgr: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    geometry,
    output_size: tuple[int, int],
) -> np.ndarray:
    corrected = cv2.remap(
        frame_bgr,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    cropped = apply_pool_crop(corrected, geometry, output_size=output_size, mask_outside_pool=False)
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)


def sampled_frame_indices(total_frames: int, video_fps: float, sample_fps: float) -> set[int]:
    sample_fps = max(0.1, sample_fps)
    frame_interval = max(1, int(round(video_fps / sample_fps)))
    return set(range(0, total_frames, frame_interval))


def remove_existing_video_entries(coco: dict, split_dir: Path, video_id: str) -> dict:
    kept_images = []
    removed_ids: set[int] = set()
    for image in coco["images"]:
        if image.get("video_id") == video_id:
            removed_ids.add(image["id"])
            image_path = split_dir / image["file_name"]
            if image_path.exists():
                image_path.unlink()
            continue
        kept_images.append(image)

    coco["images"] = kept_images
    coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] not in removed_ids]
    return coco


def next_id(items: list[dict]) -> int:
    return max((item["id"] for item in items), default=0) + 1


def main() -> None:
    args = parse_args()
    video_path = args.video.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    split_dir = IMAGES_DIR / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    train_coco = remove_existing_video_entries(load_coco(TRAIN_JSON), split_dir, args.video_id)
    val_coco = load_coco(VAL_JSON)
    image_id = next_id(train_coco["images"])
    annotation_id = next_id(train_coco["annotations"])

    processor, detector, segmenter, segmenter_name = BUILDER.load_models()
    polygon = np.array(CONFIG["pool_polygon"], dtype=np.int32)
    pool_mask = BUILDER.polygon_mask((CONFIG["crop"]["height"], CONFIG["crop"]["width"]), CONFIG["pool_polygon"])
    core_pool_mask = BUILDER.build_core_pool_mask(pool_mask)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(round(fps * 1))
    map1, map2 = build_undistort_maps(
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
        CONFIG,
    )
    geometry = detect_pool_geometry(
        video_path,
        CONFIG,
        preprocess=lambda frame: cv2.remap(
            frame,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        ),
    )
    output_size = target_crop_size(CONFIG)
    target_indices = sampled_frame_indices(total_frames, fps, args.sample_fps)

    frame_index = 0
    saved_images = 0
    saved_annotations = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index not in target_indices:
                frame_index += 1
                continue

            frame_rgb = preprocess_frame(frame_bgr, map1, map2, geometry, output_size)
            image = Image.fromarray(frame_rgb)
            detections = BUILDER.detect_people(image, processor, detector, polygon)
            refined = BUILDER.filter_candidates(
                BUILDER.refine_with_segmenter(frame_rgb, detections, segmenter, segmenter_name),
                frame_rgb.shape,
                pool_mask,
                core_pool_mask,
                BUILDER.build_water_mask(frame_rgb, pool_mask),
            )
            if not refined:
                frame_index += 1
                continue

            filename = f"{args.video_id}_{saved_images:04d}.jpg"
            BUILDER.save_rgb(frame_rgb, split_dir / filename)
            train_coco["images"].append(
                {
                    "id": image_id,
                    "file_name": filename,
                    "width": int(frame_rgb.shape[1]),
                    "height": int(frame_rgb.shape[0]),
                    "video_id": args.video_id,
                    "frame_index": frame_index,
                }
            )

            for candidate in refined:
                if not candidate.segments:
                    continue
                train_coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": BUILDER.CATEGORY["id"],
                        "bbox": BUILDER.bbox_xyxy_to_coco(candidate.bbox_xyxy),
                        "area": round(float(candidate.area), 2),
                        "iscrowd": 0,
                        "segmentation": candidate.segments,
                    }
                )
                annotation_id += 1
                saved_annotations += 1

            image_id += 1
            saved_images += 1
            frame_index += 1
    finally:
        cap.release()

    save_coco(TRAIN_JSON, train_coco)

    metadata = json.loads((DATASET_DIR / "metadata.json").read_text(encoding="utf-8"))
    metadata["train_images"] = len(train_coco["images"])
    metadata["train_annotations"] = len(train_coco["annotations"])
    metadata["val_images"] = len(val_coco["images"])
    metadata["val_annotations"] = len(val_coco["annotations"])
    metadata["extended_videos"] = metadata.get("extended_videos", [])
    metadata["extended_videos"] = [item for item in metadata["extended_videos"] if item.get("video_id") != args.video_id]
    metadata["extended_videos"].append(
        {
            "video_id": args.video_id,
            "source_file": video_path.name,
            "sample_fps": args.sample_fps,
            "saved_images": saved_images,
            "saved_annotations": saved_annotations,
            "fps": round(float(fps), 4),
            "duration_seconds": round(total_frames / fps, 2) if fps else None,
            "segmentation_backend": segmenter_name,
        }
    )
    (DATASET_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "video_id": args.video_id,
                "saved_images": saved_images,
                "saved_annotations": saved_annotations,
                "train_images": len(train_coco["images"]),
                "train_annotations": len(train_coco["annotations"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
