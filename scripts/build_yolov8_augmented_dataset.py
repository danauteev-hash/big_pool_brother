from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
SRC_IMAGES = DATASET_DIR / "images"
SRC_LABELS = DATASET_DIR / "yolov8" / "labels"
OUT_DIR = DATASET_DIR / "yolov8_augmented"


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_split(split: str) -> None:
    out_images = OUT_DIR / "images" / split
    out_labels = OUT_DIR / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for image_path in sorted((SRC_IMAGES / split).glob("*.jpg")):
        shutil.copy2(image_path, out_images / image_path.name)

    for label_path in sorted((SRC_LABELS / split).glob("*.txt")):
        shutil.copy2(label_path, out_labels / label_path.name)


def load_yolo_label(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    parsed: list[list[float]] = []
    for line in lines:
        parts = line.split()
        parsed.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return parsed


def save_yolo_label(path: Path, annotations: list[list[float]]) -> None:
    lines = [
        f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for cls, cx, cy, w, h in annotations
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def rng_for_name(name: str) -> np.random.Generator:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(seed)


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray, quality: int = 95) -> None:
    success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(path)


def augment_image(image_bgr: np.ndarray, labels: list[list[float]], rng: np.random.Generator) -> tuple[np.ndarray, list[list[float]]]:
    image = image_bgr.copy()
    annotations = [label[:] for label in labels]

    if rng.random() < 0.45:
        image = cv2.flip(image, 1)
        for ann in annotations:
            ann[1] = 1.0 - ann[1]

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
        noisy = image.astype(np.float32) + noise
        image = np.clip(noisy, 0, 255).astype(np.uint8)

    if rng.random() < 0.25:
        sharpen = np.array([[0, -1, 0], [-1, 5.2, -1], [0, -1, 0]], dtype=np.float32)
        image = cv2.filter2D(image, -1, sharpen)

    return image, annotations


def write_data_yaml() -> None:
    (OUT_DIR / "data.yaml").write_text(
        "\n".join(
            [
                "path: .",
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


def build_augmented_train() -> int:
    src_train_images = SRC_IMAGES / "train"
    out_train_images = OUT_DIR / "images" / "train"
    out_train_labels = OUT_DIR / "labels" / "train"
    generated = 0

    for image_path in sorted(src_train_images.glob("*.jpg")):
        label_path = SRC_LABELS / "train" / f"{image_path.stem}.txt"
        labels = load_yolo_label(label_path)
        if not labels:
            continue

        image = read_image(image_path)
        if image is None:
            continue

        rng = rng_for_name(image_path.stem)
        augmented_image, augmented_labels = augment_image(image, labels, rng)
        out_image_path = out_train_images / f"{image_path.stem}__aug01.jpg"
        out_label_path = out_train_labels / f"{image_path.stem}__aug01.txt"
        write_image(out_image_path, augmented_image)
        save_yolo_label(out_label_path, augmented_labels)
        generated += 1

    return generated


def main() -> None:
    ensure_clean_dir(OUT_DIR)
    copy_split("train")
    copy_split("val")
    generated = build_augmented_train()
    write_data_yaml()

    metadata_path = DATASET_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    original_train = len(list((SRC_IMAGES / "train").glob("*.jpg")))
    val_count = len(list((SRC_IMAGES / "val").glob("*.jpg")))
    metadata["yolov8_augmented"] = {
        "original_train_images": original_train,
        "synthetic_train_images": generated,
        "total_train_images": original_train + generated,
        "val_images": val_count,
        "dataset_dir": str(OUT_DIR),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "generated_augmented_train_images": generated,
                "total_train_images": original_train + generated,
                "val_images": val_count,
                "output_dir": str(OUT_DIR),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
