# Pool Video Crop + Auto-Label Pipeline

This repository contains a local pipeline for:

1. Correcting fish-eye distortion with OpenCV fisheye remapping.
2. Cropping the corrected swimming videos to the pool area.
3. Building a COCO-style swimmer dataset using `PekingU/rtdetr_r18vd_coco_o365` detections refined with FastSAM masks.

## Inputs

- Source archive: `C:\Users\A006\Downloads\Telegram Desktop\swim_vids.zip`
- Extracted videos: `data/raw_videos/swim_vids/*.mp4`

## Outputs

- Cropped videos: `dataset/videos_cropped/*.mp4`
- Undistorted videos: `dataset/videos_undistorted/*.mp4`
- Images: `dataset/images/train/*.jpg`, `dataset/images/val/*.jpg`
- COCO annotations: `dataset/annotations/instances_train.json`, `dataset/annotations/instances_val.json`
- YOLOv8 labels: `dataset/yolov8/labels/train/*.txt`, `dataset/yolov8/labels/val/*.txt`
- YOLOv8 config: `dataset/yolov8/data.yaml`
- Augmented YOLOv8 dataset: `dataset/yolov8_augmented/{images,labels}` with `dataset/yolov8_augmented/data.yaml`
- Dataset summary: `dataset/metadata.json`

## Run

```powershell
pip install -r requirements.txt
python .\scripts\undistort_fisheye_videos.py
python .\scripts\crop_pool_videos.py
python .\scripts\build_autolabeled_dataset.py
python .\scripts\extend_dataset_from_video.py --video <path-to-video> --video-id <video-id> --target-images 260
python .\scripts\export_yolov8_labels.py
python .\scripts\build_yolov8_augmented_dataset.py
```

## Notes

- The scene configuration lives in `config/pool_scene.json`.
- `ffmpeg` must be available in `PATH` to run the crop step.
- The repository vendors FastSAM in `vendor/FastSAM` and includes the model weights used for the final export.
- Fish-eye correction uses the same `cv2.fisheye` flow as the referenced gist, with scene-tuned parameters for this camera.
- Validation split is video-based: `video_004` is reserved for validation.
- The dataset class is `swimmer`, derived from RT-DETR `person` detections limited to the pool polygon and refined by FastSAM.
- The auto-label pass includes additional edge-case filtering based on a core pool mask, water-color overlap, and stricter handling of right-border detections.
- `dataset/yolov8_augmented` is a self-contained YOLOv8 training export with offline color/contrast/noise/blur augmentation and optional horizontal flip for train images.
