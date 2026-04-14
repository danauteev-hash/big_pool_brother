# Pool Video Crop + Auto-Label Pipeline

This repository contains a local pipeline for:

1. Correcting fish-eye distortion with OpenCV fisheye remapping using scene calibration values.
2. Cropping the corrected swimming videos to the mask-derived pool bounding box.
3. Building a COCO-style swimmer dataset using `PekingU/rtdetr_r18vd_coco_o365` detections refined with a configurable segmentation backend.
4. Replacing dense refinement with the official vendored `SAM3` backend and a lightweight fallback when the machine cannot run SAM3.
5. Exporting an augmented `YOLOv26 pose` dataset with per-image visualization overlays and a Label Studio review bundle.
6. Training the pose model from a repo-local script with stable relative paths.

## Inputs

- Source videos can live anywhere local if you pass an explicit path.
- Default full-video location for the one-command runner: `auto-dataset/input_videos/full_pool_video.mp4`
- Repo-local intermediate videos: `dataset/videos_undistorted/*.mp4`
- Short source videos used for base rebuilds: `data/raw_videos/swim_vids/*.mp4`

## Outputs

- Cropped videos: `dataset/videos_cropped/*.mp4`
- Undistorted videos: `dataset/videos_undistorted/*.mp4`
- Images: `dataset/images/train/*.jpg`, `dataset/images/val/*.jpg`
- COCO annotations: `dataset/annotations/instances_train.json`, `dataset/annotations/instances_val.json`
- YOLOv8 labels: `dataset/yolov8/labels/train/*.txt`, `dataset/yolov8/labels/val/*.txt`
- YOLOv8 config: `dataset/yolov8/data.yaml`
- Augmented YOLOv8 dataset: `dataset/yolov8_augmented/{images,labels}` with `dataset/yolov8_augmented/data.yaml`
- YOLOv26 pose dataset: `dataset/yolov26_pose/{images,labels,visualizations,label_studio}` with `dataset/yolov26_pose/data.yaml`
- Training entrypoint: `scripts/train_yolov26_pose.py`
- Auto-dataset runner: `auto-dataset/run_auto_dataset.py` with `auto-dataset/config.json`
- Dataset summary: `dataset/metadata.json`
- Official SAM3 source: `vendor/sam3`

## Run

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 .\scripts\undistort_fisheye_videos.py
py -3.11 .\scripts\crop_pool_videos.py
py -3.11 .\scripts\build_autolabeled_dataset.py
py -3.11 .\scripts\extend_dataset_from_video.py --video <path-to-video> --video-id <video-id> --sample-fps 4.0
py -3.11 .\scripts\export_yolov8_labels.py
py -3.11 .\scripts\build_yolov8_augmented_dataset.py
py -3.11 .\scripts\build_yolov26_pose_dataset.py
py -3.11 .\auto-dataset\run_auto_dataset.py
py -3.11 .\scripts\train_yolov26_pose.py --epochs 120 --imgsz 960 --batch 8
```

## Notes

- The scene configuration lives in `config/pool_scene.json`.
- `ffmpeg` must be available in `PATH` to run the crop step.
- `build_autolabeled_dataset.py` now imports the official `facebookresearch/sam3` code from `vendor/sam3`.
- The expected checkpoint path is `models/sam3.pt`. The checkpoint itself is not committed because upstream distributes it through gated Hugging Face access.
- If SAM3 dependencies or checkpoints are unavailable, the active fallback is `BoxRefine`, so the repo still runs end-to-end after a clean clone.
- Fish-eye correction uses the same `cv2.fisheye` flow as the referenced gist and now supports both legacy `scale` mode and the newer `balance + fov_scale + auto_crop + crop_margin_px` flow. The current repo config uses `balance=0.35` and `crop_margin_px=8`.
- The undistort step writes `*_undistorted_plan.json` alongside each corrected video so the exact crop ROI survives after cloning.
- `crop_pool_videos.py` no longer relies on a hard-coded rectangle: it estimates the pool contour from blue water, extracts the crop coordinates from the binary mask in OpenCV, writes a per-video `*_pool_geometry.json` debug file, and outputs the rectangular pool crop from that bbox.
- `build_autolabeled_dataset.py` resolves the active pool polygon per cropped video from the saved geometry JSON, so slightly different crop sizes no longer break the label pass.
- Validation split is video-based: `video_004` is reserved for validation.
- The dataset class is `swimmer`, derived from RT-DETR `person` detections limited to the pool polygon and refined by the active segmentation backend.
- The auto-label pass includes additional edge-case filtering based on a core pool mask, water-color overlap, and stricter handling of right-border detections.
- `dataset/yolov8_augmented` is a self-contained YOLOv8 training export with offline color/contrast/noise/blur augmentation and optional horizontal flip for train images.
- `extend_dataset_from_video.py` now samples the full source video by exact `sample_fps`, and the current full-video pass was rebuilt at `4 fps`.
- `dataset/yolov26_pose` is a self-contained pose export using `yolo26n-pose.pt` to predict 17-keypoint human poses on swimmer crops, generate offline augmentations, save visualization overlays, and create a Label Studio COCO review bundle.
- `auto-dataset` is a one-command reproduction folder that can rerun `undistort -> crop -> auto-label -> yolo exports -> 4 fps full-video extension -> yolov26_pose` from `auto-dataset/config.json`.
- Tracked config and metadata use repo-relative paths so the project stays portable after cloning.
- Raw full-length input videos are intentionally not committed. Put them under `auto-dataset/input_videos/` or pass `--video <path>` to the runner.
