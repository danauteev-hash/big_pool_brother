# Auto-Dataset

This folder is a one-entry pipeline for rebuilding the swimmer auto-dataset from the full source video.

## What it does

1. Samples the full pool video at `4 fps`.
2. Runs the swimmer auto-label pass.
3. Builds a self-contained `YOLOv26 pose` dataset in `dataset/yolov26_pose`.
4. Saves visualization overlays and a `Label Studio` import bundle.

## SAM3 note

- The preferred backend is the official vendored `SAM3` code from `facebookresearch/sam3`.
- The expected checkpoint path is `models/sam3.pt`.
- On this machine the pipeline is configured to fall back to lightweight box refinement if `SAM3` is unavailable or too heavy.

## Run

```powershell
python .\auto-dataset\run_auto_dataset.py
python .\auto-dataset\run_auto_dataset.py --video C:\path\to\your\full_pool_video.mp4
```

## Files

- Config: `auto-dataset/config.json`
- Runner: `auto-dataset/run_auto_dataset.py`
- Put the source video here by default: `auto-dataset/input_videos/full_pool_video.mp4`
- Last run summary: generated as `auto-dataset/last_run_summary.json`
- `run_auto_dataset.py` syncs the SAM3-related keys from `auto-dataset/config.json` into `config/pool_scene.json` before it runs the pipeline.
