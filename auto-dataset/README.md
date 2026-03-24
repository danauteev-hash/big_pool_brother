# Auto-Dataset

This folder is a one-entry pipeline for rebuilding the swimmer auto-dataset from the full source video.

## What it does

1. Samples the full pool video at `4 fps`.
2. Runs the swimmer auto-label pass.
3. Builds a self-contained `YOLOv26 pose` dataset in `dataset/yolov26_pose`.
4. Saves visualization overlays and a `Label Studio` import bundle.

## SAM3 note

- The preferred backend is `SAM3` from the official `facebookresearch/sam3` project.
- The expected checkpoint path is `models/sam3_b.pt`.
- On this machine the pipeline is configured to fall back to lightweight box refinement if `SAM3` is unavailable or too heavy.

## Run

```powershell
python .\auto-dataset\run_auto_dataset.py
```

## Files

- Config: `auto-dataset/config.json`
- Runner: `auto-dataset/run_auto_dataset.py`
- Last run summary: `auto-dataset/last_run_summary.json`
