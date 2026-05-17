# YOLO Training Workspace

This folder has two separate pipelines. They should not be mixed.

## 1) Wheel BBox Pipeline (existing)

Purpose:
- Detect wheel bounding boxes for side-view images.

Main files:
- `dataset/` (wheel bbox dataset)
- `dataset.yaml` (wheel bbox config)
- `train.py` (wheel bbox training entry point)
- `runs/wheel_bbox_phase2/` (wheel bbox outputs)

Use this when you need wheel center extraction from bbox.

## 2) Side-View Pose Pipeline (new clean path)

Purpose:
- Train 19 keypoints from LabelMe point annotations.

Main files:
- `side_view_dataset/annotation_batches/` (human annotation batches)
- `side_view_dataset/labelme_json/` (saved LabelMe JSON)
- `side_view_dataset/labels_pose/` (YOLO pose labels from converter)
- `train_pose.py` (pose training entry point)
- `side_view_dataset/pose_dataset/` (auto-built train/val for pose)
- `side_view_dataset/dataset_pose.yaml` (auto-generated pose config)
- `runs/side_view_pose_phase1/` (pose outputs)

## Typical Pose Workflow

1. Annotate images in LabelMe (point mode only).
2. Convert JSON to YOLO pose labels.
  Use wheel bbox training images as source so pose and wheelbox share the same pool:

```powershell
cd D:\project\sdi-helper
.\.venv\Scripts\python.exe yolo_training\labelme_to_yolo_pose.py `
  --input yolo_training\side_view_dataset\labelme_json `
  --output yolo_training\side_view_dataset\labels_pose `
  --img-dir yolo_training\dataset\images\train
```

If your LabelMe JSON was created from validation images, run the converter again with:

```powershell
cd D:\project\sdi-helper
.\.venv\Scripts\python.exe yolo_training\labelme_to_yolo_pose.py `
  --input yolo_training\side_view_dataset\labelme_json `
  --output yolo_training\side_view_dataset\labels_pose `
  --img-dir yolo_training\dataset\images\val
```

3. Train pose model (separate from wheel bbox):

```powershell
cd D:\project\sdi-helper
$env:POSE_EPOCHS='20'
.\.venv\Scripts\python.exe yolo_training\train_pose.py
```

After a successful run, `train_pose.py` now moves any labeled source images it
finds in `side_view_scrape/images/quality_pass/valid_candidates/` into
`side_view_scrape/images/quality_pass/labeled_from_phase1/` so they do not get
selected again. Set `POSE_ARCHIVE_LABELED=0` if you need to disable that
cleanup for a one-off debug run.

## Roboflow Workflow (Optional)

If you maintain labels in Roboflow, you can sync directly into this local
pipeline and optionally train immediately:

```powershell
cd D:\project\sdi-helper
.\.venv\Scripts\python.exe scripts\sync_pose_from_roboflow.py `
  --workspace <workspace-slug> `
  --project <project-slug> `
  --version <dataset-version> `
  --train
```

Notes:
- API key is read from `ROBOFLOW_API_KEY` (env or `.env`).
- `train` split is copied to `yolo_training/dataset/images/train`.
- `valid` split is copied to `yolo_training/dataset/images/val`.
- Labels are copied to `yolo_training/side_view_dataset/labels_pose`.
- Use `--clear-targets` if you want to replace local data instead of merging.

To upload your long-term local labeled pool (front/side/rear) directly from
`dataset_raw/images/train` and `dataset_raw/labels/train`:

```powershell
cd D:\project\sdi-helper
.\.venv\Scripts\python.exe scripts\upload_pose_subset_to_roboflow.py `
  --workspace <workspace-slug> `
  --project <project-slug>
```

This command now defaults to recursive upload from `dataset_raw/images/train`.
Use `--source-mode pose_subset` if you only want the recent pose training subset.

## Important Guardrail

- Do not use `train.py` for keypoint training.
- Do not use `train_pose.py` for wheel bbox training.

`train_pose.py` now enforces wheel-image alignment by default:
- Pose labels must match images from `yolo_training/dataset/images/train|val`.
- To temporarily bypass this guardrail for legacy UUID datasets, set `POSE_REQUIRE_WHEEL_IMAGES=0`.
- Optional legacy lookup from `dataset_raw/images/train/side` can be enabled with `POSE_ALLOW_LEGACY_SOURCE=1`.

They use different datasets and different YOLO tasks.
