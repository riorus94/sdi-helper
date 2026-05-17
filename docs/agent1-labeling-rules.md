# Agent 1 Labeling Rules

All side-view labeling work must follow this document. Do not create or train
pose labels from image sets that skipped Agent 1 staging, filtering, and review
rules.

## Current Committed Baseline

These recent commits define the active labeling workflow:

- `8db4558` fixes duplicate LabelMe labels and derives 5KP `ground_ref`
- `13c0cef` adds `fix_ground_ref.py`, 9KP config, and left-looking mirroring
- `03b259f` fixes pose `flip_idx` and tunes Colab pose training
- `ace62eb` adds Agent 1 non-side rejection and removes `stanford_00002`
- `2e635c6` adds clean 300-image side candidate staging

If a Colab/runtime clone does not include these commits, pull `origin master`
before labeling, converting, or training.

## Allowed Workflows

Use one of these workflows only:

1. **5KP no-roof stabilization**
   - Use `config/agent1_keypoint_priority_5kp_no_roof_apex.json`
   - Allowed labels:
     - `ground_ref`
     - `front_wheel_center`
     - `front_wheel_ground`
     - `rear_wheel_center`
     - `rear_wheel_ground`

2. **9KP side-view expansion**
   - Use `config/agent1_keypoint_priority_9kp_side.json`
   - Allowed labels:
     - 5KP labels above
     - `fender_arch_front`
     - `fender_arch_rear`
     - `front_bumper`
     - `rear_bumper`

Do not mix 5KP, 9KP, and 19KP outputs in the same YOLO pose label directory.

## Candidate Staging

Stage new candidates through the clean side workflow:

```powershell
.\.venv\Scripts\python.exe scripts\stage_side_pose_candidates.py
```

The staged output is ignored by Git and lives under:

```text
yolo_training/side_view_dataset/subsets/pose_candidates_300/
```

The staging script must exclude:

- stems in `config/scrape_exceptions.yaml`
- known rejected/non-side stems
- already labeled `labelme_json` stems
- duplicate stems
- duplicate image hashes
- corrupt or too-small images
- obvious filename anomalies such as front/rear/quarter/interior/top

Staging is not final approval. A visual skim is still required before training.

## Agent 1 Generation

For 9KP side labels, run:

```powershell
.\.venv\Scripts\python.exe scripts\suggest_keypoints.py `
  --image-dir yolo_training\side_view_dataset\subsets\pose_candidates_300\images `
  --output yolo_training\side_view_dataset\subsets\pose_candidates_300\labelme_json_9kp `
  --priority-config config\agent1_keypoint_priority_9kp_side.json `
  --phase-only phase1 `
  --orientation-classifier clip `
  --reject-non-side `
  --quality-report yolo_training\side_view_dataset\subsets\pose_candidates_300\agent1_quality_report.csv `
  --overwrite
```

CLIP orientation is mandatory for Agent 1 keypoint LabelMe generation. If CLIP
cannot load or download, fix the CLIP setup first; do not generate training
labels without it.

The `--reject-non-side` flag is mandatory. Images rejected with
`rejected_non_side_view` must not be used for pose training unless a human
explicitly promotes them.

The only approved bypass is `--allow-no-clip-experimental`, and only for
throwaway experiments. Any report from that mode must include
`orientation_clip_missing: experimental run not valid for training`; those JSONs
must not be promoted to training.

## Orientation Rules

Label names are semantic, not screen-position shortcuts.

- right-looking vehicle:
  - `front_wheel_*` is right of `rear_wheel_*`
- left-looking vehicle:
  - `front_wheel_*` is left of `rear_wheel_*`

Agent 1 must use CLIP orientation to assign front/rear before writing keypoint
JSON. Runs without CLIP should fail loudly unless the experimental bypass is
explicitly requested.

Horizontal flip augmentation requires front/rear swap. `train_pose.py` now
computes `flip_idx` from keypoint names. Do not hand-edit dataset YAML to an
identity `flip_idx` for 5KP or 9KP.

## Ground Reference Rules

`ground_ref` is a geometric reference, not a free visual landmark.

For 5KP and 9KP side-view labels:

```text
ground_ref.x = midpoint(front_wheel_ground.x, rear_wheel_ground.x)
ground_ref.y = midpoint(front_wheel_ground.y, rear_wheel_ground.y)
```

Before converting old LabelMe JSON, repair it when needed:

```powershell
.\.venv\Scripts\python.exe scripts\fix_ground_ref.py `
  --json-dir yolo_training\side_view_dataset\labelme_json `
  --report logs\ground_ref_fix.csv
```

The 5KP converter also derives `ground_ref` during export, but source JSON
should still be repaired so review and future exports remain consistent.

## Review Gates

Use the Agent 1 quality report:

- `REVIEW_LOW`: best first-pass candidates
- `REVIEW_MEDIUM`: usable but inspect
- `REVIEW_HIGH`: inspect before training
- `rejected_non_side_view`: exclude unless manually promoted

Training labels must not include:

- front/three-quarter/rear/top/interior images
- images with only fallback wheel anchors unless human-corrected
- images with swapped front/rear labels
- `ground_ref` not aligned to wheel-ground midpoint
- missing required keypoints for the chosen config

## Conversion Rules

For 5KP no-roof:

```powershell
.\.venv\Scripts\python.exe yolo_training\labelme_to_yolo_pose.py `
  --input yolo_training\side_view_dataset\labelme_json `
  --output yolo_training\side_view_dataset\labels_pose_5kp_no_roof `
  --img-dir dataset_raw\images\train\side `
  --keypoints ground_ref,front_wheel_center,front_wheel_ground,rear_wheel_center,rear_wheel_ground
```

Always delete stale generated label directories before reconverting:

```powershell
Remove-Item yolo_training\side_view_dataset\labels_pose_5kp_no_roof -Recurse -Force
```

## Training Hygiene

Before Colab/local training:

- pull latest `origin master`
- regenerate YOLO pose labels from repaired/reviewed LabelMe JSON
- delete old run directories if comparing clean runs
- use the tuned Colab training env from `notebooks/colab_pose_training.ipynb`

Do not judge model quality from box metrics alone. Pose quality must be tracked
with `metrics/mAP50(P)` and `metrics/mAP50-95(P)`.
