# Clean Side 300 Workflow

Use this workflow to stage a reproducible 300-image side-view candidate set for
Agent 1 labeling and human review.

```powershell
.\.venv\Scripts\python.exe scripts\stage_side_pose_candidates.py
```

For the Stanford-only mapper workflow, prefer:

```powershell
.\.venv\Scripts\python.exe scripts\stage_side_pose_candidates.py `
  --source-roots dataset_raw\images\train\side `
  --output-root yolo_training\side_view_dataset\subsets\stanford_side_clean `
  --required-stem-prefix stanford_ `
  --limit 300
```

The script writes:

- `yolo_training/side_view_dataset/subsets/pose_candidates_300/images/`
- `yolo_training/side_view_dataset/subsets/pose_candidates_300/manifest.csv`
- `yolo_training/side_view_dataset/subsets/pose_candidates_300/rejections.csv`
- `yolo_training/side_view_dataset/subsets/pose_candidates_300/summary.json`

Selection rules:

- exclude `config/scrape_exceptions.yaml` side image stems
- exclude known rejected/non-side stems from reject roots
- skip already-labeled `yolo_training/side_view_dataset/labelme_json` stems
- dedupe by image stem
- dedupe by image content hash
- reject obvious front/rear/quarter/interior/top filename anomalies
- reject corrupt or too-small images
- copy only image files with supported extensions

After staging, run Agent 1 with CLIP orientation and non-side rejection. Both
flags are mandatory for training-bound side-view keypoint JSON:

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

Review `REVIEW_HIGH` and any `rejected_non_side_view` rows before using labels
for pose training.

If CLIP cannot load or download, stop and fix setup first. The
`--allow-no-clip-experimental` bypass is only for throwaway experiments; reports
from that mode are marked `orientation_clip_missing: experimental run not valid
for training`.
