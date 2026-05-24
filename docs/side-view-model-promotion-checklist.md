# Side-View Model Promotion Checklist

Use this checklist before copying any side-view YOLO pose weights into the
backend `cv_service/models/` directory.

## Required Gate

The fixed side-view holdout gate must pass before promotion.

```powershell
cd D:\project\sdi-helper
.\.venv\Scripts\python.exe scripts\evaluate_7kp_body_end_model.py `
  --model D:\project\vehicle-sdi-system\cv_service\models\best.pt `
  --manifest yolo_training\runs\side_view_pose_7kp_bumper_oos_20260524\holdout_manifest.txt `
  --output-dir yolo_training\runs\side_view_pose_7kp_pre_promotion_gate `
  --device cpu
```

Equivalent Make target:

```powershell
make side-holdout-gate
```

The gate is strict: any failed holdout image returns a nonzero exit code and
blocks promotion.

## Acceptance Rules

- The holdout manifest stays fixed unless a new manifest is reviewed and
  documented.
- The keypoint order must remain:
  `front_wheel_center`, `front_wheel_ground`, `rear_wheel_center`,
  `rear_wheel_ground`, `ground_ref`, `front_bumper`, `rear_bumper`.
- `prediction_summary.csv` must show 0 failed rows.
- `bumper_review_contact_sheet.jpg` must be archived with the promotion
  evidence.
- A passing 7KP geometry gate does not mean the full 19KP side-view task is
  complete. It only preserves the promoted 7KP body-end geometry contract.

## Manual CI Gate

Run the GitHub Actions workflow `Side-View Holdout Gate` for any candidate
weights that are staged in the checked-out workspace. The workflow uploads the
prediction CSV, holdout manifest copy, and review contact sheet as artifacts.

## Promotion Record

Record the following in the promotion PR:

- Candidate model path and checksum.
- Holdout manifest path.
- Command or workflow run URL.
- PASS/FAIL count.
- Artifact path or uploaded artifact name.
- Any residual risk, especially orientation-risk rows or 19KP coverage gaps.
