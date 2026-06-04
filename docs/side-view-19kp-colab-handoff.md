# Side-View 19KP Colab Handoff

This handoff documents the manual GPU step that produces the 19KP candidate
model for the local promotion gate.

## Inputs

- Staged dataset: `yolo_training/side_view_dataset/_colab_staging/`
- Dataset config: `yolo_training/side_view_dataset/_colab_staging/dataset_pose.yaml`
- Holdout manifest: `yolo_training/runs/side_view_pose_19kp_candidate/holdout_manifest.txt`

## Colab Training Output

Run the 19KP pose training job in Colab using the staged dataset. Copy the best
weights back to:

```text
yolo_training/runs/side_view_pose_19kp_candidate/weights/best.pt
```

Do not rename this file for the promotion gate. The Makefile targets expect the
candidate model at that path unless `SIDE_19KP_MODEL` is overridden.

## Local Evaluation

After copying `best.pt` locally, run:

```powershell
make side-19kp-evaluate
```

This writes:

- `yolo_training/runs/side_view_pose_19kp_candidate/prediction_summary.csv`
- `yolo_training/runs/side_view_pose_19kp_candidate/evaluation_metadata.json`
- `yolo_training/runs/side_view_pose_19kp_candidate/holdout_manifest.txt`

`prediction_summary.csv` is the gate input. Every row must have `status` /
`verdict` equal to `PASS` before promotion can proceed.

## Promotion Gate

Run:

```powershell
make side-19kp-gate
```

This writes:

```text
yolo_training/runs/side_view_pose_19kp_candidate/gate_decision.json
```

Promotion is allowed only when `gate_decision.json` contains `"decision":
"PASS"`. Any failed holdout row, missing manifest, missing model, or missing
prediction summary blocks backend 19KP parser promotion.
