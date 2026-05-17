# Roboflow Wheelbox Best Practices

Use this as the default context when working on wheelbox uploads, annotation, and Roboflow sync in `sdi-helper`.

## Default Setup

- Workspace: `sdi-j1mci`
- Wheelbox project: `wheel-bbox-agent1-prelabel`
- Local wheelbox model source: `yolo_training/runs/roboflow_v3_local/weights/best.pt`
- Upload target: Roboflow project images + annotations, not model weights, on free tier

## Format Choice

- For wheelbox object detection, export/upload as `YOLOv8`
- Do not select oriented bounding boxes for this pipeline
- Do not use keypoint or pose export formats for wheelbox detector training

## Data Rules

- Keep Stanford data separated with the `stanford_` prefix until it is reviewed and promoted
- Treat AI-generated labels as drafts only
- Promote only human-reviewed labels into training-ready datasets
- Prefer false reject over false accept for wheelbox quality gates

## Annotation Rules

- Use Agent 1 to pre-label wheelbox boxes from the current best model
- For Pareto-first work, prelabel Phase 1 only on the same image set that was uploaded to Roboflow
- Keep the wheelbox prelabel report and label files together for traceability
- Stage training-ready pairs as image + `.txt` label files before upload or training

## Roboflow Upload Rules

- Use `download zip to computer` only when you need the dataset export
- On the free tier, expect dataset export access rather than custom-weights upload
- If the UI asks for a format, choose `YOLOv8`
- If the UI asks for architecture, choose `YOLOv8` for this wheelbox detector

## Operational Notes

- Upload a fresh project if the current project is not writable or returns permission errors
- Keep the workspace explicit in scripts and environment variables so uploads do not drift across workspaces
- Prefer a small validated batch first, then scale up after the upload path is confirmed
