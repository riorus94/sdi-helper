# Roboflow Setup Checklist

> **Goal**: Migrate B1 side-view keypoint annotation to Roboflow web UI for parallel labeling  
> **Timeline**: May 17–18, 2026  
> **Team**: 2–3 annotators  

---

## Pre-Setup (May 16 — Today)

- [ ] **Continue batch_006 in local LabelMe** (~1 hour)
  - 10 images → 63 total labeled
  - Command: `python -m labelme yolo_training/side_view_dataset/annotation_batches/batch_006/images --labels yolo_training/labelme_labels.txt`
  - Save JSON files to `yolo_training/side_view_dataset/labelme_json/`

- [ ] **Verify Roboflow API key**
  - Check: `.env` contains `ROBOFLOW_API_KEY=...`
  - Test: `python -c "from roboflow import Roboflow; rf = Roboflow(api_key='YOUR_KEY'); print(f'Workspace: {rf.workspace(...)}')"` (if needed)

---

## Setup Day (May 17 Morning — ~30 min)

### Step 1: Create Roboflow Project
- [ ] Log into https://app.roboflow.com
- [ ] Workspace: `akhmad-rio-rusdiano`
- [ ] Click **"Create New Project"**
- [ ] **Project name**: `sdi-side-pose-19kp`
- [ ] **Project type**: `Instance Segmentation` or `Keypoint Detection` (check Roboflow's latest options)
- [ ] **License**: Internal/Private
- [ ] Click **Create**

### Step 2: Define Keypoint Schema (if required)
- [ ] In project settings, go to **Labels/Keypoints**
- [ ] Add 19 keypoints with exact names from `yolo_training/labelme_labels.txt`:
  - `roof_apex`, `side_window_top_front`, `side_window_top_rear`, `front_bumper`, `rear_bumper`
  - `front_wheel_center`, `front_wheel_ground`, `rear_wheel_center`, `rear_wheel_ground`
  - `fender_arch_front`, `fender_arch_rear`, `hood_edge`
  - `body_waist_front`, `body_waist_rear`, `panel_front`, `panel_rear`
  - `windshield_base`, `rear_glass_base`, `ground_ref`
- [ ] Save keypoint schema

### Step 3: Upload Images
**Option A — Web UI (simpler)**:
- [ ] Navigate to **"Upload"** in project
- [ ] Drag-drop all images from `dataset_raw/images/train/side/` (~103 images)
- [ ] Let Roboflow auto-split into train/val (or specify 80/20 manually)
- [ ] Click **Upload**

**Option B — Python API** (if web UI is slow):
- [ ] Create script `scripts/upload_to_roboflow_pose.py`:
  ```python
  from roboflow import Roboflow
  from pathlib import Path

  rf = Roboflow(api_key="YOUR_API_KEY")
  project = rf.workspace("akhmad-rio-rusdiano").project("sdi-side-pose-19kp")

  image_dir = Path("dataset_raw/images/train/side")
  for img in sorted(image_dir.glob("*.jpg")):
      project.upload(path=str(img), batch_name="batch_001")
      print(f"Uploaded: {img.name}")
  ```
- [ ] Run: `python scripts/upload_to_roboflow_pose.py`

### Step 4: Organize Into Batches
- [ ] In Roboflow dashboard, **Create batches/assignments**:
  - Batch 1: Images 1–25 → Annotator A
  - Batch 2: Images 26–50 → Annotator B
  - Batch 3: Images 51–103 → Annotator C (if 3rd person available)

---

## Annotation Phase (May 17 Afternoon — May 18 Morning)

### For Each Annotator:
- [ ] Receive Roboflow task assignment (email/dashboard notification)
- [ ] Log into https://app.roboflow.com
- [ ] Click **"Annotate"** on assigned batch
- [ ] For each image:
  1. Click to place 19 keypoints (follow reference image if provided)
  2. Keyboard shortcut to confirm placement (check Roboflow UI)
  3. Move to next image
- [ ] Mark batch as **"Complete"** when done

### Progress Tracking:
- [ ] Check dashboard for completion %
- [ ] Target: 50–70 images labeled by end of May 17
- [ ] Target: 100+ images labeled by mid-May 18

---

## Export & Integration (May 18 Afternoon)

### Step 1: Export from Roboflow
**Option A — Web UI**:
- [ ] Go to **"Versions"** in project
- [ ] Create new version (latest annotations) with:
  - Format: **YOLOv8** (or **YOLO Pose**)
  - Train/Val split: **80/20** (or current split)
- [ ] Export as ZIP

**Option B — Automated Sync**:
- [ ] Use existing script:
  ```bash
  cd d:\project\sdi-helper
  python scripts/sync_pose_from_roboflow.py \
    --workspace akhmad-rio-rusdiano \
    --project sdi-side-pose-19kp \
    --version 1 \
    --train
  ```
  This will:
  1. Download YOLO-format dataset from Roboflow
  2. Copy images to `yolo_training/dataset/images/{train,val}`
  3. Copy labels to `yolo_training/side_view_dataset/labels_pose`
  4. Trigger `yolo_training/train_pose.py` automatically

### Step 2: Validate
- [ ] Check `yolo_training/dataset/images/train/` — should have ~80 images
- [ ] Check `yolo_training/side_view_dataset/labels_pose/` — should have .txt files
- [ ] Quick spot-check: `ls yolo_training/dataset/images/train/ | wc -l`

### Step 3: Train
- [ ] Run:
  ```bash
  cd d:\project\sdi-helper
  python yolo_training/train_pose.py
  ```
- [ ] Monitor training output (should complete in ~15–30 min on small dataset)
- [ ] Model saved to: `yolo_training/runs/detect/trainN/weights/best.pt`

---

## Post-Training (May 18 Evening)

### Step 1: Evaluate Model
- [ ] Check metrics in training logs:
  - `yolo_training/runs/detect/trainN/results.csv`
  - Look for class precision/recall per keypoint
  - Identify "hard" keypoints (low precision)

### Step 2: Feedback (Optional)
- [ ] If certain keypoints have low precision (e.g., hood_edge):
  - Share metric feedback with annotation team
  - Re-review samples where that keypoint was labeled
  - Improve annotations in next batch

### Step 3: Integrate into Backend
- [ ] Copy model: `yolo_training/runs/detect/trainN/weights/best.pt` → `vehicle-sdi-system/cv_service/models/body_pose.pt`
- [ ] Update `vehicle-sdi-system/cv_service/` to load new model (if not auto-detected)
- [ ] Commit and merge to main

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Roboflow API key not found** | Check `.env`: `ROBOFLOW_API_KEY=...` |
| **Images not uploading** | Check file format (JPG/PNG), max size (~10 MB) |
| **Keypoints not saving** | Refresh browser, check network connection |
| **Export fails** | Ensure version was generated (may take 5 min after batch completion) |
| **train_pose.py fails on small dataset** | Check `--epochs`, `--batch`, `patience` settings; may need `--epochs 50` for tiny sets |

---

## Success Criteria

✅ **Phase 1 Complete** (May 18 EOD):
- [ ] 100+ side-view images annotated
- [ ] All 100 images exported from Roboflow in YOLO format
- [ ] Local training completed with new body_pose model
- [ ] Model integrated into backend
- [ ] Backlog updated with B1/B2 completion status

---

## Notes

- **Annotator time**: ~5–8 min per image (varies by experience)
- **Batch size**: Keep to 10–25 images per annotator per batch (easier to track)
- **QA**: Spot-check ~10% of annotations (senior person reviews)
- **Future batches**: Once playbook is established, can scale to front/rear views
- **Keyboard shortcuts**: Roboflow should provide; summarize for team

