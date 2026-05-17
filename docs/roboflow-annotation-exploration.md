# Roboflow Web Annotation vs. Local LabelMe

**Objective**: Evaluate whether Roboflow's web-based annotation interface can improve B1 (side-view keypoint annotation) by enabling **parallel labeling**.

---

## Current Setup: Local LabelMe

### Workflow
```
Batch Prep (batch_NNN/images)
    ↓
LabelMe GUI (local, single-user, ~5-10 min/image)
    ↓
Manual JSON → YOLO conversion (labelme_to_yolo_pose.py)
    ↓
Training (train_pose.py)
```

### Strengths
- ✅ Offline (no network dependency)
- ✅ Fast keyboard shortcuts
- ✅ Full control over schema
- ✅ Output directly compatible with YOLO format

### Weaknesses
- 🔴 **Single-user only** — bottleneck on annotation speed
- ⚠️ Manual script chains (batch prep → annotation → conversion → training)
- ⚠️ No progress visibility across batches
- ⚠️ No annotation feedback (e.g., difficult keypoints)
- ⚠️ Requires Python environment setup per machine

---

## Roboflow Web Annotation

### Existing Roboflow Integration
The project **already uses Roboflow** for:
1. **Classification** (`side_view_cls_rf.py`): Upload side-view validity labels via REST API
2. **Pose Dataset Sync** (`sync_pose_from_roboflow.py`): Download annotated YOLO-pose datasets → train locally

**Workspace**: `akhmad-rio-rusdiano` (from code)  
**API Key**: Stored in `.env` as `ROBOFLOW_API_KEY`

### How Roboflow Web Annotation Works

1. **Upload images** to Roboflow (via web UI or API)
2. **Create annotation task** with:
   - Keypoint schema (19 points: A, J, D, E, F1–F4, G, H, I, body extras)
   - Team members assigned to batch
3. **Web-based annotator UI**:
   - Drag-and-drop points on image
   - Keyboard shortcuts for fast labeling
   - Real-time validation
4. **Export**:
   - Auto-generates YOLO format
   - Download or sync via API (`sync_pose_from_roboflow.py`)

### Strengths
- ✅ **Multi-user parallel annotation** — 2–3 people label simultaneously
- ✅ **Web-based** — no client software setup needed
- ✅ **Progress tracking** — real-time dashboard
- ✅ **Auto-export to YOLO** — no conversion script needed
- ✅ **Built-in QA** — review annotations, reassign, etc.
- ✅ **Works with existing sync script** (`sync_pose_from_roboflow.py`)

### Weaknesses
- ⚠️ **Requires network** (cloud-hosted)
- ⚠️ **Cost** (Roboflow has free tier limits; check usage)
- ⚠️ **Learning curve** for new annotators
- ⚠️ **Keypoint schema must be set up in Roboflow** (one-time)

---

## Cost/Benefit Analysis

### Time to Implement Roboflow Annotation

| Task | Effort | Blocker? |
|------|--------|----------|
| Set up Roboflow pose project (19 keypoints) | 30 min | No |
| Upload 53 existing LabelMe JSONs to Roboflow | 1 h | No — optional, can start fresh |
| Upload batch_006 to Roboflow | 15 min | No |
| Test web annotation UI + export | 30 min | No |
| Migrate 2–3 team members to Roboflow | 1 h | Yes — requires training |
| **Total** | **~3 h** | Yes (team sync) |

### Annotation Speed Impact

**Local LabelMe** (current):
- 1 person × 27 remaining images × 8 min/image = **3.6 hours wall-clock time**

**Roboflow Web** (parallel):
- 2 people × 27 images × 8 min/image ÷ 2 = **1.8 hours wall-clock time** (50% reduction)
- 3 people × 27 images × 8 min/image ÷ 3 = **1.2 hours wall-clock time** (67% reduction)

**ROI**: Implementation cost (~3h team sync) breaks even after annotating ~50 images in parallel.

---

## Recommended Migration Path

### Option A: Hybrid Approach (Recommended)
**Continue local LabelMe for batch_006** (10 images, ~1h), then **switch to Roboflow for remaining batches**.

```
TODAY (May 16)
└── Finish batch_006 in LabelMe (53 → 63 labeled)
    └── Convert JSON → YOLO
    └── Train on 63 images (baseline)

TOMORROW (May 17)
└── Set up Roboflow pose project + keypoint schema
└── Upload batches 001, 002, 007-010 (37 images) to Roboflow
└── Assign to 2–3 annotators
└── Wait for completion (~2 hours with parallel work)
└── Export YOLO format from Roboflow
└── Retrain on ~100 labeled images (Phase 2 achieved)
```

### Option B: Full Migration
**Skip local batch_006**, migrate everything to Roboflow now (saves setup time later).

```
TODAY (May 16)
└── Set up Roboflow pose project (30 min)
└── Upload all 103 side images to Roboflow (10 min)
└── Assign 53 completed images to "Done" (in Roboflow)
└── Create batch for 50 remaining images
└── Assign to team

TOMORROW-NEXT DAY (May 17–18)
└── Parallel annotation (2 people × 25 images each = ~4 hours wall-clock)
└── Export from Roboflow (5 min)
└── Sync to local pipeline (5 min)
└── Retrain on ~103 labeled images
```

---

## Step-by-Step: Set Up Roboflow Pose Project

### 1. Log into Roboflow Web UI
- URL: https://app.roboflow.com
- Workspace: akhmad-rio-rusdiano
- Create new **Detection/Keypoint** project (NOT classification)

### 2. Create Keypoint Schema
**Name**: sdi-side-pose-19kp  
**Type**: Instance Keypoint Detection  
**Keypoints** (19 total) — from `yolo_training/labelme_labels.txt`:
```
1. A (rear_bumper)
2. B (rear_wheel_center)
3. C (rear_wheel_ground)
4. D (fender_arch_rear)
5. E (hood_edge)
6. F (body_waist_rear)
7. G (panel_rear)
8. H (rear_glass_base)
9. I (roof_apex)
10. J (side_window_top_rear)
11. K (side_window_top_front)
12. L (front_bumper)
13. M (front_wheel_center)
14. N (front_wheel_ground)
15. O (fender_arch_front)
16. P (body_waist_front)
17. Q (panel_front)
18. R (windshield_base)
19. S (ground_ref)
```

These are the exact labels in `yolo_training/labelme_labels.txt`.

### 3. Upload Images
**Option A**: Web UI upload (~10 min for 103 images)
- Dashboard → Create Dataset
- Drag-drop or select images
- Auto-organize into train/val

**Option B**: API upload (programmatic)
```python
# Use existing roboflow Python SDK
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("akhmad-rio-rusdiano").project("sdi-side-pose-19kp")
# Upload images via project.upload() or REST API
```

### 4. Assign Annotation Tasks
- Create batches (e.g., "Batch 001–010", "Batch 011–020")
- Assign to team members
- Set due date + priority

### 5. Annotate
- Annotators log in to https://app.roboflow.com
- Click "Annotate" on assigned task
- Click keypoints on image
- Save and move to next

### 6. Export
- **Option A**: Dashboard → Export → YOLOv8 format (download ZIP)
- **Option B**: Sync via existing script:
  ```bash
  python scripts/sync_pose_from_roboflow.py \
    --workspace akhmad-rio-rusdiano \
    --project sdi-side-pose-19kp \
    --version 1 \
    --train
  ```

---

## FAQ

### Q: Can we import existing LabelMe JSONs into Roboflow?
**A**: Not directly. You'd need to:
1. Convert LabelMe JSON → COCO format (or YOLO)
2. Upload to Roboflow via web UI (drag-drop images + pre-labeled JSONs)
3. Roboflow will re-serve them for QA/review

**Effort**: ~1 hour for 53 images + conversion script.  
**Recommendation**: Skip this; start fresh with Roboflow. 53 images are already labeled locally.

### Q: What's the Roboflow free tier limit?
**A**: Free tier supports:
- Up to 5,000 images per project
- 1 project
- Basic export formats
- Community support

**Status**: We're well under limit (103 images).  
**Cost**: Free for our use case.

### Q: Can multiple people annotate the same batch?
**A**: Yes. Roboflow allows:
- Multiple annotators per task
- Individual image assignments
- Re-review and reassignment if needed

### Q: How do we ensure consistency across annotators?
**A**: 
1. **Pre-annotation briefing**: Show reference images + keypoint diagram
2. **QA review**: Spot-check ~10% of annotations (senior person)
3. **Feedback loop**: If keypoint X is consistently misaligned, retrain annotators

### Q: What if an annotator marks a 3/4 view?
**A**: Roboflow allows marking images as "rejected" or moving them to an "invalid" category. These can be excluded from export.

---

## Comparison Table

| Dimension | Local LabelMe | Roboflow Web |
|-----------|---------------|--------------|
| **Setup time** | 0 min (already running) | 30 min |
| **Parallel annotators** | 1 | 2–3+ |
| **Estimated time to label 50 images** | 6–8 hours | 2–3 hours |
| **Off-network capable** | ✅ Yes | ❌ No |
| **Auto YOLO export** | ❌ Script required | ✅ Native |
| **Progress visibility** | ❌ Manual checking | ✅ Dashboard |
| **QA/review features** | ❌ Manual | ✅ Built-in |
| **Cost** | Free | Free (under 5K images) |
| **Learning curve** | Low | Medium |

---

## Recommendation

**Go with Option A (Hybrid)** for immediate impact:

1. **Finish batch_006 in LabelMe today** (10 images, ~1 hour)
   - Keeps local pipeline flowing
   - Minimal disruption

2. **Set up Roboflow pose project tomorrow morning** (30 min)
   - Create 19-keypoint schema
   - Upload remaining 40 images

3. **Launch parallel annotation** (2 people, 4–5 hours wall-clock)
   - Reach ~100 labeled images by **May 18**
   - Unblock B2 retraining

4. **Make Roboflow standard** for future annotation
   - Establish playbook (batch assignment, QA, export)
   - Scale team annotation capacity

---

## Next Actions

- [ ] Verify Roboflow API key in `.env`
- [ ] Create Roboflow pose project (web UI)
- [ ] Define keypoint schema in Roboflow (copy from `labelme_labels.txt`)
- [ ] Test upload + export pipeline
- [ ] Brief team on Roboflow annotation workflow
- [ ] Launch first batch on Roboflow (~40 remaining images)

