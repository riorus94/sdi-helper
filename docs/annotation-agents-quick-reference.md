# B1 Annotation: 4 AI Agents Architecture (Quick Reference)

> Designed for 1 human + 4 AI agents → 100+ labeled images by May 18

---

## Agent 1: Keypoint Suggester (Priority 1 — Start May 16)

**Goal**: Pre-label raw images with initial keypoint guesses  
**Input**: `dataset_raw/images/train/side/*.jpg`  
**Output**: `yolo_training/side_view_dataset/labelme_json/*.json` (pre-labeled)

### Approach
1. **Load Phase 1 wheel detections** (existing `cv_service` wheels model)
2. **Apply geometric heuristics** (from `docs/geometry.md`):
   - Wheels → infer vehicle center line & orientation
   - From wheel points → estimate roof apex, bumpers, fenders
   - Use car aspect ratio (length:height ~2.5:1) to scale keypoint estimates

3. **Generate LabelMe JSON**:
   ```python
   # Pseudocode
   def suggest_keypoints(image_path):
       wheels = detect_wheels(image_path)  # Phase 1 model
       car_bounds = estimate_bounding_box(wheels)
       keypoints = {}
       for kp_name in KEYPOINT_NAMES:
           kp_pos = heuristic_position(kp_name, car_bounds)
           confidence = estimate_confidence(kp_name, wheels)
           keypoints[kp_name] = (kp_pos, confidence)
       return save_labelme_json(image_path, keypoints)
   ```

### File: `scripts/suggest_keypoints.py`
**Dependencies**:
- Phase 1 model weights (already in `cv_service/models/`)
- `sdi_helper/domain/geometry/__init__.py` (geometry rules)
- `ultralytics` (YOLO inference)
- `labelme` (JSON serialization)

**Test Case**:
```bash
cd d:\project\sdi-helper
python scripts/suggest_keypoints.py --image-dir dataset_raw/images/train/side --output yolo_training/side_view_dataset/labelme_json --batch batch_006
```

**Success Metric**: 
- Average confidence > 0.70
- Human correction time < 3 min per image

---

## Agent 2: Geometry Validator (Implemented — May 17)

**Goal**: Check if labeled keypoints make geometric sense  
**Input**: `labelme_json/*.json`  
**Output**: Validation report + optionally corrected JSONs

### Constraints to Check
1. **Wheel alignment** (horizontal distance ±5% tolerance)
   - `|rear_wheel_ground.y - front_wheel_ground.y| < 50 px`

2. **Roof apex geometry** (above all other points)
   - `roof_apex.y < min(hood_edge.y, all_bumpers.y)`

3. **Bumper order** (rear before front, left-to-right)
   - `rear_bumper.x < front_bumper.x`

4. **Keypoint distances** (within car geometry bounds)
   - `distance(rear_wheel, front_wheel) ∈ [1500, 5000] px` (estimated)
   - `distance(roof_apex, ground_ref) ∈ [400, 1500] px`

5. **Confidence filtering** (if from Agent 1)
   - Flag keypoints with confidence < 0.5

### File: `scripts/validate_keypoints.py`
**Dependencies**:
- `sdi_helper/domain/geometry/__init__.py` (geometry module)
- `labelme` (JSON parsing)

**Current behavior**:
- Parses LabelMe point JSONs and emits a CSV triage report.
- Flags missing required keypoints, wheel misalignment, roof clearance issues, perspective distortion, and Agent 1 low-confidence metadata when present.
- Reads confidence from LabelMe shape descriptions like `confidence=0.908`.

**Test Case**:
```bash
python scripts/validate_keypoints.py --json-dir yolo_training/side_view_dataset/labelme_json --report validation_report.csv
```

**Output Example**:
```
batch_006_001.json: ✓ VALID (0 warnings)
batch_006_002.json: ✓ VALID (1 warning: front_wheel_center confidence 0.65 < threshold)
batch_006_003.json: ⚠️ REVIEW (roof_apex.y > hood_edge.y — possible mislabel)
```

**Success Metric**:
- <5% false positives (valid images marked invalid)
- 100% catch rate on geometric impossibilities

---

## Agent 3: Orchestrator (Priority 3 — May 17)

**Goal**: Automate the full batch pipeline  
**Input**: Batch number (e.g., `006`)  
**Output**: Trained model + metrics

### Workflow Automation
1. **Prepare batch** → Call `start_side_annotation_batch.py`
2. **Pre-label** → Call Agent 1
3. **Validate** → Call Agent 2
4. **Report** → Summary of pre-labeling quality
5. **Wait for human correction** → Human opens LabelMe, corrects JSONs
6. **Detect completion** → Monitor `labelme_json/` for new/modified files
7. **Convert format** → Call `labelme_to_yolo_pose.py`
8. **Train** → Call `train_pose.py`
9. **Log metrics** → Save training results + feedback

### File: `scripts/orchestrate_annotation.py`
**Dependencies**:
- All previous agents
- `labelme_to_yolo_pose.py` (conversion)
- `train_pose.py` (training)

**Usage**:
```bash
python scripts/orchestrate_annotation.py --batch batch_006 --wait-for-human
# 1. Pre-labels batch_006
# 2. Prints: "Ready for human correction. When done, press Enter..."
# 3. Waits for Enter
# 4. Converts JSONs → YOLO format
# 5. Trains model
# 6. Reports metrics
```

**Config File** (`config/orchestration.yaml`):
```yaml
batch_size: 10
keypoint_confidence_threshold: 0.5
geometric_validation_mode: "strict"  # or "lenient"
auto_correct: false  # Agent 2 suggests but doesn't auto-fix
training_params:
  epochs: 50
  batch_size: 16
  patience: 10
```

**Success Metric**:
- Batch completes end-to-end without user intervention (except correction step)
- Training logs saved to `logs/batch_006_training.log`

---

## Agent 4: Annotation Advisor (Priority 4 — May 18)

**Goal**: Analyze training results and recommend improvements  
**Input**: Training metrics + annotation logs  
**Output**: Advisor report + next batch recommendations

### Analysis
1. **Per-keypoint metrics** (from `runs/detect/trainN/results.csv`):
   - Precision, recall, mAP for each of 19 keypoints
   - Identify "hard" keypoints (precision < 0.75)

2. **Error analysis**:
   - Which images have highest loss?
   - Which batches are better/worse?
   - Do any images have systematic errors?

3. **Recommendations**:
   - "Focus next batch on side-window geometry (precision 0.68)"
   - "Consider re-reviewing these 3 images from batch_006 (high loss)"
   - "Training converged well; increase epochs to 70 next time"

4. **Next batch preview**:
   - Which 10 images to annotate next for best improvement?
   - Suggest annotation order (harder images first for faster feedback)

### File: `scripts/advise_annotation.py`
**Dependencies**:
- Training metrics parser
- Annotation logs
- Geometric domain knowledge

**Usage**:
```bash
python scripts/advise_annotation.py --run-dir yolo_training/runs/detect/train1 --advice-output advice_batch_006.md
```

**Output Example**:
```markdown
# Annotation Advice after Batch 006

## Hard Keypoints (precision < 0.75)
- side_window_top_front: 0.68 (10 mislabels detected)
- hood_edge: 0.71 (8 mislabels)
- windshield_base: 0.74 (5 mislabels)

## Recommendation
Focus batch_007 on images with strong side-window and hood geometry.
Consider pairing each image with a reference photo.

## Next Batch Preview (Batch 007)
Top 10 candidates for maximum improvement:
  1. batch_007_003.jpg (estimated improvement: +0.05 precision)
  2. batch_007_008.jpg (estimated improvement: +0.04 precision)
  ...
```

**Success Metric**:
- Report identifies 2–3 actionable improvements
- Subsequent batch shows measurable improvement (e.g., precision +1–2%)

---

## Integration: How Agents Work Together

```
Day 1 (May 16)
└── Agent 1: Pre-label batch_006 (2h build + test)
└── Human: Correct batch_006 in LabelMe (30 min)
└── Agent 3: Convert & train on 63 images (1h)
└── Metrics logged

Day 2 (May 17)
└── Agent 2: Validate all 63 JSONs (30 min)
└── Agent 1 + Agent 3: Pre-label + process batch_007 (30 min)
└── Human: Correct batch_007 in LabelMe (30 min)
└── Agent 3: Convert & train on 73 images (1h)
└── Agent 4: Analyze metrics & recommend next steps (30 min)

Day 3 (May 18)
└── Agent 1 + Agent 3: Pre-label & process batches 008–010 (1h)
└── Human: Correct all 30 images (1h)
└── Agent 3: Convert & train on 100+ images (1h)
└── Agent 4: Final report + B2 ready for integration (30 min)

Result: 100+ labeled images + fully automated pipeline
```

---

## Implementation Checklist

- [ ] **Agent 1 Build** (4h)
  - [ ] Load Phase 1 wheel model
  - [ ] Implement geometry heuristics
  - [ ] Generate LabelMe JSON format
  - [ ] Test on 5 sample images

- [ ] **Agent 2 Build** (3h)
  - [ ] Define geometric constraints
  - [ ] Implement validation checks
  - [ ] Generate validation report
  - [ ] Test on existing 63 JSONs

- [ ] **Agent 3 Build** (6h)
  - [ ] Integrate Agents 1 & 2
  - [ ] Automate format conversion
  - [ ] Automate training trigger
  - [ ] Log metrics to file

- [ ] **Agent 4 Build** (4h)
  - [ ] Parse YOLO metrics
  - [ ] Implement per-keypoint analysis
  - [ ] Generate advisor report
  - [ ] Test on batch_006 results

**Total build time**: ~17 hours (can parallelize non-blocking work)

---

## Quick Start (May 16 Afternoon)

**Start with Agent 1** (highest ROI):

```bash
cd d:\project\sdi-helper
git checkout -b feature/annotation-agents
mkdir scripts/agents
touch scripts/agents/__init__.py scripts/suggest_keypoints.py

# In suggest_keypoints.py, start with:
# 1. Load Phase 1 model from cv_service/models/
# 2. Detect wheels on batch_006 samples
# 3. Apply simple heuristic: roof_apex ≈ (center_x, top_y - margin)
# 4. Generate LabelMe JSON
# 5. Test on 2–3 images
```

Then move to Agent 3 (integration) to close the loop.

Ready to start?

