# B1 Annotation: AI-Assisted Human-in-the-Loop Workflow

**Revised approach**: No additional humans needed. Use **agentic AI to accelerate B1** via pre-labeling, validation, and orchestration.

> **New timeline**: May 16–18 (same 2 days, but with AI assistance)  
> **Human time**: ~1–2 hours (correction only, not full labeling)  
> **Result**: 100+ labeled images with higher consistency

---

## Why Agentic AI > Parallel Humans for B1

| Metric | Parallel Humans (unavailable) | AI-Assisted (1 human) |
|--------|------|------|
| **Available** | ❌ No team members | ✅ Yes, immediately |
| **Setup time** | 3h | 4h (worth it) |
| **Time per image** | 5–8 min | 2–3 min (correction only) |
| **Consistency** | Medium (varies) | High (AI baseline) |
| **QA/feedback** | Manual | Automated |
| **Scalability** | Limited | Unlimited (more agents) |
| **Total time for 50 images** | 4–6h | 2–3h |

---

## Proposed AI Agent Architecture

### Agent 1: Keypoint Suggester
**Purpose**: Pre-label images with initial keypoint guesses  
**Input**: Raw side-view image  
**Process**:
1. Detect vehicle bounding box (existing wheel detection)
2. Use a Pareto-first keypoint split so we do not try to solve all 19 points at once
3. Apply geometric constraints (roof apex above hood, wheels aligned horizontally)
4. Generate LabelMe JSON with confidence scores

**Output**: LabelMe JSON + confidence per keypoint (0–1.0)

**Example**:
```json
{
  "imagePath": "batch_006_001.jpg",
  "shapes": [
    {"label": "roof_apex", "points": [[640, 200]], "confidence": 0.75},
    {"label": "rear_bumper", "points": [[100, 450]], "confidence": 0.9},
    ...
  ]
}
```

**Effort**: 4h to build (rules-based + CV model integration)

**Pareto split**:
- Phase 1: wheel anchors + ground reference
  - `front_wheel_center`, `front_wheel_ground`, `rear_wheel_center`, `rear_wheel_ground`, `ground_ref`
- Phase 2: body outline and roofline
  - `front_bumper`, `rear_bumper`, `roof_apex`, `hood_edge`, `windshield_base`, `rear_glass_base`, `side_window_top_front`, `side_window_top_rear`
- Phase 3: detailed contour points
  - `fender_arch_front`, `fender_arch_rear`, `body_waist_front`, `body_waist_rear`, `panel_front`, `panel_rear`

This keeps the first pass small and high-value, then expands only after the core geometry is stable.

---

### Agent 2: Geometry Validator
**Purpose**: Check if labeled keypoints make geometric sense  
**Input**: LabelMe JSON with labeled keypoints  
**Process**:
1. Load 19 keypoints from JSON
2. Check geometric constraints:
   - Wheels aligned horizontally (±10% tolerance)
   - Roof apex above hood (always higher Y-coord)
   - Bumpers below fenders
   - Keypoint distances within expected range (car dimensions 3–6m typical)
3. Flag outliers & suspicious patterns
4. Optionally auto-correct obvious errors (e.g., swap swapped wheel points)

**Output**: Validation report + corrected JSON (optional)

**Example**:
```
{
  "filename": "batch_006_001.jpg",
  "valid": true,
  "warnings": [
    "front_wheel_center close to rear_wheel_center (possible mislabel?)",
    "roof_apex confidence low (0.65), consider manual review"
  ],
  "corrected_count": 0
}
```

**Effort**: 3h to build (geometry rules from `docs/geometry.md`)

---

### Agent 3: Pipeline Orchestrator
**Purpose**: Automate batch workflow (prep → pre-label → validate → convert → train)  
**Input**: Batch number (e.g., 006)  
**Process**:
1. Load images from `annotation_batches/batch_006/images`
2. Call Agent 1 (Keypoint Suggester) for each image → generate pre-labeled JSONs
3. Save JSONs to `labelme_json/`
4. Call Agent 2 (Geometry Validator) on each JSON
5. Generate summary report (% confident, % flagged)
6. Wait for human correction (LabelMe GUI)
7. On completion:
   - Call `labelme_to_yolo_pose.py`
   - Upload to Roboflow (optional)
   - Trigger `train_pose.py`
   - Log metrics

**Output**: Trained model + metrics report

**Effort**: 6h to build (integrates existing scripts)

---

### Agent 4: Annotation Advisor
**Purpose**: Analyze training metrics and recommend next steps  
**Input**: Training metrics (precision/recall per keypoint), annotation logs  
**Process**:
1. Parse `runs/detect/trainN/results.csv` (per-class metrics)
2. Identify "hard" keypoints (precision < 0.75)
3. Identify "bad" annotations (high loss on specific images)
4. Recommend:
   - Which keypoints need re-annotation focus
   - Which images should be re-reviewed
   - Training hyperparameter tweaks
5. Generate next batch preview (which images to annotate next)

**Output**: Advisor report + recommended actions

**Example**:
```
Hard Keypoints (precision < 0.75):
  - side_window_top_front: 0.68
  - hood_edge: 0.71
  
Recommendation:
  Focus next batch on side-window + hood geometry.
  Consider reviewing 5 sample images from this batch for quality.

Next batch suggestion:
  batch_007: 10 images (predicted improvement: +2% overall precision)
```

**Effort**: 4h to build (metrics parsing + heuristics)

---

## Revised B1 Timeline (With AI Agents)

### Day 1: May 16 (Today) — Setup
- [ ] **2h**: Build Agent 1 (Keypoint Suggester)
  - Input: Phase 1 wheel model + geometry heuristics
  - Output: Pre-labeled JSONs for batch_006
  - Test on 2–3 images
  
- [ ] **1h**: Finish batch_006 (10 images) in LabelMe
  - Load pre-labeled JSONs
  - Correct/verify keypoints (~2 min per image instead of 8)
  - Result: 63 labeled total

### Day 2: May 17 — Validation & Feedback
- [ ] **1h**: Build Agent 2 (Geometry Validator)
  - Run on all 63 existing LabelMe JSONs
  - Generate validation report
  - Flag/correct any inconsistencies

- [ ] **1h**: Run Agent 3 (Orchestrator) on batch_007 (10 images)
  - Pre-label with Agent 1
  - Validate with Agent 2
  - Display summary (confidence, warnings)

- [ ] **1h**: Human correction on batch_007
  - Review pre-labeled JSON
  - Adjust 2–3 keypoints per image
  - Result: 73 labeled total

- [ ] **30 min**: Convert batch_007 → YOLO + train
  - Call `labelme_to_yolo_pose.py`
  - Run `train_pose.py` on 73 images
  - Log metrics

### Day 3: May 18 — Scaling & Feedback Loop
- [ ] **30 min**: Build Agent 4 (Annotation Advisor)
  - Parse training metrics
  - Identify hard keypoints & bad annotations
  - Recommend focus areas

- [ ] **2h**: Batches 008–009 (20 images)
  - Pre-label + validate + correct (faster, ~2–3 min per image)
  - Result: 93 labeled

- [ ] **1h**: Final batch (remaining 10 images) + retrain
  - Result: 103 labeled
  - Full Phase 2 dataset ready

### Result by EOD May 18
✅ **100+ labeled images**  
✅ **All 4 AI agents operational**  
✅ **Trained body_pose model ready for integration**  
✅ **Annotation feedback loop established**

---

## Implementation Plan

### Phase A: Build Agents (May 16–17, ~8 hours developer time)

| Agent | File | Dependencies | Est. Time |
|-------|------|--------------|-----------|
| 1. Suggester | `scripts/suggest_keypoints.py` | Phase 1 model, geometry rules | 4h |
| 2. Validator | `scripts/validate_keypoints.py` | `docs/geometry.md`, geometry module | 3h |
| 3. Orchestrator | `scripts/orchestrate_annotation.py` | Agents 1–2, labelme_to_yolo, train_pose | 6h |
| 4. Advisor | `scripts/advise_annotation.py` | Training metrics parser | 4h |

**Total agent development**: ~16 hours (staggered, non-blocking)

### Phase B: Use Agents (May 16–18, ~2–3 hours human time)

**Human effort**: Correction only, not full labeling
- Batch 006: 10 images × 2 min = 20 min
- Batch 007: 10 images × 2 min = 20 min
- Batches 008–009: 20 images × 2 min = 40 min
- Batch 010: 10 images × 2 min = 20 min
- **Total**: ~2 hours

---

## Key Assumptions & Risks

### ✅ Assumptions (Likely)
- Phase 1 wheel detection is robust enough for geometric constraints
- Geometric rules in `docs/geometry.md` are comprehensive
- Pre-labeled confidence scores are useful signals
- Human correction is 3–4× faster than full annotation

### ⚠️ Risks (Mitigations)
| Risk | Mitigation |
|------|-----------|
| Agent 1 pre-labels poorly | Start with small batch (006), validate quality before scaling |
| Geometric constraints too strict | Run Agent 2 in "validation-only" mode, let human decide |
| Human correction slower than expected | Adjust agent confidence thresholds (show only uncertain points) |
| Agent training overhead | Parallelize: agent development ≠ human annotation time |

---

## Success Criteria

✅ **Phase 1 (Agent 1 Ready)**
- [ ] Pre-labeled images have >70% average confidence
- [ ] Spot-check: 5/10 batch_006 images are >80% correct
- [ ] Correction time <3 min per image

✅ **Phase 2 (Agent 2 Ready)**
- [ ] Validator flags <5% false positives (good images marked bad)
- [ ] Geometric constraints match reality (sample 10 images)

✅ **Phase 3 (Agent 3 Ready)**
- [ ] Orchestrator runs end-to-end: pre-label → validate → convert → train
- [ ] Training completes without errors
- [ ] Metrics logged & analyzed

✅ **Phase 4 (Agent 4 Ready)**
- [ ] Advisor report is actionable (identifies 2–3 hard keypoints)
- [ ] Next-batch recommendations improve overall precision
- [ ] Annotation feedback loop operational

✅ **Overall B1 Complete**
- [ ] 100+ side-view images labeled
- [ ] All agents operational & documented
- [ ] B2 retraining ready (May 18 EOD)

---

## Files to Create/Modify

**New scripts** (in `sdi-helper/scripts/`):
- `suggest_keypoints.py` — Agent 1
- `validate_keypoints.py` — Agent 2
- `orchestrate_annotation.py` — Agent 3
- `advise_annotation.py` — Agent 4

**New documentation** (in `sdi-helper/docs/`):
- `annotation-agents-guide.md` (this file)
- `agent-development-log.md` (progress tracking)

**Modified files**:
- `sdi_helper/domain/geometry/__init__.py` — add validator functions
- `Makefile` — add targets: `make suggest-keypoints`, `make validate`, `make orchestrate-batch`

---

## Next Steps

1. **Decision**: Build agents in-house vs. use existing tools (e.g., LabelImg API, cloud vision APIs)?
2. **Agent 1 design**: Should we use existing YOLO wheels model + heuristics, or train a lightweight keypoint model?
3. **Priority**: Start with Agent 1 (highest ROI), then 3 → 2 → 4

Ready to start coding Agent 1 (Keypoint Suggester)?

