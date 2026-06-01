# Roadmap: Side View 19KP → Production

_Decisions locked in grill-with-docs session, 2026-05-26._  
_Finish line: backend serving real 19KP keypoints end-to-end (geometry API returns actual CV measurements)._

---

## Decision log (9 questions)

| Q | Decision |
|---|----------|
| Q1 — Finish line | (c) Backend serving real 19KP keypoints end-to-end |
| Q2 — Training set | 79 canonical JSONs now; don't wait for 5 drafts |
| Q3 — Holdout source | Carve ~12 from the 79; ~67 train / ~12 holdout (85/15) |
| Q4 — Gate criterion | All 19 KPs detected above confidence threshold = PASS per image |
| Q5 — B3 scope | Model swap + `_POSE_19KP_LABELS` list; `analysis.py` geometry already aligned |
| Q6 — Label order SOT | `DEFAULT_KP_ORDER` in `sdi-helper/yolo_training/labelme_to_yolo_pose.py` |
| Q7 — flip_idx | Defined and enforced in live/staging dataset configs: `[0,2,1,4,3,7,8,5,6,10,9,11,13,12,15,14,17,16,18]` |
| Q8 — Training location | Colab GPU; `best.pt` is the handoff artefact |
| Q9 — evaluate_19kp_holdout.py | Part of B2 definition of done; not deferred |

---

## Critical path to production

```
B2 — Train & gate 19KP model
│
├── [DONE] Update dataset_pose.yaml (both live + _colab_staging)
│   kpt_shape: [19, 3]
│   flip_idx: [0, 2, 1, 4, 3, 7, 8, 5, 6, 10, 9, 11, 13, 12, 15, 14, 17, 16, 18]
│
├── Run convert_accepted_19kp_dataset on 79 canonical JSONs
│   → ~67 train / ~12 holdout split
│   → populate _colab_staging/{images,labels}/
│   → write holdout_manifest.txt
│
├── Update _colab_staging/dataset_pose.yaml to 19KP config
│
├── Upload _colab_staging/ to Colab (Google Drive / zip)
│
├── Colab training run → best.pt
│   copy to: yolo_training/runs/side_view_pose_19kp_candidate/weights/best.pt
│
├── [DONE] scripts/evaluate_19kp_holdout.py
│   → load best.pt, run on holdout images
│   → write prediction_summary.csv (per-image PASS/FAIL)
│
└── make side-19kp-gate
    → gate_decision.json { "decision": "PASS" }

B3 — Wire 19KP model to backend
│
├── vehicle-sdi-system/cv_service/yolo_cv_client.py
│   → add _POSE_19KP_LABELS tuple (mirrors DEFAULT_KP_ORDER exactly)
│   → update _extract_pose_keypoints to use 19KP labels when model is 19KP
│
├── Cross-repo integration test
│   → assert _POSE_19KP_LABELS == DEFAULT_KP_ORDER
│
└── Deploy best.pt to vehicle-sdi-system
    → geometry API returns real CV measurements ← FINISH LINE
```

---

## ADRs written

- [ADR-001](ADR-001-19kp-label-order-contract.md) — 19KP label order contract (`DEFAULT_KP_ORDER` is SOT)
- [ADR-002](ADR-002-19kp-flip-idx-derivation.md) — flip_idx must be derived from DEFAULT_KP_ORDER; hardcoded value committed before training
- [ADR-003](ADR-003-19kp-holdout-strategy.md) — holdout carved from canonical 79; `evaluate_19kp_holdout.py` is B2 deliverable

---

## Pending human tasks (not agent-actionable)

- 5 draft 19KP JSONs in `b1_19kp_labeling_queue/labelme_json_draft_19kp/` need correction in LabelMe
- `f591a494` has mislabeled `front_bumper` near B-pillar — needs re-labeling
- Merge `chore-b1-review-gates` → `master` on `riorus94/sdi-helper`

---

## Key file references

| File | Status | Action required |
|------|--------|-----------------|
| `sdi-helper/yolo_training/labelme_to_yolo_pose.py` | ✅ SOT for label order | None |
| `sdi-helper/yolo_training/side_view_dataset/dataset_pose.yaml` | ✅ 19KP | None |
| `sdi-helper/yolo_training/side_view_dataset/_colab_staging/dataset_pose.yaml` | ✅ 19KP | None |
| `sdi-helper/scripts/evaluate_19kp_holdout.py` | ✅ Exists | Run after Colab `best.pt` handoff |
| `sdi-helper/scripts/gate_side_view_19kp_candidate.py` | ✅ Exists | None |
| `vehicle-sdi-system/cv_service/yolo_cv_client.py` | ⚠️ Uses 7KP labels | Add `_POSE_19KP_LABELS` (B3) |
| `vehicle-sdi-system/vehicle_sdi/api/v1/analysis.py` | ✅ Already has 19KP dim pairs | None |
