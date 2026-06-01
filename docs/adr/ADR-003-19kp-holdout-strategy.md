# ADR-003 — 19KP Holdout Set Strategy

**Status:** Accepted  
**Date:** 2026-05-26  
**Repos affected:** `sdi-helper`

---

## Context

The gate `make side-19kp-gate` requires a `prediction_summary.csv` produced by running
the trained 19KP model against a held-out evaluation set. The existing holdout manifest
(`yolo_training/side_view_dataset/holdout/`) contains images that were labelled under
the 7KP schema and have no 19KP canonical annotations. Generating 19KP labels for
those images would require full re-annotation effort that is not justified given the
79 canonical 19KP JSONs already available.

## Decision

**The 19KP holdout set is carved from the canonical 79-image LabelMe set at dataset
preparation time using an approximately 85/15 train/holdout split.**

- Target split: ~67 training images / ~12 holdout images.
- Split is performed by `convert_accepted_19kp_dataset` (or an equivalent preparation
  script) at the time labels are converted to YOLO format.
- The holdout partition is written to `_colab_staging/labels/holdout/` and the
  corresponding images to `_colab_staging/images/holdout/`.
- A `holdout_manifest.txt` listing holdout image filenames is written to
  `yolo_training/runs/side_view_pose_19kp_candidate/holdout_manifest.txt` and passed
  to the gate as evidence.
- The existing 7KP holdout manifests are **not reused or merged** into the 19KP
  evaluation set.

### Gate criterion (from Q4 decision)

A holdout image **passes** if the trained model detects all 19 keypoints with
confidence ≥ `_CONF_THRESHOLD` (defined in `scripts/gate_side_view_19kp_candidate.py`).
`prediction_summary.csv` records one row per image with a PASS/FAIL verdict.
The gate blocks promotion if any row is FAIL.

### evaluate_19kp_holdout.py (B2 deliverable)

The script `scripts/evaluate_19kp_holdout.py` must be written as part of B2. It:

1. Loads the Colab-trained `best.pt`.
2. Runs inference on each image listed in `holdout_manifest.txt`.
3. For each image, checks that all 19 keypoints are present with confidence ≥ threshold.
4. Writes `prediction_summary.csv` with columns: `image`, `kps_detected`, `min_conf`,
   `verdict` (PASS/FAIL).

This script is the B2 definition-of-done gate; training is not complete until
`make side-19kp-gate` produces a `gate_decision.json` with `decision: PASS`.

## Consequences

- With 79 images total, the holdout set is small (~12 images). This is acceptable for
  a B2 gate — it validates that the model generalises beyond training data, not that
  it achieves benchmark-level mAP.
- When additional 19KP canonical JSONs are available, the split should be re-run
  (not patched) to maintain a clean partition.
- The 5 draft JSONs currently in `b1_19kp_labeling_queue/labelme_json_draft_19kp/`
  count toward neither training nor holdout until they pass human correction and are
  promoted to the canonical set.
