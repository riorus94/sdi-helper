# Side-View 7KP Bumper Validation

**Status:** Completed  
**Owner:** Sage  
**Sprint:** Sprint 3 Gate A

## Objective

Validate whether `yolo_training/runs/side_view_pose_7kp_bumper_smoke/weights/best.pt` is reliable enough on out-of-sample side-view images to justify promotion.

## Holdout Requirements

- True side-view vehicles only.
- Front and rear bumpers visible.
- Exclude images known to be part of the 63 clean in-frame smoke training slice.
- Prefer mixed makes, crops, lighting, and mild occlusion.

## Review Procedure

1. Define the holdout slice and record its source.
2. Run inference with the 7KP bumper smoke model.
3. Review every image for:
   - front bumper localization
   - rear bumper localization
   - front/rear swap behavior
   - catastrophic misses
   - confidence behavior vs visual quality
4. Record a verdict per image: `pass`, `borderline`, or `fail`.
5. Summarize error taxonomy and final decision: `promote`, `hold`, or `reject`.

## Failure Taxonomy

- `swap_front_rear`
- `front_bumper_miss`
- `rear_bumper_miss`
- `catastrophic_miss`
- `crop_or_visibility_issue`
- `non_true_side_view`

## Evidence Log

| Image | Source | Verdict | Failure Tags | Notes |
|---|---|---|---|---|
| 01 stanford_raw_00058_cars_test_00060.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss | FB lands inside body; not at front body end |
| 02 stanford_raw_00132_cars_test_00140.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Both bumper points land on side/body area |
| 03 stanford_raw_00156_cars_test_00164.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss | FB is inside body near wheel, not body end |
| 04 stanford_raw_00159_cars_test_00167.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Both bumper endpoints are interior |
| 05 stanford_raw_00166_cars_test_00174.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss | FB/RB are not anchored to visible body ends |
| 06 stanford_raw_00195_cars_test_00203.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Interior side-body predictions |
| 07 stanford_raw_00210_cars_test_00218.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Van bumper endpoints missed |
| 08 stanford_raw_00212_cars_test_00220.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss | FB predicted at body center, not front end |
| 09 stanford_raw_00247_cars_test_00255.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss | FB/RB cluster around body/wheel area |
| 10 stanford_raw_00256_cars_test_00264.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Both bumper endpoints are interior |
| 11 stanford_raw_00258_cars_test_00266.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss | FB is inside body, not front bumper |
| 12 stanford_raw_00263_cars_test_00271.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Jeep body ends missed |
| 13 stanford_raw_00280_cars_test_00288.jpg | `dataset_raw/images/train/side` holdout | fail | catastrophic_miss | Rendered review does not show usable bumper localization |
| 14 stanford_raw_00353_cars_test_00362.jpg | `dataset_raw/images/train/side` holdout | fail | rear_bumper_miss | FB/RB assignment is not usable for length line |
| 15 stanford_raw_00440_cars_test_00450.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Bumper points land near rear quarter/interior |
| 16 stanford_raw_00474_cars_test_00485.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Bumper endpoints missed on pickup |
| 17 stanford_raw_00493_cars_test_00505.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Points are interior, not body ends |
| 18 stanford_raw_00501_cars_test_00513.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Truck bed/front endpoints missed |
| 19 stanford_raw_00518_cars_test_00530.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Convertible body ends missed |
| 20 stanford_raw_00527_cars_test_00540.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Bumper predictions are interior |
| 21 stanford_raw_00554_cars_test_00567.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss | Front bumper point not on front end |
| 22 stanford_raw_00570_cars_test_00583.jpg | `dataset_raw/images/train/side` holdout | fail | crop_or_visibility_issue, front_bumper_miss | Non-standard small car crop; endpoints unreliable |
| 23 stanford_raw_00617_cars_test_00631.jpg | `dataset_raw/images/train/side` holdout | fail | catastrophic_miss | No usable bumper localization in review overlay |
| 24 stanford_raw_00626_cars_test_00640.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Pickup endpoints missed |
| 25 stanford_raw_00715_cars_test_00729.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Roadster bumper points are interior |
| 26 stanford_raw_00741_cars_test_00755.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Bumper points are not at body ends |
| 27 stanford_raw_00748_cars_test_00762.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | SUV endpoints missed |
| 28 stanford_raw_00771_cars_test_00786.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Bumper points are unreliable |
| 29 stanford_raw_00793_cars_test_00809.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Coupe endpoints missed |
| 30 stanford_raw_00808_cars_test_00824.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Bumper points are interior |
| 31 stanford_raw_00969_cars_test_00989.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss | Front bumper point is not at front end |
| 32 stanford_raw_00985_cars_test_01006.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Bumper endpoints missed |
| 33 stanford_raw_00987_cars_test_01008.jpg | `dataset_raw/images/train/side` holdout | fail | front_bumper_miss, rear_bumper_miss | Non-true side/perspective plus endpoint miss |

## Run Artifacts

- Inference output: `yolo_training/runs/side_view_pose_7kp_bumper_oos_20260524/`
- Holdout manifest: `yolo_training/runs/side_view_pose_7kp_bumper_oos_20260524/holdout_manifest.txt`
- Prediction CSV: `yolo_training/runs/side_view_pose_7kp_bumper_oos_20260524/prediction_summary.csv`
- Bumper review CSV: `yolo_training/runs/side_view_pose_7kp_bumper_oos_20260524/bumper_review_summary.csv`
- Clean review sheet: `yolo_training/runs/side_view_pose_7kp_bumper_oos_20260524/bumper_review_contact_sheet.jpg`

## Summary

- Holdout size: 33 images.
- Detections written: 33 label files.
- Bumper minimum confidence range: 0.4942-0.7269, average 0.5748.
- Strict geometry sanity check: 0/33 images had both front and rear bumper points outside their corresponding wheel centers.
- Visual review: bumper points repeatedly land inside the body or near wheels instead of at visible body ends.

## Final Decision

- Decision: Reject promotion.
- Rationale: The 7KP bumper smoke checkpoint is not reliable enough for front/rear overhang or overall length confidence. Do not promote this run to backend line-confidence work, and do not spend GPU/Colab retraining time on the same recipe until the bumper label recipe is corrected.

## Follow-Up Gate

- Added `scripts/validate_7kp_body_end_labels.py` to block 7KP labels where `front_bumper`/`rear_bumper` are not outside the semantic wheel centers for the inferred vehicle-facing direction.
- Added learned-prior guards so Agent 1 ignores body-end priors that place `front_bumper` inside the front wheel or `rear_bumper` inside the rear wheel in canonical side-view space.
- Current 63-label smoke subset result: 0 valid, 63 invalid; all 63 fail `rear_endpoint_inside_body`.
- Rebuilt corrected labels from trusted wheel/contact points:
  - `yolo_training/side_view_dataset/labelme_json_7kp_bumper_corrected_20260524/`: 50 written, 13 skipped for out-of-frame recomputed body ends.
  - `yolo_training/side_view_dataset/labelme_json_7kp_bumper_corrected_valid_20260524/`: 41 fully valid labels after ratio QA.
  - `yolo_training/side_view_dataset/labels_pose_7kp_bumper_corrected_valid_20260524/`: 41 converted YOLO-pose labels.
- Local full-source 30-epoch recovery train:
  - Run: `yolo_training/runs/side_view_pose_7kp_bumper_corrected_valid_fullsrc_30ep_20260524/`
  - Dataset split: train=32, val=9.
  - Internal validation: pose precision 0.991, recall 1.000, mAP50 0.995, mAP50-95 0.392.
- Corrected-checkpoint out-of-sample gate:
  - Run: `yolo_training/runs/side_view_pose_7kp_bumper_corrected_valid_30ep_oos_20260524/`
  - Result: 0/33 pass strict body-end geometry.
  - Failure taxonomy: 33/33 `rear_endpoint_inside_body`, 29/33 `front_endpoint_inside_body`.

## Corrected Checkpoint Decision

- Decision: Reject promotion.
- Rationale: The label recipe and QA gate are fixed, but the small corrected 7KP subset still does not generalize to out-of-sample endpoint localization. Internal validation is over-optimistic on 41 labels.
- Next step: do not spend more local epochs on this subset. Build a larger, manually verified body-end set with hard holdout-like examples, or keep bumper endpoints as geometry-derived estimates until enough real endpoint labels exist.
