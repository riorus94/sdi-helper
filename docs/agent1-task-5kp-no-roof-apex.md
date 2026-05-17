# Agent 1 Task: Labeling 5 Keypoints (No roof_apex)

## Objective
Generate pre-label suggestions for side-view vehicle images using exactly 5 keypoints and explicitly exclude roof_apex.

## Scope
- Agent: Agent 1 (Keypoint Suggester)
- Input images: side-view images only
- Output: LabelMe JSON suggestion files
- Keypoint count: exactly 5

## Allowed Keypoints
1. ground_ref
2. front_wheel_center
3. front_wheel_ground
4. rear_wheel_center
5. rear_wheel_ground

## Strict Rules
1. Do not add roof_apex.
2. Do not add any keypoint outside the 5 allowed labels.
3. If a wheel is occluded, still estimate wheel center/ground conservatively and set lower confidence in shape description metadata.
4. Keep left-to-right consistency:
   - rear_wheel_* should be left of front_wheel_* for standard side orientation.
5. Keep wheel-ground y positions nearly aligned unless clear slope is visible.

## Runtime Config
Use this priority config file:
- config/agent1_keypoint_priority_5kp_no_roof_apex.json

## Suggested Command
Run from repository root:

```powershell
python scripts/suggest_keypoints.py \
  --image-dir dataset_raw/images/train/side \
  --output yolo_training/side_view_dataset/labelme_json_stanford_screening \
  --priority-config config/agent1_keypoint_priority_5kp_no_roof_apex.json \
  --phase-only phase1 \
  --orientation-classifier clip \
  --overwrite
```

The CLIP orientation option prompts for left-looking vs right-looking side
profiles before assigning detected wheel boxes to front/rear labels. If CLIP is
not confident enough, Agent 1 keeps the conservative right-looking assignment
and records the CLIP score in the quality report for human review.

## 9KP Upgrade Path

After the 5KP ground-ref labels are clean, Agent 1 can generate a 9-keypoint
side-view draft with wheel anchors, ground_ref, fender arches, and bumpers:

```powershell
python scripts/suggest_keypoints.py \
  --image-dir dataset_raw/images/train/side \
  --output yolo_training/side_view_dataset/labelme_json_9kp_side \
  --priority-config config/agent1_keypoint_priority_9kp_side.json \
  --phase-only phase1 \
  --orientation-classifier clip \
  --quality-report logs/agent1_9kp_side_quality.csv \
  --overwrite
```

The 9KP output should still go through human review, especially for
`front_bumper` and `rear_bumper` on cropped or non-90-degree images.

## Validation Gate (Mandatory)
After generation, verify every JSON contains only the 5 labels above.

## Review Priority Semantics
- REVIEW_LOW: image/keypoint suggestion quality is good, lowest manual review urgency.
- REVIEW_MEDIUM: acceptable but should be checked.
- REVIEW_HIGH: likely problematic and should be reviewed first.

Example quick check:

```powershell
python scripts/validate_keypoints.py \
  --json-dir yolo_training/side_view_dataset/labelme_json_stanford_screening \
  --report logs/validation_5kp_no_roof.csv
```

## Definition of Done
1. All target JSONs generated.
2. No roof_apex appears in output.
3. No extra label outside the 5 allowed keypoints.
4. Validation report is produced and attached in run log.

## Handoff Notes
- If any file has missing keypoint shape count (< 5), move it to human review queue.
- Record run timestamp, image count processed, and failed files in logs.
