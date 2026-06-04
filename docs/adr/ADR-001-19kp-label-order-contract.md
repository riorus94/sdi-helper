# ADR-001 — 19KP Label Order Contract

**Status:** Accepted  
**Date:** 2026-05-26  
**Repos affected:** `sdi-helper`, `vehicle-sdi-system`

---

## Context

The YOLOv8-pose model outputs keypoints as a fixed-length ordered array. Every consumer
of that array must agree on which index maps to which anatomical landmark. There are
currently two places that encode this mapping:

1. `sdi-helper/yolo_training/labelme_to_yolo_pose.py` — `DEFAULT_KP_ORDER` list, used
   during LabelMe → YOLO conversion to assign indices.
2. `vehicle-sdi-system/cv_service/yolo_cv_client.py` — `_POSE_*_LABELS` tuple, used
   at inference time to name each output index.

If these two lists disagree, the backend silently maps the wrong pixel coordinates to
the wrong anatomical names, producing corrupted geometry dimensions with no runtime
error.

## Decision

**`DEFAULT_KP_ORDER` in `sdi-helper/yolo_training/labelme_to_yolo_pose.py` is the
single source of truth (SOT) for the 19KP label order.**

The canonical 19KP order (indices 0–18) is:

| idx | label                 |
|-----|-----------------------|
|  0  | roof_apex             |
|  1  | side_window_top_front |
|  2  | side_window_top_rear  |
|  3  | front_bumper          |
|  4  | rear_bumper           |
|  5  | front_wheel_center    |
|  6  | front_wheel_ground    |
|  7  | rear_wheel_center     |
|  8  | rear_wheel_ground     |
|  9  | fender_arch_front     |
| 10  | fender_arch_rear      |
| 11  | hood_edge             |
| 12  | body_waist_front      |
| 13  | body_waist_rear       |
| 14  | panel_front           |
| 15  | panel_rear            |
| 16  | windshield_base       |
| 17  | rear_glass_base       |
| 18  | ground_ref            |

### Rules

1. `_POSE_19KP_LABELS` in `vehicle-sdi-system/cv_service/yolo_cv_client.py` **must**
   be a verbatim copy of `DEFAULT_KP_ORDER`, in the same order, with the same strings.
2. `dataset_pose.yaml` (both the live config and `_colab_staging/dataset_pose.yaml`)
   must reflect `kpt_shape: [19, 3]` derived from this list.
3. Any new consumer of 19KP model output **must** import or copy from `DEFAULT_KP_ORDER`
   rather than redefining its own list.
4. If `DEFAULT_KP_ORDER` ever changes (label added, removed, or reordered), a new model
   must be trained from scratch — this is not a hot-swap operation.

## Consequences

- A cross-repo integration test that asserts the two label lists are identical is
  required before the first 19KP model is promoted to production. (See B3 tasks.)
- The 5 labels present in training data but unused in geometry computations
  (`body_waist_rear`, `rear_glass_base`, `side_window_top_front`,
  `side_window_top_rear`, `windshield_base`) still appear in the keypoints dict
  returned by `yolo_cv_client.py`; `analysis.py` simply ignores them.
- `_colab_staging/dataset_pose.yaml` is a **derived artefact** of `DEFAULT_KP_ORDER`
  and must be regenerated (not hand-edited) when the label list changes.
