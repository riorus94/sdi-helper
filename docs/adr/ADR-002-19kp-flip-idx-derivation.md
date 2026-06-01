# ADR-002 — 19KP flip_idx Must Be Derived from DEFAULT_KP_ORDER

**Status:** Accepted  
**Date:** 2026-05-26  
**Repos affected:** `sdi-helper`

---

## Context

YOLOv8-pose applies horizontal-flip augmentation during training. For pose models,
`flip_idx` tells the trainer which keypoints are symmetric partners (so that after
a horizontal flip, the left-wheel index swaps with the right-wheel index rather than
staying in the original position). An incorrect `flip_idx` silently trains the model
with corrupted augmentation data, degrading per-keypoint accuracy with no warning.

`sdi-helper/yolo_training/side_view_dataset/dataset_pose.yaml` was previously
configured for the 7KP model (`flip_idx: [2, 3, 0, 1, 4, 6, 5]`). That value is
wrong for the 19KP model and must not be carried forward.

As of 2026-05-26 the `_colab_staging/dataset_pose.yaml` still contains the 7KP config.
This is a **pre-training blocker** for B2.

## Decision

**`flip_idx` for the 19KP model is derived mechanically from the symmetric pairs in
`DEFAULT_KP_ORDER` and must be committed to `dataset_pose.yaml` before Colab training
begins.**

### Symmetric pairs (left ↔ right in side-view flip)

| left label            | idx | right label           | idx |
|-----------------------|-----|-----------------------|-----|
| side_window_top_front |  1  | side_window_top_rear  |  2  |
| front_bumper          |  3  | rear_bumper           |  4  |
| front_wheel_center    |  5  | rear_wheel_center     |  7  |
| front_wheel_ground    |  6  | rear_wheel_ground     |  8  |
| fender_arch_front     |  9  | fender_arch_rear      | 10  |
| body_waist_front      | 12  | body_waist_rear       | 13  |
| panel_front           | 14  | panel_rear            | 15  |
| windshield_base       | 16  | rear_glass_base       | 17  |

Unpaired (midline / symmetric): `roof_apex (0)`, `hood_edge (11)`, `ground_ref (18)`.

### Resulting flip_idx value

```python
flip_idx: [0, 2, 1, 4, 3, 7, 8, 5, 6, 10, 9, 11, 13, 12, 15, 14, 17, 16, 18]
```

Verified: for every index `i`, `flip_idx[flip_idx[i]] == i` (self-inverse check).

### Required changes to `dataset_pose.yaml`

```yaml
kpt_shape: [19, 3]
flip_idx: [0, 2, 1, 4, 3, 7, 8, 5, 6, 10, 9, 11, 13, 12, 15, 14, 17, 16, 18]
```

Both the live `side_view_dataset/dataset_pose.yaml` **and** the staged
`_colab_staging/dataset_pose.yaml` must be updated before any training run.

## Consequences

- A unit test that asserts `flip_idx[flip_idx[i]] == i` for all `i` and that the
  length equals `len(DEFAULT_KP_ORDER)` should be added to `tests/` to prevent
  hand-edit regressions.
- If `DEFAULT_KP_ORDER` ever changes, `flip_idx` must be re-derived — it is never
  valid to copy `flip_idx` from a different KP-count model.
- The `_colab_staging/` staging pipeline must regenerate `dataset_pose.yaml` from
  the live config (or a template) rather than storing a separate hand-maintained copy.
