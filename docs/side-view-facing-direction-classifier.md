# Side-View Facing Direction Classifier

**Status:** Training path complete; current model is **not promoted**.

## Objective

Train a side-view classifier that decides whether a true side-view car is:

- `left-looking`
- `right-looking`

This decision must happen before trusting semantic `front_*` / `rear_*`
keypoints for wheel centers, bumpers, overhangs, or overall length.

## Training Data Source

Labels are derived from verified LabelMe JSONs in:

```text
yolo_training/side_view_dataset/labelme_json
```

The script infers direction from semantic point order:

- `right-looking`: `front_bumper.x > rear_bumper.x`
- `left-looking`: `front_bumper.x < rear_bumper.x`

Wheel center order is used as a consistency check. If bumper and wheel direction
disagree, the sample is skipped.

## Script

```text
scripts/train_side_orientation_classifier.py
```

The classifier is a NumPy softmax linear probe over CLIP image embeddings. It
does not require scikit-learn.

## Missing Image Recovery

Command:

```powershell
.\.venv\Scripts\python.exe scripts\recover_side_orientation_images.py
```

Result:

```text
Candidates: 6
Recovered: 6
Existing: 0
Failed: 0
```

Recovered images are written to:

```text
dataset_raw/images/train/labeled_from_candidates
```

## Manifest Check

Command:

```powershell
.\.venv\Scripts\python.exe scripts\train_side_orientation_classifier.py --manifest-only
```

Result:

```text
Samples: 79
Counts: {'left-looking': 27, 'right-looking': 52}
Skipped: 5
Manifest: yolo_training\side_view_orientation_classifier\manifest.csv
```

Skipped rows are currently:

```text
5 missing_or_ambiguous_front_rear_points
```

## Current Training Run

Command:

```powershell
.\.venv\Scripts\python.exe scripts\train_side_orientation_classifier.py --epochs 800 --min-samples-per-class 10
```

Result:

```text
Samples: 79
Counts: {'left-looking': 27, 'right-looking': 52}
Metrics: {'samples': 79.0, 'train_samples': 64.0, 'val_samples': 15.0, 'train_accuracy': 1.0, 'val_accuracy': 0.7333333333333333}
Model: yolo_training\side_view_orientation_classifier\side_orientation_clip_linear.npz
```

## Decision

Do **not** promote this classifier yet.

The high train accuracy and weak validation accuracy indicate overfitting on a
small/noisy direction dataset. Missing-image recovery improved validation from
58.3% to 73.3%, but the model is still not reliable enough to become the
production side-view direction gate.

## Next Gate

Build a stronger direction dataset before promotion:

1. Human-review the 5 ambiguous front/rear rows in the manifest.
2. Add more verified left-looking examples.
3. Add a held-out hard set with mixed crops, lighting, and vehicle types.
4. Promote only when validation accuracy is high enough and failures are
   visually reviewed.
