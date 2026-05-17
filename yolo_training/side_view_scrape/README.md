# Side-View Validity — Dataset Pipeline

## Pipeline Order

```
collect.py
    ↓
quality_filter.py   ← deterministic quality gate
    ↓
Roboflow upload     ← images/quality_pass/ only
    ↓
Human review & final labeling
    ↓
YOLO classification training
```

---

## Important Distinction

| Concept | Keep? |
|---|---|
| **Invalid view** (wrong angle, 3/4 view, rear quarter…) | ✅ KEEP — required training class |
| **Bad quality** (blurry, corrupt, tiny thumbnail…) | ❌ DISCARD — unusable for training |

Invalid view ≠ bad image quality.

> **Strict angle rule:** `side_view_valid` means the vehicle body axis is **exactly
> horizontal** in the frame — a pure 90-degree lateral shot. Any 3/4 angle, front-
> quarter angle, or rear-quarter angle is `side_view_invalid`, even if the image
> looks clean and detailed. Human reviewers must enforce this during Roboflow labeling.

---

## Quality Filter Rules

`quality_filter.py` applies **6 deterministic checks** in order.
An image must pass **all** checks to proceed.

| # | Check | Rule | Reject Reason |
|---|---|---|---|
| 1 | File integrity | Fully readable by PIL | `corrupt_file` |
| 2 | Resolution | Long edge ≥ **640 px** | `low_resolution` |
| 3 | Aspect ratio | long / short ≤ **3.5** | `extreme_aspect_ratio` |
| 4 | Blur | Laplacian variance ≥ **50** | `excessive_blur` |
| 5 | Color diversity | ≥ **200** unique colors in 64×64 thumbnail | `low_color_diversity` |
| 6 | Brightness | Mean pixel value in **[15, 240]** | `too_dark` / `too_bright` |

No ML inference is used. All thresholds are explainable and adjustable at the top of `quality_filter.py`.

---

## Output Directory Structure

```
side_view_scrape/
├── images/
│   ├── raw/                        ← collect.py output (do not modify)
│   │   ├── valid/{query}/
│   │   └── invalid/{query}/
│   ├── quality_pass/               ← upload ONLY these to Roboflow
│   │   ├── valid_candidates/       ← suggested class: side_view_valid
│   │   └── invalid_candidates/     ← suggested class: side_view_invalid
│   └── discarded_bad_quality/      ← rejected images (audit only)
├── logs/
│   └── quality_filter_log.csv      ← per-image audit trail
└── README.md
```

---

## Auto-Label Intent (After Quality Filtering)

Auto-labels are **suggested only** — human review is mandatory.

| Source folder | Suggested label |
|---|---|
| `raw/valid/` (queries: "90 degree side profile", "lateral view"…) | `side_view_valid` |
| `raw/invalid/` (queries: "3/4 view", "angled", "front side"…) | `side_view_invalid` |

---

## Usage

```bash
# Dry run — see what would be discarded without touching files
python yolo_training/side_view_scrape/quality_filter.py --dry-run

# Full run — copy passing images to quality_pass/, rejects to discarded_bad_quality/
python yolo_training/side_view_scrape/quality_filter.py
```

Audit results are written to `logs/quality_filter_log.csv` with columns:
`source_path, intended_class, action, reason, long_edge_px, aspect_ratio, blur_score, color_diversity, mean_brightness`

---

## Upload to Roboflow

After filtering, upload from `quality_pass/` using `side_view_cls_rf.py`:

```bash
python yolo_training/side_view_scrape/side_view_cls_rf.py upload-quality-pass
```

Only images in `quality_pass/` are sent to Roboflow. Do not upload from `raw/` directly.
