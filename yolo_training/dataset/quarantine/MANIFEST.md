# Dataset Quarantine Manifest

Quarantined on: 2026-04-20
Reason: Dataset hygiene audit (Step 2 of wheelbase hardening pipeline)

All files are **moved, not deleted**. To restore any file, move it back from
`quarantine/<split>/` to the corresponding `<split>/` directory.

## Quarantined Files

### Train — Missing Label
| File | Reason |
|------|--------|
| `000018.jpeg` (image only) | No matching label file; YOLO treats as negative sample (no wheels), teaching the model that a side-view image has zero wheels. Root cause: `.jpeg` extension not covered by `check_labels.py` QA glob. |

### Train — Stem Collision
| File | Reason |
|------|--------|
| `000014.jpg` (image only) | Stem collision with `000014.jpeg`. Two different images (30 KB vs 2.2 MB) shared one label file `000014.txt`. The 30 KB file is a thumbnail/different image; the 2.2 MB `.jpeg` is retained with the label. |

### Train — Duplicate Annotations
| File | Paired With | Reason |
|------|-------------|--------|
| `000009_suv_side_view_press_release.{jpg,txt}` | `000008_suv_side_view` | Identical bbox coordinates on different images — copy-pasted annotation. |
| `000012.{jpg,txt}` | `000009` | Identical bbox coordinates on different images — copy-pasted annotation. |

### Train — Extreme Bbox Outliers
| File | Size Ratio | Min Area | Reason |
|------|-----------|----------|--------|
| `side_profile_car_review__000007.{jpg,txt}` | 12.81× | 0.0020 | Tiny bbox + extreme asymmetry — one wheel at frame edge. |
| `sedan_side_view_brochure__000018.{jpg,txt}` | 8.39× | 0.0034 | Tiny bbox + extreme asymmetry — partial occlusion. |
| `suv_side_view__000009.{jpg,txt}` | 3.92× | OK | Size asymmetry beyond 2.0× threshold. |
| `side_view_car_review__000017.{jpg,txt}` | 3.50× | OK | Size asymmetry beyond 2.0× threshold. |
| `car_side_view__000019.{jpg,txt}` | 2.41× | OK | Size asymmetry beyond 2.0× threshold. |

### Val — Non-Side-View
| File | Reason |
|------|--------|
| `000045.{jpg,txt}` | Δcx = 0.095 (threshold: 0.15). 3/4 or front-angled view — violates side-view geometry assumption. |

### Val — Extreme Bbox Outlier
| File | Size Ratio | Min Area | Reason |
|------|-----------|----------|--------|
| `suv_side_view_street__000017.{jpg,txt}` | 8.78× | 0.0030 | Tiny bbox + extreme asymmetry — one wheel at frame edge. |

### Test — Invalid Split (Labels Copied from Val)
| File | Copied From | Reason |
|------|-------------|--------|
| `test_000001.{jpg,txt}` | `val/000001` | Label content identical to val counterpart; images differ. Test set provides zero independent evaluation signal. |
| `test_000001_suv_side_view.{jpg,txt}` | `val/000001_suv_side_view` | Same as above. |

## Impact Summary

| Split | Before | After | Removed |
|-------|--------|-------|---------|
| Train images | 85 | 77 | 8 |
| Train labels | 83 | 75 | 8 |
| Val images | 21 | 19 | 2 |
| Val labels | 21 | 19 | 2 |
| Test images | 2 (+.gitkeep) | 0 (+.gitkeep) | 2 |
| Test labels | 2 | 0 | 2 |
