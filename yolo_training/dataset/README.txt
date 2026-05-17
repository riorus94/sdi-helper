# YOLO training dataset — side-view wheel detection (Phase 1)
# ============================================================
#
# Task:    DETECT  (bounding box, NOT pose / keypoints)
# Scope:   side-view vehicle images only
# Class:   wheel (class_id = 0)
#
# ====================
# FOLDER STRUCTURE
# ====================
#
#   dataset/
#     images/
#       train/   <- side-view .jpg/.png  (80 % of total dataset)
#       val/     <- side-view .jpg/.png  (20 % of total dataset)
#     labels/
#       train/   <- one .txt per image in images/train/
#       val/     <- one .txt per image in images/val/
#
# ====================
# ANNOTATION FORMAT
# ====================
#
# YOLO detection format — one line per wheel, all values normalised [0, 1]:
#
#   <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
#
#   class_id  = 0          (always 0 — only class is 'wheel')
#   cx_norm   = bbox centre x / image width
#   cy_norm   = bbox centre y / image height
#   w_norm    = bbox width  / image width
#   h_norm    = bbox height / image height
#
# Example (two wheels visible in a 1080×1080 image):
#
#   0 0.750 0.713 0.130 0.190    <- front wheel
#   0 0.218 0.713 0.130 0.190    <- rear wheel
#
# Bbox guidelines:
#   - Draw a TIGHT bbox around the VISIBLE tyre (rubber, not the arch)
#   - Include the full wheel diameter if not occluded
#   - Do NOT include the wheel arch or bodywork in the bbox
#   - Label wheels that are at least 50% visible
#   - Do NOT label wheels behind other vehicles / objects
#
# Wheel centre (derived at inference — no extra annotation needed):
#   cx_px = cx_norm * image_width
#   cy_px = cy_norm * image_height
#
# ====================
# ACCEPTANCE CRITERIA (Phase 1)
# ====================
#
#   val mAP50      >= 0.70
#   val Recall     >= 0.80  (low false negatives is the priority)
#   2 wheels found in >= 90% of clean side-view images
#
# ====================
# COLLECTION GUIDE
# ====================
#
# Minimum before first training run : 50 images (train + val)
# Recommended for reliable Phase 1  : 150–200 images
#
# Image requirements:
#   - Side view only (vehicle roughly perpendicular to camera)
#   - Both front and rear wheels must be at least partially visible
#   - Mix of vehicle types: sedans, SUVs, hatchbacks, trucks
#   - Mix of lighting: daylight, overcast, golden hour
#   - Avoid: heavily occluded wheels, extreme tilt > 15°, night shots
#
# Free sources:
#   - CompCars dataset (side-view split)
#   - Stanford Cars (many side-view images)
#   - BDD100K (extract side-view frames)
#   - Personal photos of parked vehicles
#
# Recommended annotation tool:
#   - Roboflow (free tier, exports YOLO format directly)
#   - CVAT (self-hosted, free)
#   - LabelImg (lightweight, local)
#
# ====================
# DATASET SPLIT RULE
# ====================
#
#   Train : Val = 80 : 20
#   Do NOT create a test split until Phase 1 target is met.
#   Separate test images can be added later for final evaluation.

