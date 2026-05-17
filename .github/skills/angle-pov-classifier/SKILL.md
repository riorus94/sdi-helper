---
name: angle-pov-classifier
description: "Add, tune, or debug the CLIP-based camera-angle (POV) filter in the sdi-helper scraping pipeline. Use when: adding an angle/pov classifier; tuning 3/4-shot rejection prompts; wiring a new CLIP filter into process_candidate_image; reviewing rejected_angle outcomes; changing min_straight_margin threshold; debugging why diagonal or 3/4 shots are accepted or rejected."
argument-hint: "front | side | rear | all"
---

# Angle / POV Classifier Skill

Adds and maintains a CLIP-based gate that rejects 3/4-angle, diagonal, overhead, and close-up-detail car photos, allowing only straight-on shots to pass into the dataset.

## Mental Model — What "Straight-On" Means

| View | Photographer position |
|------|-----------------------|
| FRONT | Stand directly in front of the grille, camera at bumper height, step back until the full front face fills ~70-80% of the frame. Both headlamps must be roughly symmetric left/right. |
| SIDE | Stand at the midpoint between the front and rear axle, camera at door-handle height, step back until both wheels are fully in frame. No front or rear face should dominate. |
| REAR | Same as front but from behind. Boot lid centred, both tail lights equidistant from centre. |

Anything else (3/4 angle, diagonal, aerial, detail close-up) must be rejected.

## File Layout

```
sdi_helper/
  application/
    ports/
      angle_filter.py              ← Protocol / port
    dto/
      process_result.py            ← REJECTED_ANGLE outcome
    use_cases/
      process_candidate_image.py   ← step 9.5 (after clean filter, before view classifier)
  infrastructure/
    models/
      clip_angle_filter.py         ← CLIP implementation
  interfaces/
    cli/
      run_scrape.py                ← composition root wiring
```

## Pipeline Position

Gate order in `process_candidate_image.py`:
```
1.  download
2.  decode
3.  size/aspect heuristic
4.  YOLO car detection
5.  car-area ratio
5.5 truncation check
6.  face detector
7.  real-photo filter
8.  interior filter
9.  clean-photo filter
9.5 ← angle filter  (REJECTED_ANGLE)
10. view classifier
11. quota check
12. dedup
13. ACCEPTED
```

## Procedure — Adding or Modifying

### 1. Tune straight-on prompts
Edit `_STRAIGHT_ON_PROMPTS` in `clip_angle_filter.py`.
Good prompts are **symmetric** descriptions — mention that both headlamps / wheels are equidistant or centred.

### 2. Tune angled prompts
Edit `_ANGLED_PROMPTS`. Cover all four corners (FL, FR, RL, RR) explicitly. Studio/promo diagonal shots are common false negatives — add a prompt for those.

### 3. Adjust the margin threshold
```python
ClipAngleFilter(min_straight_margin=0.04)  # default
```
- Raise (e.g. `0.08`) → stricter, more 3/4 shots rejected, risk losing borderline good shots
- Lower (e.g. `0.02`) → looser, more 3/4 shots pass

### 4. Test against a known bad image
```python
import cv2
from sdi_helper.infrastructure.models.clip_angle_filter import ClipAngleFilter

img = cv2.imread("path/to/bad_image.jpg")
f = ClipAngleFilter()
print(f.is_straight_on(img))  # should be False for 3/4 shots
```

### 5. Run the scraper with --verbose to see REJECTED_ANGLE counts
```powershell
cd d:\project\sdi-helper
python -m sdi_helper.interfaces.cli.run_scrape --query-contains "front view" --max-results 50 --verbose
```
Look for `rejected_angle` lines in stderr.

## Prompt Engineering Rules for CLIP Angle Prompts

1. **Be geometrically explicit** — say "45-degree diagonal", "exactly 90 degrees", "both headlamps equal distance from center"
2. **Name what's visible** — "front grille AND left door panel simultaneously" pins the 3/4 geometry
3. **Cover all four diagonal corners** — FL, FR, RL, RR get separate prompts
4. **Include studio/promo angles** — promotional photos are almost always 3/4; add an explicit prompt
5. **Keep straight-on prompts symmetry-focused** — asymmetry is the hallmark of 3/4 shots

## Common Bad Image Types & Matching Prompts

| Bad type | Rejection prompt keyword |
|----------|--------------------------|
| Front-left 3/4 | "front-left corner … front grille and left door" |
| Front-right 3/4 | "front-right corner … front and right side simultaneously" |
| Rear-left 3/4 | "rear-left corner … tail lights and left door panels" |
| Rear-right 3/4 | "rear-right corner … boot and right side panels" |
| Studio promo diagonal | "stylised diagonal angle for advertising" |
| Overhead / bird's-eye | handled by `clip_clean_photo_filter.py` noisy prompts |
| Detail close-up (headlamp) | handled by `clip_clean_photo_filter.py` noisy prompts |

## Adding a New Corner Pattern
1. Identify which corner (FL/FR/RL/RR) and what's co-visible in the frame.
2. Add one prompt to `_ANGLED_PROMPTS` naming the corner and the two visible faces.
3. Re-run a test batch (`--max-results 20`) and check REJECTED_ANGLE rate.
4. If good shots start getting rejected, lower `min_straight_margin` slightly or soften the prompt wording.

## Related Files
- `clip_clean_photo_filter.py` — handles detail close-ups and overhead shots (noisy prompts)
- `clip_view_classifier.py` — runs AFTER this filter; classifies FRONT/SIDE/REAR
- `clip_real_photo_filter.py` — runs BEFORE this filter; rejects cartoons/illustrations
