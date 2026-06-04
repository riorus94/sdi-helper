# Sedan Proportions — context (Agent 1)

Purpose
- This context file lives alongside Agent 1 labeling rules and priors. It provides the authoritative modular-proportion grid and keypoint anchors used by Agent 1 when creating or verifying 19KP / 7KP labels for sedan side-views.

Units and intent
- Base unit: Wheel diameter (D). All linear measures expressed as multiples of `D`.
- Orientation: front of car is on the LEFT, rear on the RIGHT. Do not mirror.
- Construction grid: repeating circles of diameter = 1 D arranged along the baseline; wheel centers are primary anchors.

Primary proportions (multiples of D)
- `I (wheel_diameter)`: 1.00 D
- `A (wheelbase)`: 4.40 D
- `B (front overhang)`: 1.25 D
- `C (rear overhang)`: 1.40 D
- `J (overall length)`: 7.05 D (A + B + C)

Vertical proportions (multiples of D)
- `D` (overall height): 2.20 D
- `E` (hood height): 1.50 D
- `H` (beltline / waistline): 1.35 D

Keypoints (coordinates in D units; baseline ground Y = 0)
- `front_bumper`: (0.00, 0.00)
- `front_wheel_center`: (1.25, 0.50)
- `front_wheel_ground`: (1.25, 0.00)
- `rear_wheel_center`: (5.65, 0.50)
- `rear_wheel_ground`: (5.65, 0.00)
- `rear_bumper`: (7.05, 0.00)
- `hood_edge`: (2.15, 1.50)
- `roof_apex`: (3.89, 2.20)
- `waist_front`: (2.70, 1.35)
- `waist_rear`: (4.80, 1.35)

Grid and construction rules
- Use circles of diameter `D` placed with centers at 0.5D, 1.5D, 2.5D, ... along the baseline to form a modular proportional grid.
- Use wheel center positions as anchor references for measuring `A`, and for aligning hood/cabin/trunk zones.
- Semantic zones: hood = `front_bumper` → `front_wheel_center`; cabin ≈ between wheel centers; trunk = `rear_wheel_center` → `rear_bumper`.

Reference assets (for Agent 1 visuals)
- Technical SVG (visual, D = 100 px used for display): ../../designs/sedan_proportional_D.svg

Notes for Agent 1
- When a visible landmark is clear, label the actual landmark — these ratios are priors for estimating missing or occluded points.
- Mark any ratio-derived points as `requires_manual_review` when they substitute for visually ambiguous landmarks.

Linear measurements using the circle grid
- The repeating D-diameter circles form a modular horizontal grid: each circle = 1 D. Use the circle centers and circumferences to define, measure and visually check proportions.
- Wheelbase (`A`) spans multiple circle units: measure from the front wheel center to the rear wheel center and count the intervening D-circles (A = 4.40 D → ~4 full circles + fractional remainder).
- Front and rear overhangs (`B`, `C`) extend beyond the wheel centers into additional circle units; these are measured from wheel center to bumper and typically equal 1–1.5 D as priors.
- Total vehicle length (`J`) includes all circle units from front bumper through rear bumper (J = A + B + C → 7.05 D in this spec).

Layered horizontal arrows (visual conventions for annotations)
- Short arrow — wheelbase (A): anchored at the two wheel centers; primary horizontal span used for ratio calculations.
- Medium arrow — body length (A + small overhangs or cabin extents): a body-centric measure useful for cabin massing checks.
- Long arrow — full length (J): spans front bumper → rear bumper; used for packaging and report displays.

Annotation rules
- Draw arrows horizontally along the baseline at distinct vertical offsets (short above, medium slightly below, long further below) so they do not overlap visually.
- Label arrows in `D` units (e.g., `A = 4.40 D`, `J = 7.05 D`) and include a mm conversion note only when a real wheel diameter in mm is supplied.
- Use wheel center anchors to snap arrow endpoints to the nearest circle center when estimating fractional D values.

Visual and export guidance
- When exporting diagram assets for Agent 1 review, include the circle grid layer as a togglable overlay so reviewers can verify proportional alignment quickly.
- Ensure orientation metadata `facing: left` is present in exported JSON or filenames to avoid accidental mirroring during downstream processing.

Vertical measurement lines (baseline → point)
- Use vertical measurement lines (arrows) that start at the baseline (ground Y = 0) and extend upward to define the following heights in D units:
	- `E` (hood height): 1.50 D — draw a vertical arrow from baseline to `hood_edge`.
	- `H` (beltline / waistline): 1.35 D — draw a vertical arrow from baseline to the beltline location.
	- `D` (overall / roof apex): 2.20 D — draw a vertical arrow from baseline to `roof_apex`.
- Visual rules:
	- Place vertical arrows at distinct horizontal offsets (e.g., hood arrow near the front third, beltline arrow near mid-cabin, roof arrow toward roof apex) so they do not overlap the silhouette or each other.
	- Use consistent stroke styles and colors for vertical measurements and label them with the `D` value (for example: `E = 1.50 D`).
	- When rendering to pixels for visual proofs use a nominal `D` scale (e.g., `D = 100 px`) to position arrows and sample labels; store source values in `D` units in exported JSON.
- Export guidance:
	- Include the vertical-measurement layer as a togglable overlay in review assets.
	- When supplying mm conversions, compute: `value_mm = value_in_D * reference_wheel_diameter_mm` and display both `D` and `mm` values if a reference wheel diameter is provided.

