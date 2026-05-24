````markdown
# Vehicle Proportion Measurement Instruction

## Objective

You are assisting in building or improving a vehicle visual analysis system.

The main objective is to define a systematic method for estimating vehicle proportions from a side-view image by using the **wheel/tire diameter as the main reference unit**.

This method should make measurement easier, more consistent, and easier to reproduce across different vehicle images, even when the original image scale, resolution, or actual vehicle dimensions are unknown.

---

## Core Principle

Use the vehicle wheel diameter as the base measurement unit.

```text
Wheel Diameter = 1 unit
````

All other vehicle dimensions should be estimated as a ratio or multiple of the wheel diameter.

Example:

```text
Wheel Base = ± 4.4x wheel diameter
Front Over Hang = ± 1.25x wheel diameter
Rear Over Hang = ± 1.4x wheel diameter
Overall Height = ± 2.2x wheel diameter
Engine Hood Height = ± 1.5x wheel diameter
```

Do not rely only on absolute pixel size or millimeter values at the beginning.
First, calculate visual ratios against the wheel diameter.

---

## Why Wheel Diameter Is Used

The wheel diameter is used as the primary reference because:

1. The wheel is usually clearly visible in a side-view vehicle image.
2. The wheel has a relatively consistent circular shape.
3. The wheel position is strongly connected to the vehicle body structure.
4. It can be used as a visual scale even when the real-world dimension is unknown.
5. It allows different vehicle images to be compared using the same proportional logic.

---

## Required Measurement References

The system or implementation must identify the following reference points:

### 1. Front Wheel Center

The center point of the front wheel circle.

### 2. Rear Wheel Center

The center point of the rear wheel circle.

### 3. Wheel Diameter

The vertical or horizontal diameter of the visible wheel/tire.

Use the clearest visible wheel as the primary reference.

### 4. Ground Line

A horizontal line passing through the tire contact points with the ground.

### 5. Front Body End

The most forward visible point of the vehicle body, usually the front bumper.

### 6. Rear Body End

The most rearward visible point of the vehicle body, usually the rear bumper.

### 7. Highest Vehicle Point

The highest visible point of the vehicle roof.

Do not include antenna, roof accessories, or visual noise unless explicitly required.

### 8. Engine Hood Line

The upper visual line of the engine hood, usually from the windshield area toward the front body.

### 9. Body Waist / Window Line

The visual separation line between the lower body and the side window area.

---

## Measurement Method

Follow this process step by step.

### Step 1: Detect or Select the Wheel

Identify the wheel that is most clearly visible and least distorted.

Preferred criteria:

* Full circular wheel is visible.
* Wheel is not blocked by shadow, fender, or object.
* Wheel is not heavily distorted by perspective.
* Tire contact point with ground is visible.

Set this wheel diameter as:

```text
Wheel Diameter = 1 unit
```

---

### Step 2: Define the Ground Line

Create a horizontal line through the bottom contact points of the front and rear wheels.

This line becomes the base reference for all vertical measurements.

```text
Ground Line = baseline for vertical measurement
```

---

### Step 3: Define Wheel Centers

Find the center point of the front and rear wheels.

The center points are required for:

* Wheel Base
* Front Over Hang
* Rear Over Hang

---

### Step 4: Measure Wheel Base

Wheel Base is the horizontal distance between the rear wheel center and the front wheel center.

```text
Wheel Base = distance(rear_wheel_center, front_wheel_center)
Wheel Base Ratio = Wheel Base / Wheel Diameter
```

Expected approximate ratio:

```text
Wheel Base ≈ 4.4x wheel diameter
```

---

### Step 5: Measure Front Over Hang

Front Over Hang is the horizontal distance from the front wheel center to the most forward body point.

```text
Front Over Hang = distance(front_wheel_center, front_body_end)
Front Over Hang Ratio = Front Over Hang / Wheel Diameter
```

Expected approximate ratio:

```text
Front Over Hang ≈ 1.25x wheel diameter
```

---

### Step 6: Measure Rear Over Hang

Rear Over Hang is the horizontal distance from the rear wheel center to the most rearward body point.

```text
Rear Over Hang = distance(rear_wheel_center, rear_body_end)
Rear Over Hang Ratio = Rear Over Hang / Wheel Diameter
```

Expected approximate ratio:

```text
Rear Over Hang ≈ 1.4x wheel diameter
```

---

### Step 7: Measure Overall Height

Overall Height is the vertical distance from the ground line to the highest vehicle roof point.

```text
Overall Height = distance(ground_line, highest_vehicle_point)
Overall Height Ratio = Overall Height / Wheel Diameter
```

Expected approximate ratio:

```text
Overall Height ≈ 2.2x wheel diameter
```

---

### Step 8: Measure Engine Hood Height

Engine Hood Height is the vertical distance from the ground line to the upper engine hood line.

```text
Engine Hood Height = distance(ground_line, engine_hood_line)
Engine Hood Height Ratio = Engine Hood Height / Wheel Diameter
```

Expected approximate ratio:

```text
Engine Hood Height ≈ 1.5x wheel diameter
```

---

### Step 9: Identify Body Waist / Window Line

Identify the visual line separating the lower body and side window area.

This line is useful for reading the vehicle’s body proportion and visual character.

```text
Body Waist / Window Line = lower boundary of side window area
```

This measurement may not always have a fixed ratio but should be stored as a proportional reference.

---

## Formula

Use this formula when converting ratios into actual estimated dimensions:

```text
Actual Size = Ratio × Actual Wheel Diameter
```

Example:

```text
Actual Wheel Diameter = 650 mm
Wheel Base Ratio = 4.4

Wheel Base = 4.4 × 650 mm
Wheel Base = 2,860 mm
```

Additional examples:

```text
Front Over Hang = 1.25 × 650 mm = 812.5 mm
Rear Over Hang = 1.4 × 650 mm = 910 mm
Overall Height = 2.2 × 650 mm = 1,430 mm
Engine Hood Height = 1.5 × 650 mm = 975 mm
```

---

## Expected Output Format

The implementation should produce structured output similar to this:

| Measurement Name         | Definition                                  | Start Point              | End Point                 | Ratio to Wheel Diameter | Validation Note                       |
| ------------------------ | ------------------------------------------- | ------------------------ | ------------------------- | ----------------------- | ------------------------------------- |
| Wheel Diameter           | Tire/wheel diameter as base unit            | Lowest wheel point       | Highest wheel point       | 1.0x                    | Use the clearest visible wheel        |
| Ground Line              | Vehicle baseline                            | Rear tire ground contact | Front tire ground contact | Reference               | Must align with tire contact points   |
| Wheel Base               | Distance between wheel centers              | Rear wheel center        | Front wheel center        | ± 4.4x                  | Ensure wheel centers are consistent   |
| Front Over Hang          | Distance from front wheel to front body end | Front wheel center       | Front bumper end          | ± 1.25x                 | Valid only if front bumper is visible |
| Rear Over Hang           | Distance from rear wheel to rear body end   | Rear wheel center        | Rear bumper end           | ± 1.4x                  | Valid only if rear bumper is visible  |
| Overall Height           | Total vehicle height                        | Ground line              | Highest roof point        | ± 2.2x                  | Exclude antenna or roof accessories   |
| Engine Hood Height       | Height of engine hood                       | Ground line              | Upper hood line           | ± 1.5x                  | Follow actual hood line, not shadow   |
| Body Waist / Window Line | Body-to-window separation                   | Lower side window area   | Lower side window area    | Visual ratio            | Used for body proportion analysis     |

---

## Implementation Rules

When writing or modifying code:

1. Do not create a new measurement logic if an existing one already exists.
2. First inspect the current project structure and reuse existing modules, functions, or classes.
3. Keep the implementation simple and explainable.
4. Prioritize measurement consistency over visual decoration.
5. Store all calculated dimensions as ratios against wheel diameter.
6. Absolute dimensions in millimeters should only be calculated if actual wheel diameter is provided.
7. Make every reference point explicit and traceable.
8. Avoid hardcoding values unless they are used as configurable default estimates.
9. Add validation notes when the image condition may affect accuracy.
10. Keep output readable for both technical and non-technical users.

---

## Accuracy Requirements

The system must consider the following limitations:

* Image may not be perfectly side-view.
* Vehicle may be captured with perspective distortion.
* Wheels may be partially hidden.
* Image may be low-resolution or blurry.
* Bumper edges may not be clearly visible.
* Roof line may be affected by reflection or accessories.
* Ground line may not be horizontal due to camera angle.
* Front and rear wheels may appear different in size due to perspective.

If one or more of these issues is detected, include a validation warning.

Example:

```json
{
  "measurement": "wheel_base",
  "ratio": 4.4,
  "confidence": "medium",
  "warning": "Image is not perfectly side-view; result should be treated as visual estimation."
}
```

---

## Suggested Data Structure

Use a clear and extensible data structure.

Example:

```json
{
  "reference_unit": {
    "name": "wheel_diameter",
    "value": 1.0,
    "unit": "ratio"
  },
  "measurements": {
    "wheel_base": {
      "ratio_to_wheel_diameter": 4.4,
      "start_point": "rear_wheel_center",
      "end_point": "front_wheel_center",
      "confidence": "high"
    },
    "front_over_hang": {
      "ratio_to_wheel_diameter": 1.25,
      "start_point": "front_wheel_center",
      "end_point": "front_body_end",
      "confidence": "medium"
    },
    "rear_over_hang": {
      "ratio_to_wheel_diameter": 1.4,
      "start_point": "rear_wheel_center",
      "end_point": "rear_body_end",
      "confidence": "medium"
    },
    "overall_height": {
      "ratio_to_wheel_diameter": 2.2,
      "start_point": "ground_line",
      "end_point": "highest_vehicle_point",
      "confidence": "medium"
    },
    "engine_hood_height": {
      "ratio_to_wheel_diameter": 1.5,
      "start_point": "ground_line",
      "end_point": "engine_hood_line",
      "confidence": "medium"
    }
  }
}
```

---

## Validation Logic

Before accepting a measurement result, validate:

1. Is the wheel diameter visible?
2. Are both wheel centers detected or manually defined?
3. Is the ground line clear?
4. Are front and rear body ends visible?
5. Is the roof line visible?
6. Is the image close to a true side-view?
7. Are the calculated ratios within a reasonable range?

If validation fails, return a warning instead of silently producing a misleading result.

---

## Expected Behavior for Codex / Copilot

When assisting with this project:

1. Always read the existing code before making changes.
2. Do not invent a new architecture unless explicitly requested.
3. Use the current file and folder structure.
4. Add only the minimum necessary changes.
5. Keep naming simple and understandable.
6. Prefer small functions with clear responsibility.
7. Add comments only where they clarify measurement logic.
8. Do not over-engineer the solution.
9. Keep the output easy to inspect and debug.
10. Confirm the implementation step is complete before moving to the next task.

---

## Important Instruction

The most important idea is:

```text
Use wheel diameter as the base unit first.
Calculate all vehicle proportions as ratios against the wheel diameter.
Only convert to actual size if actual wheel diameter is known.
```

This makes the measurement process faster, more consistent, and easier to compare across different vehicle images.

```
```
