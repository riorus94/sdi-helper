# Geometry Filtering Guide for Vehicle Side-View Images

## Purpose

This document defines the geometry rules used to filter scraped vehicle images before sending them to Roboflow for annotation.

The goal is to keep only images that are suitable for vehicle proportion measurement, especially:

- Wheelbase
- Front overhang
- Rear overhang
- Overall height
- Engine hood height
- Wheel diameter
- Overall length

The main principle is:

> False reject is acceptable. False accept is dangerous.

Bad 3/4-view images should not enter the annotation dataset because they can corrupt measurement accuracy.

---

# 1. Accepted Image Criteria

An image can be auto-accepted only if it is a clean side-view image.

## Required conditions

The vehicle must have:

- Full vehicle body visible
- Two visible wheels
- Front and rear bumper visible
- Roof line visible
- Hood line visible
- Ground or tire contact baseline visible
- Minimal perspective distortion
- View angle close to 90-degree side profile

---

# 2. Rejected Image Criteria

Reject the image if any of these conditions exist:

- Front 3/4 view
- Rear 3/4 view
- Front view
- Rear view
- Vehicle is cropped
- Only one wheel is visible
- Front or rear bumper is missing
- Image is too blurry or low resolution
- Strong perspective distortion
- Vehicle is partially blocked
- Wheel detection is invalid

---

# 3. Main Geometry Signals

## 3.1 Wheel Visibility

A valid side-view vehicle should show two clear wheels.

Required:

```text
number_of_visible_wheels >= 2