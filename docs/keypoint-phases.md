# Keypoint Phases for Agent 1

This document groups side-view vehicle keypoints into practical training/annotation phases.

## Phase 1 (Priority)

These are the highest-impact landmarks for SDI geometry checks and measurements:

- front_wheel_center
- rear_wheel_center
- front_wheel_ground
- rear_wheel_ground
- front_bumper
- rear_bumper
- roof_apex
- ground_ref

Use Phase 1 as the first milestone for model stability and manual review focus.

## Phase 2

These improve body profile structure once Phase 1 is stable:

- hood_edge
- windshield_base
- rear_glass_base
- side_window_top_front
- side_window_top_rear

## Phase 3

These are detail/refinement landmarks and should be optimized after Phases 1-2:

- fender_arch_front
- fender_arch_rear
- body_waist_front
- body_waist_rear
- panel_front
- panel_rear

## Agent 1 Configuration

Agent 1 reads keypoint phase priorities from:

- config/agent1_keypoint_priority.json

Current policy is to prioritize Phase 1 by:

1. Listing Phase 1 points first in generated LabelMe JSON output.
2. Increasing review urgency when Phase 1 confidence is low.
