# ADR-004 - Progressive Side-View KP Promotion Ladder

**Status:** Accepted  
**Date:** 2026-05-31  
**Repos affected:** `sdi-helper`, `vehicle-sdi-system`, `vehicle-sdi-frontend`

---

## Context

Vehicle SDI has been moving from a promoted 7-keypoint side-view pose path toward a
full 19-keypoint model. ADR-001, ADR-002, and ADR-003 locked the 19KP label order,
derived `flip_idx`, and 19KP holdout strategy so that training, backend parsing,
and promotion evidence could not silently drift.

That contract is still valuable, but the direct 7KP-to-19KP promotion jump makes
the next model gate depend on the weakest landmarks in the full 19KP set. A model
that is already good enough for height, hood, fender, waistline, or panel geometry
could still fail promotion because later glass/window landmarks are immature.

The product needs reliable geometry improvements earlier than a full 19KP model
may become promotion-ready. Smaller promotion rungs reduce training risk while
preserving strict schema contracts and the existing human review gate.

## Decision

Vehicle SDI will promote side-view pose models through a progressive keypoint
ladder instead of treating the next production migration as a single 7KP-to-19KP
jump.

Approved ladder:

| Rung | Landmark set |
|------|--------------|
| 7KP  | Current baseline: wheels, wheel-ground contacts, ground reference, front bumper, rear bumper |
| 9KP  | 7KP + `roof_apex`, `hood_edge` |
| 11KP | 9KP + `fender_arch_front`, `fender_arch_rear` |
| 13KP | 11KP + `body_waist_front`, `body_waist_rear` |
| 15KP | 13KP + `panel_front`, `panel_rear` |
| 17KP | 15KP + `windshield_base`, `rear_glass_base` |
| 19KP | 17KP + `side_window_top_front`, `side_window_top_rear` |

Each rung is a distinct schema and a distinct retraining/promotion event. A model
trained for one rung must not be treated as compatible with another rung unless a
promotion artifact explicitly records that compatibility.

## Rules

1. A side-view keypoint contract module must define the approved rung schemas,
   ordered labels, symmetric pairs, derived `flip_idx`, `kpt_shape`, and geometry
   capability metadata.
2. Dataset pose config and Colab staging config must be generated from the selected
   rung schema, not hand-maintained independently.
3. 19KP accepted LabelMe annotations may be projected into smaller rung datasets by
   selecting the target rung labels in canonical order.
4. Holdout evaluation must be schema-aware: it checks every required keypoint for
   the active rung and ignores future-rung landmarks.
5. Promotion gate artifacts must record the target rung, model path, manifest,
   prediction summary, confidence threshold, and pass/fail decision.
6. Backend pose parsing must select a declared side-view schema intentionally.
   Tensor keypoint count may be validation input, but model metadata or deployment
   config should be the authority when available.
7. Geometry and frontend overlays must consume reviewed backend-backed capabilities,
   not raw KP-count assumptions. Dimensions remain unavailable until their required
   reviewed landmarks exist.
8. Human review approval remains the only path from raw CV output to geometry and
   SDI calculations.

## Relationship to Existing ADRs

- ADR-001 remains accepted for the final 19KP canonical label order. This ADR
  supersedes only the delivery assumption that the next production jump must be
  directly from 7KP to full 19KP.
- ADR-002 remains accepted in principle: `flip_idx` must be derived mechanically.
  The derivation now applies per rung, not only to the 19KP schema.
- ADR-003 remains accepted for full 19KP promotion. The holdout strategy now also
  applies per rung: carve the evaluation set from accepted canonical annotations,
  evaluate only active-rung landmarks, and record the rung in promotion evidence.

## Consequences

- The first implementation task is a side-view keypoint contract module that gives
  callers one place to obtain rung schema facts.
- Direct 19KP promotion issues that depend on a full 19KP model should be paused or
  reinterpreted until the progressive ladder issues supersede them.
- The 9KP rung becomes the first likely incremental promotion target because it
  unlocks roof and hood geometry while preserving the current wheel/bumper baseline.
- Each rung adds schema-management overhead, so generated config and contract tests
  are mandatory.
- Frontend behavior should continue to show unavailable measurements explicitly and
  draw only backend-backed numeric dimension lines.
