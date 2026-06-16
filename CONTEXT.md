# SDI Helper

Builds side-view vehicle image datasets for YOLO pose training and governs how
trained pose models are promoted along a progressive keypoint ladder.

## Language

### Side-view pose ladder

**Rung**:
A fixed subset of the canonical side-view keypoints on the progressive ladder (7KP, 9KP, … 19KP). A model is grown one rung at a time.
_Avoid_: level, stage, tier

**Promotion gate**:
The authoritative pass/fail check deciding whether a candidate model may be promoted to a target rung, judged from holdout prediction evidence. It is the single source of truth for promote/hold.
_Avoid_: check, validation, approval

**Promotion recommendation**:
An advisory signal — the highest rung a candidate already clears, plus the keypoints blocking the next rung. It informs but never overrides the promotion gate.
_Avoid_: verdict (when meaning the authoritative decision), suggestion

**Holdout**:
The set of labelled side-view images held back from training and used as the evidence the promotion gate judges.
_Avoid_: validation set, test set

**Promotion record**:
The append-only trail of every promotion attempt and its outcome — promoted, or held with the blocking keypoints. Holds are recorded as well as promotions.
_Avoid_: log, history, audit
