# Stanford Agent Mapper

This project should treat Stanford as the only approved source for the clean
side/front/rear training datasets until the mapper is explicitly changed.

Mapper config:

```text
config/stanford_agent_mapper.yaml
```

## Agent Responsibilities

### Agent 1: Keypoint Suggester

Agent 1 determines keypoint locations. It must only run after the image has
passed view/dataset eligibility checks.

For side view, Agent 1 currently supports:

- 5KP no-roof stabilization
- 9KP side-view expansion

Agent 1 must:

- use the mapper-approved priority config
- derive `ground_ref` from wheel-ground anchors
- emit LabelMe JSON with confidence metadata
- emit an Agent 1 quality report
- use `--reject-non-side` for side-view runs

Agent 1 must not:

- decide dataset source eligibility
- silently accept front/rear/three-quarter samples
- mix label schemas in one training label directory
- train the pose model

### Agent 2: View And Geometry Validator

Agent 2 decides whether a Stanford image is eligible for a clean view dataset.
It owns side/front/rear view validation skills.

Agent 2 must:

- accept only `stanford_` stems from mapper-approved roots
- separate side, front, and rear views
- reject non-side images from side datasets
- reject corrupt, tiny, duplicate, top, interior, front/rear/quarter anomalies
- validate Agent 1 keypoint geometry after labels are generated
- produce rejection reasons that Agent 3 can audit

Agent 2 must not:

- create final training labels without Agent 1 output
- override `scrape_exceptions.yaml` without a documented reason

### Agent 3: Dataset Mapper And Orchestrator

Agent 3 reads the mapper config and runs the pipeline in order.

Agent 3 must:

1. read `config/stanford_agent_mapper.yaml`
2. stage clean Stanford-only candidates
3. call Agent 2 view validation
4. call Agent 1 keypoint suggestion
5. call Agent 2 geometry validation
6. write manifests, rejections, summaries, and handoff notes
7. prepare reviewed LabelMe JSON for conversion/training

Agent 3 must not:

- include non-Stanford images
- mix view datasets
- mix 5KP, 9KP, and 19KP labels
- train from `REVIEW_HIGH` or `rejected_non_side_view` without human approval

## Source Rules

Until changed in the mapper config, the only accepted source pattern is:

```text
dataset_raw/images/train/side/stanford_*.jpg
```

Non-Stanford sources such as scraped UUID images, Roboflow exports, or ad hoc
downloads may be used for experiments, but they must not be added to the clean
Stanford dataset.

## Side-View Workflow

Stage Stanford-only side candidates:

```powershell
.\.venv\Scripts\python.exe scripts\stage_side_pose_candidates.py `
  --source-roots dataset_raw\images\train\side `
  --output-root yolo_training\side_view_dataset\subsets\stanford_side_clean `
  --required-stem-prefix stanford_ `
  --limit 300
```

Before labeling, inspect:

```text
yolo_training/side_view_dataset/subsets/stanford_side_clean/manifest.csv
yolo_training/side_view_dataset/subsets/stanford_side_clean/rejections.csv
yolo_training/side_view_dataset/subsets/stanford_side_clean/summary.json
```

Run Agent 1 for 9KP side labels:

```powershell
.\.venv\Scripts\python.exe scripts\suggest_keypoints.py `
  --image-dir yolo_training\side_view_dataset\subsets\stanford_side_clean\images `
  --output yolo_training\side_view_dataset\subsets\stanford_side_clean\labelme_json_9kp `
  --priority-config config\agent1_keypoint_priority_9kp_side.json `
  --phase-only phase1 `
  --orientation-classifier clip `
  --reject-non-side `
  --quality-report yolo_training\side_view_dataset\subsets\stanford_side_clean\agent1_quality_report.csv `
  --overwrite
```

If CLIP is unavailable offline, omit `--orientation-classifier clip`; then
left-looking samples require manual front/rear review.

## Front And Rear Workflow

Front and rear view datasets are planned but not yet approved for training.

Before enabling them, create separate:

- source roots
- mapper entries
- keypoint schemas
- Agent 1 priority configs
- Agent 2 geometry validation rules
- training label directories

Do not reuse side-view keypoints for front or rear view training.

## Promotion Rules

A sample can be promoted into training only when:

- it is Stanford-sourced
- it belongs to the correct view dataset
- it is absent from `scrape_exceptions.yaml`
- it passed Agent 2 view validation
- it has Agent 1 labels with the correct schema
- it passed Agent 2 geometry validation or was manually corrected
- it is not `rejected_non_side_view`

All promotions must be traceable through a manifest.
