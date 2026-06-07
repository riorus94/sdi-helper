.PHONY: install test test-domain test-fast test-slow lint type clean \
	scrape scrape-run scrape-smoke scrape-side scrape-debug \
	build-dataset inspect side-holdout-gate side-19kp-evaluate side-19kp-gate \
	side-rung promote-rung b1-queue-build b1-19kp-accept b2-readiness-gate

# Optional CLI args passthrough for scrape-run.
# Example:
#   make scrape-run SCRAPE_ARGS='--query-contains "side view" --max-queries 10 --max-results 80'
SCRAPE_ARGS ?=
SIDE_HOLDOUT_MODEL ?= ../vehicle-sdi-system/cv_service/models/best.pt
SIDE_HOLDOUT_MANIFEST ?= yolo_training/runs/side_view_pose_7kp_bumper_oos_20260524/holdout_manifest.txt
SIDE_HOLDOUT_OUTPUT ?= yolo_training/runs/side_view_pose_7kp_pre_promotion_gate
SIDE_HOLDOUT_DEVICE ?= cpu
SIDE_HOLDOUT_PYTHON ?= poetry run python
SIDE_19KP_SUMMARY ?= yolo_training/runs/side_view_pose_19kp_candidate/prediction_summary.csv
SIDE_19KP_DECISION ?= yolo_training/runs/side_view_pose_19kp_candidate/gate_decision.json
SIDE_19KP_EVIDENCE ?= yolo_training/runs/side_view_pose_19kp_candidate/holdout_manifest.txt
SIDE_19KP_MODEL ?= yolo_training/runs/side_view_pose_19kp_candidate/weights/best.pt
SIDE_19KP_MANIFEST ?= yolo_training/runs/side_view_pose_19kp_candidate/holdout_manifest.txt
SIDE_19KP_OUTPUT ?= yolo_training/runs/side_view_pose_19kp_candidate
SIDE_19KP_DEVICE ?= cpu
SIDE_19KP_PYTHON ?= poetry run python

SIDE_RUNG ?= 9KP
SIDE_RUNG_INPUT ?= yolo_training/side_view_dataset/labelme_json
SIDE_RUNG_IMG ?= yolo_training/side_view_dataset/images/all
SIDE_RUNG_OUTPUT ?= yolo_training/side_view_dataset/rung_dataset
SIDE_RUNG_VAL_FRACTION ?= 0.15
SIDE_RUNG_PYTHON ?= poetry run python

PROMOTE_RUNG ?= 9KP
PROMOTE_RUNG_MODEL ?= yolo_training/side_view_dataset/rung_dataset/weights/best.pt
PROMOTE_RUNG_MANIFEST ?= yolo_training/side_view_dataset/rung_dataset/holdout_manifest.txt
PROMOTE_RUNG_IMG ?= yolo_training/side_view_dataset/rung_dataset/images/holdout
PROMOTE_RUNG_OUTPUT ?= yolo_training/side_view_dataset/rung_dataset/promotion
PROMOTE_RUNG_CONF ?= 0.5
PROMOTE_RUNG_DEVICE ?= cpu
PROMOTE_RUNG_PYTHON ?= poetry run python

B1_AGENT_REPORT ?= yolo_training/side_view_dataset/b13_agent1_report.csv
B1_VALIDATION_REPORT ?= yolo_training/side_view_dataset/b13_validation_report.csv
B1_QUEUE_OUTPUT ?= yolo_training/side_view_dataset/b13_b1_verification_queue.csv
B1_REVIEW_LOG ?= yolo_training/side_view_dataset/review_queue/b1_19kp_labeling_queue/manual_review_log.csv
B1_DRAFT_JSON_DIR ?= yolo_training/side_view_dataset/review_queue/b1_19kp_labeling_queue/labelme_json_draft_19kp
B1_ACCEPTED_JSON_DIR ?= yolo_training/side_view_dataset/labelme_json
B1_ACCEPTANCE_REPORT ?= yolo_training/side_view_dataset/labelme_json/acceptance_report.csv
B1_PYTHON ?= poetry run python

B2_CANONICAL_JSON_DIR ?= yolo_training/side_view_dataset/labelme_json
B2_QUALITY_CSV ?= yolo_training/side_view_dataset/annotation_batches/batch_013/agent1_quality_report.csv
B2_REPORT ?= yolo_training/side_view_dataset/b2_readiness_report.json
B2_TARGET_RUNG ?= 19KP
B2_PYTHON ?= poetry run python



test:
	poetry run pytest

test-domain:
	poetry run pytest tests/domain -v

test-fast:
	poetry run pytest -m "not slow" -v

test-slow:
	poetry run pytest -m slow -v

lint:
	poetry run ruff check sdi_helper tests

type:
	poetry run mypy sdi_helper

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

scrape:
	poetry run sdi-helper

scrape-run:
	poetry run sdi-helper $(SCRAPE_ARGS)

scrape-smoke:
	poetry run sdi-helper --max-queries 1 --max-results 10 --verbose

scrape-side:
	poetry run sdi-helper --query-contains "side view" --max-queries 10 --max-results 80

scrape-debug:
	poetry run sdi-helper --query-contains "side view" --max-queries 5 --max-results 40 --verbose

build-dataset:
	poetry run python -m sdi_helper.interfaces.cli.build_dataset

inspect:
	poetry run python -m sdi_helper.interfaces.cli.inspect_state

# Regenerate the 41 corrected 7KP YOLO pose labels from LabelMe JSON source.
# Keypoint order must match dataset_pose.yaml: fw_c, fw_g, rw_c, rw_g, g_ref, fb, rb
gen-7kp-labels:
	poetry run python yolo_training/labelme_to_yolo_pose.py \
		--input  yolo_training/side_view_dataset/labelme_json_7kp_bumper_corrected_valid_20260524 \
		--output yolo_training/side_view_dataset/labels_pose_7kp_bumper_corrected_valid_20260524 \
		--img-dir yolo_training/side_view_dataset/pose_dataset/images/train \
		--keypoints "front_wheel_center,front_wheel_ground,rear_wheel_center,rear_wheel_ground,ground_ref,front_bumper,rear_bumper"

# Mandatory pre-promotion gate for side-view 7KP pose candidates.
# Fails nonzero if any holdout image violates the body-end geometry rule.
side-holdout-gate:
	$(SIDE_HOLDOUT_PYTHON) scripts/evaluate_7kp_body_end_model.py \
		--model "$(SIDE_HOLDOUT_MODEL)" \
		--manifest "$(SIDE_HOLDOUT_MANIFEST)" \
		--output-dir "$(SIDE_HOLDOUT_OUTPUT)" \
		--device "$(SIDE_HOLDOUT_DEVICE)"

# Mandatory pre-promotion decision gate for 19KP side-view candidates.
# Fails nonzero if prediction_summary contains FAIL rows or evidence is missing.
side-19kp-evaluate:
	$(SIDE_19KP_PYTHON) scripts/evaluate_19kp_holdout.py \
		--model "$(SIDE_19KP_MODEL)" \
		--manifest "$(SIDE_19KP_MANIFEST)" \
		--output-dir "$(SIDE_19KP_OUTPUT)" \
		--device "$(SIDE_19KP_DEVICE)"

side-19kp-gate:
	$(SIDE_19KP_PYTHON) scripts/gate_side_view_19kp_candidate.py \
		--prediction-summary "$(SIDE_19KP_SUMMARY)" \
		--decision-out "$(SIDE_19KP_DECISION)" \
		--candidate-model "$(SIDE_19KP_MODEL)" \
		--evidence "$(SIDE_19KP_SUMMARY)" \
		--evidence "$(SIDE_19KP_EVIDENCE)" \
		--evidence "$(SIDE_19KP_MODEL)"

# Build the side-view pose dataset for one progressive rung (ADR-004 ladder).
# Validates the source labels first, then converts to the rung's kpt_shape/flip_idx.
#   make side-rung SIDE_RUNG=9KP
side-rung:
	$(SIDE_RUNG_PYTHON) scripts/validate_keypoints.py \
		--json-dir "$(SIDE_RUNG_INPUT)" \
		--report "$(SIDE_RUNG_OUTPUT)/validation_report.csv"
	$(SIDE_RUNG_PYTHON) scripts/build_side_rung_dataset.py \
		--input "$(SIDE_RUNG_INPUT)" \
		--img-dir "$(SIDE_RUNG_IMG)" \
		--output "$(SIDE_RUNG_OUTPUT)" \
		--rung "$(SIDE_RUNG)" \
		--val-fraction "$(SIDE_RUNG_VAL_FRACTION)"

# Run the promotion workflow for one rung (ADR-004): evaluate the candidate on the
# rung's holdout, decide promote/hold (per-rung gate is the authority; recommended
# rung + blockers are advisory), and record the decision + consolidated summary.
# Holdout draws from the clean/screened rung dataset built by `make side-rung`
# (off-frame keypoints are already neutralized at build, PR #11), so label-quality
# issues don't skew the evidence. Exits nonzero on a hold.
#   make promote-rung PROMOTE_RUNG=9KP PROMOTE_RUNG_MODEL=path/to/best.pt
promote-rung:
	$(PROMOTE_RUNG_PYTHON) scripts/run_promotion.py \
		--model "$(PROMOTE_RUNG_MODEL)" \
		--manifest "$(PROMOTE_RUNG_MANIFEST)" \
		--image-dir "$(PROMOTE_RUNG_IMG)" \
		--target-rung "$(PROMOTE_RUNG)" \
		--run-dir "$(PROMOTE_RUNG_OUTPUT)" \
		--confidence-threshold "$(PROMOTE_RUNG_CONF)" \
		--device "$(PROMOTE_RUNG_DEVICE)"

# Build the B1 side-view verification queue from Agent 1 + validation reports.
b1-queue-build:
	$(B1_PYTHON) scripts/build_b1_verification_queue.py \
		--agent-report "$(B1_AGENT_REPORT)" \
		--validation-report "$(B1_VALIDATION_REPORT)" \
		--output "$(B1_QUEUE_OUTPUT)"

# Promote manually accepted B1 19KP draft labels into the canonical training set.
b1-19kp-accept:
	$(B1_PYTHON) scripts/accept_b1_19kp_labels.py \
		--review-log "$(B1_REVIEW_LOG)" \
		--draft-json-dir "$(B1_DRAFT_JSON_DIR)" \
		--accepted-json-dir "$(B1_ACCEPTED_JSON_DIR)" \
		--acceptance-report "$(B1_ACCEPTANCE_REPORT)"

# Consolidated go/no-go gate: is the canonical side-view set ready for a B2 retrain?
# Fails nonzero if any canonical JSON lacks the full keypoint set or any canonical
# image still carries a HIGH/MEDIUM Agent 1 review flag. Pass several CSVs via
#   make b2-readiness-gate B2_QUALITY_CSV='batch_a.csv batch_b.csv'
b2-readiness-gate:
	$(B2_PYTHON) scripts/gate_b2_readiness.py \
		--canonical-json-dir "$(B2_CANONICAL_JSON_DIR)" \
		$(foreach csv,$(B2_QUALITY_CSV),--quality-csv "$(csv)") \
		--report-out "$(B2_REPORT)" \
		--target-rung "$(B2_TARGET_RUNG)"
