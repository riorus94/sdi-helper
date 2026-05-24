.PHONY: install test test-domain test-fast test-slow lint type clean \
	scrape scrape-run scrape-smoke scrape-side scrape-debug \
	build-dataset inspect side-holdout-gate

# Optional CLI args passthrough for scrape-run.
# Example:
#   make scrape-run SCRAPE_ARGS='--query-contains "side view" --max-queries 10 --max-results 80'
SCRAPE_ARGS ?=
SIDE_HOLDOUT_MODEL ?= ../vehicle-sdi-system/cv_service/models/best.pt
SIDE_HOLDOUT_MANIFEST ?= yolo_training/runs/side_view_pose_7kp_bumper_oos_20260524/holdout_manifest.txt
SIDE_HOLDOUT_OUTPUT ?= yolo_training/runs/side_view_pose_7kp_pre_promotion_gate
SIDE_HOLDOUT_DEVICE ?= cpu
SIDE_HOLDOUT_PYTHON ?= poetry run python

install:
	poetry install

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
