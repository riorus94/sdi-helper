# GCP GPU Runner Setup for Automated Training

Run scheduled GitHub Actions pose training on a Google Cloud **GPU** VM instead of
the GitHub-hosted CPU runner. Once set up, the every-20-minutes workflow trains on
GPU automatically.

## What is already automated in the repo

- Training every 20 minutes: `.github/workflows/pose-training-20m.yml`
- The workflow auto-selects the runner target:
  - **GCP GPU** runner when repo variable `POSE_USE_GCP_GPU=1`
  - GitHub CPU runner otherwise
  - Manual override: run it from the Actions tab with `runner_target = gcp-gpu | github-cpu | auto`
- The GPU job targets `runs-on: [self-hosted, linux, x64, gcp, gpu]` — the runner
  must advertise exactly those labels (the bootstrap script sets them).

## 1) Create a GPU VM

Example (NVIDIA L4 on Ubuntu 22.04, 150 GB disk):

```bash
gcloud compute instances create sdi-gpu-runner \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=150GB
```

A T4 (`--accelerator=type=nvidia-tesla-t4,count=1` on an `n1-standard-8`) also
works. Prefer a GCP image with the NVIDIA driver preinstalled; otherwise the
bootstrap script can install it (`INSTALL_NVIDIA_DRIVER=1`, then reboot + re-run).

## 2) Get a runner registration token

Repo → **Settings → Actions → Runners → New self-hosted runner**. Copy the
registration token (it expires within ~1 hour).

## 3) Bootstrap the VM (one script)

SSH into the VM, clone the repo, and run:

```bash
sudo REPO_URL=https://github.com/riorus94/sdi-helper \
     RUNNER_TOKEN=<token from step 2> \
     ./scripts/setup_gcp_gpu_runner.sh
```

The script verifies the GPU (`nvidia-smi`), installs the OpenCV system libs,
downloads + configures the Actions runner with labels
`self-hosted,linux,x64,gcp,gpu`, and installs it as a systemd service that
survives reboot. (Python 3.11 + training deps are installed per-run by the
workflow via `actions/setup-python` + `pip install -e .`; the default PyPI
`torch` wheel is CUDA-enabled, and Ultralytics auto-detects the GPU.)

## 4) Turn it on

Set the repo **variable** (Settings → Secrets and variables → Actions → Variables):

- Name: `POSE_USE_GCP_GPU`
- Value: `1`

Scheduled runs now go to the GPU runner automatically.

## 5) Make the data available to the runner

A self-hosted runner still does a clean checkout each run, and `dataset_raw/`
(images) + the wheel model are gitignored — so the GPU job needs the data, same
as a hosted runner. Two options:

- **Set `GDRIVE_DATASET_URL`** (recommended) — the workflow's provisioning step
  pulls the dataset ZIP from Google Drive each run; idempotent, so it no-ops once
  the data is present. See `docs/dataset-provisioning-gdrive.md`.
- **Pre-stage `dataset_raw/` on the VM** and keep it out of the cleaned workspace.

## 6) Verify

1. Actions tab → **Pose Training Every 20 Minutes** → **Run workflow** →
   `runner_target = gcp-gpu`.
2. Confirm the job is picked up by your self-hosted runner.
3. Confirm `nvidia-smi`-backed training runs and a model artifact is uploaded.

## Notes

- If the GCP runner is offline while `POSE_USE_GCP_GPU=1`, scheduled runs **queue**
  waiting for it. Set `POSE_USE_GCP_GPU=0` to fall back to the GitHub CPU runner.
- Stop billing when idle: `gcloud compute instances stop sdi-gpu-runner`. The
  runner service resumes on start.
- Remove a runner: on the VM, `cd /opt/actions-runner && sudo ./svc.sh stop &&
  sudo ./svc.sh uninstall && ./config.sh remove --token <removal-token>`.
