# GCP GPU Runner Setup for Automated Training

This setup lets scheduled GitHub Actions training run on Google Cloud GPU instead of local machine.

## What Is Already Automated in Repo
- Labeling every 10 minutes: `.github/workflows/agent1-labeling-10m.yml`
- Training every 20 minutes: `.github/workflows/pose-training-20m.yml`
- Training workflow auto-selects runner target:
  - GCP GPU runner when repo variable `POSE_USE_GCP_GPU=1`
  - GitHub CPU runner otherwise

## 1) Create a GCP VM with GPU
Recommended baseline:
- Ubuntu 22.04
- NVIDIA T4 or L4
- 100+ GB disk

## 2) Install NVIDIA + CUDA on VM
Use GCP marketplace image with NVIDIA driver preinstalled, or install manually.

Validate:
```bash
nvidia-smi
```

## 3) Install GitHub Actions Self-Hosted Runner
On the VM, in a dedicated folder, install runner from your repo Settings > Actions > Runners.

Use labels:
- self-hosted
- linux
- x64
- gcp
- gpu

These labels must match workflow `runs-on: [self-hosted, linux, x64, gcp, gpu]`.

## 4) Configure Runner as Service
Run runner service so it survives reboot.

## 5) Set Repo Variable
In GitHub repo settings, set:
- Name: `POSE_USE_GCP_GPU`
- Value: `1`

With this value, schedule runs every 20 minutes on GCP GPU automatically.

## 6) Optional Manual Override
Trigger workflow manually and choose input:
- `runner_target=gcp-gpu`
- `runner_target=github-cpu`
- `runner_target=auto`

## 7) Quick Verification
1. Push workflow to default branch.
2. Open Actions tab.
3. Run `Pose Training Every 20 Minutes (5KP No Roof Apex)` manually with `gcp-gpu`.
4. Confirm job is picked by your self-hosted runner and artifacts are uploaded.

## Notes
- If GCP runner is offline and `POSE_USE_GCP_GPU=1`, scheduled training waits for runner.
- Set `POSE_USE_GCP_GPU=0` to force schedule fallback to GitHub CPU.
