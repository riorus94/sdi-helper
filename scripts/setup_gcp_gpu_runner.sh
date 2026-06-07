#!/usr/bin/env bash
# Bootstrap a GCP GPU VM as a GitHub Actions self-hosted runner for automated
# pose training (see docs/gcp-gpu-runner-setup.md).
#
# Run this ON the Ubuntu VM (not in CI). It:
#   1. verifies the NVIDIA GPU is visible (installs the driver if asked),
#   2. installs the system libs the training deps need (OpenCV),
#   3. downloads + configures the GitHub Actions runner with the labels the
#      pose-training workflow targets: self-hosted,linux,x64,gcp,gpu,
#   4. installs it as a systemd service so it survives reboot.
#
# Usage:
#   sudo REPO_URL=https://github.com/riorus94/sdi-helper \
#        RUNNER_TOKEN=<token from repo Settings > Actions > Runners > New> \
#        ./scripts/setup_gcp_gpu_runner.sh
#
# Optional env: RUNNER_VERSION (default 2.319.1), RUNNER_NAME (default gcp-gpu-$(hostname)),
#               RUNNER_DIR (default /opt/actions-runner), INSTALL_NVIDIA_DRIVER=1.
set -euo pipefail

REPO_URL="${REPO_URL:-}"
RUNNER_TOKEN="${RUNNER_TOKEN:-}"
RUNNER_VERSION="${RUNNER_VERSION:-2.319.1}"
RUNNER_NAME="${RUNNER_NAME:-gcp-gpu-$(hostname)}"
RUNNER_DIR="${RUNNER_DIR:-/opt/actions-runner}"
RUNNER_LABELS="self-hosted,linux,x64,gcp,gpu"
INSTALL_NVIDIA_DRIVER="${INSTALL_NVIDIA_DRIVER:-0}"

die() { echo "ERROR: $*" >&2; exit 1; }

[ -n "$REPO_URL" ] || die "REPO_URL is required (e.g. https://github.com/<owner>/<repo>)"
[ -n "$RUNNER_TOKEN" ] || die "RUNNER_TOKEN is required (Settings > Actions > Runners > New self-hosted runner)"

# The runner must not run as root; pick a non-root owner for the service.
RUN_AS_USER="${SUDO_USER:-$(id -un)}"
[ "$RUN_AS_USER" != "root" ] || die "Run via sudo from a non-root user, or set SUDO_USER — the runner refuses to run as root."

echo "==> 1/4 Checking NVIDIA GPU"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  if [ "$INSTALL_NVIDIA_DRIVER" = "1" ]; then
    echo "    Installing NVIDIA driver via ubuntu-drivers (this can take a while)..."
    apt-get update -y
    apt-get install -y ubuntu-drivers-common
    ubuntu-drivers autoinstall
    echo "    Driver installed — a REBOOT is required, then re-run this script."
    exit 0
  fi
  die "nvidia-smi not working. Use a GCP image with the driver preinstalled, or re-run with INSTALL_NVIDIA_DRIVER=1."
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "==> 2/4 Installing system libs (OpenCV runtime + git/curl)"
apt-get update -y
apt-get install -y --no-install-recommends git curl ca-certificates libgl1 libglib2.0-0

echo "==> 3/4 Installing the GitHub Actions runner into $RUNNER_DIR"
mkdir -p "$RUNNER_DIR"
chown "$RUN_AS_USER": "$RUNNER_DIR"
tarball="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
if [ ! -f "$RUNNER_DIR/config.sh" ]; then
  sudo -u "$RUN_AS_USER" bash -c "
    set -euo pipefail
    cd '$RUNNER_DIR'
    curl -fsSL -o '$tarball' \
      'https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${tarball}'
    tar xzf '$tarball'
    rm -f '$tarball'
  "
fi

echo "==> Configuring runner '$RUNNER_NAME' with labels: $RUNNER_LABELS"
sudo -u "$RUN_AS_USER" bash -c "
  set -euo pipefail
  cd '$RUNNER_DIR'
  ./config.sh --unattended --replace \
    --url '$REPO_URL' \
    --token '$RUNNER_TOKEN' \
    --name '$RUNNER_NAME' \
    --labels '$RUNNER_LABELS' \
    --work _work
"

echo "==> 4/4 Installing + starting the runner service"
cd "$RUNNER_DIR"
./svc.sh install "$RUN_AS_USER"
./svc.sh start
./svc.sh status || true

cat <<EOF

Done. Runner '$RUNNER_NAME' is registered with labels [$RUNNER_LABELS] and running as a service.

Next:
  - Set repo variable POSE_USE_GCP_GPU=1 (Settings > Secrets and variables > Actions > Variables).
  - Ensure training data is available to the runner: either set GDRIVE_DATASET_URL
    (the workflow provisions it per run) or pre-stage dataset_raw on this VM.
  - Verify: Actions > "Pose Training Every 20 Minutes" > Run workflow > runner_target=gcp-gpu.
EOF
