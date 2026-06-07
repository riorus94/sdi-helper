# Dataset Provisioning from Google Drive

The automated pipelines (`agent1-labeling-10m`, `pose-training-20m`) need two
artifacts that are **gitignored** and therefore absent on a fresh checkout:

- the training images under `dataset_raw/images/train/side/`
- the wheel detector weights at
  `yolo_training/runs/roboflow_v3_local/weights/best.pt`

On the self-hosted GCP runner these persist on local disk. On a GitHub-hosted
runner they are missing, so the pipelines green-skip without doing any work.

`scripts/sync_dataset_from_gdrive.py` closes that gap by pulling a single ZIP
from a **public** Google Drive link (via `gdown`, no credentials).

## 1) Build the ZIP

Zip the two artifacts at their real repo-relative paths:

```
dataset.zip
├── dataset_raw/images/train/side/000001.jpg
├── dataset_raw/images/train/side/000002.jpg
├── ...
└── yolo_training/runs/roboflow_v3_local/weights/best.pt
```

From a machine that has the data (e.g. the GCP runner), at the repo root:

```bash
zip -r dataset.zip \
  dataset_raw/images/train/side \
  yolo_training/runs/roboflow_v3_local/weights/best.pt
```

## 2) Upload to Google Drive and share

Upload `dataset.zip` to Drive, then **Share → General access → "Anyone with the
link"** (Viewer). Copy the share link, e.g.
`https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing`.

## 3) Set the repo variable

Add a repository **variable** (not a secret — the link is already public):

- **Settings → Secrets and variables → Actions → Variables → New variable**
- Name: `GDRIVE_DATASET_URL`
- Value: the share link (or just the `<FILE_ID>`)

Both workflows gate the provisioning step on `vars.GDRIVE_DATASET_URL != ''`, so
nothing changes until this is set.

## How it behaves

- **Hosted runner**: downloads + extracts the ZIP at the repo root, then the
  labeling/training steps find their inputs.
- **Self-hosted GCP runner**: the artifacts already exist on disk, so the script
  detects them and **skips the download** (idempotent). Safe to leave wired.
- **Refresh the data**: re-upload the ZIP (same link) or run the script with
  `--force` locally.

```bash
# Manual / local use
python scripts/sync_dataset_from_gdrive.py --url "$GDRIVE_DATASET_URL"
```

## Training handoff (raw-JSON fallback)

Hosted runners are ephemeral: the screening JSON that `agent1-labeling-10m`
produces does not survive to the separate `pose-training-20m` run. Two things
bridge that gap:

1. `agent1-labeling-10m` uploads its screening JSON as a build artifact
   (`labeling-output-<run>`), so the labeling result is retrievable.
2. `pose-training-20m`'s input selection **prefers** the CLIP-gated screening
   subset, but **falls back** to the committed raw LabelMe JSON
   (`yolo_training/side_view_dataset/labelme_json`, 79 files) when the screening
   subset is empty. So once images are provisioned from Drive, a GitHub-hosted
   run trains a real model instead of green-skipping.

On the GCP runner the screening subset is present, so the fallback never
triggers and behaviour is unchanged. Training on the raw subset is unscreened
(no CLIP orientation gate) — fine for bootstrap/CPU verification; promote to
screened data for production runs.

> **ZIP note for the fallback path:** the raw JSONs reference image names like
> `000001.jpg`, so the images you put under `dataset_raw/images/train/side/` in
> the ZIP must match those stems (this is exactly the `dataset_raw` set the GCP
> runner already holds).
