param(
    [ValidateSet("valid", "invalid", "labeled", "wheel_train", "wheel_val")]
    [string]$Split = "valid"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$labelmeExe = Join-Path $repoRoot ".venv\Scripts\labelme.exe"
$labelsFile = Join-Path $repoRoot "yolo_training\labelme_labels.txt"

switch ($Split) {
    "valid" {
        $imagesDir = Join-Path $repoRoot "yolo_training\side_view_scrape\images\quality_pass\valid_candidates"
    }
    "invalid" {
        $imagesDir = Join-Path $repoRoot "yolo_training\side_view_scrape\images\quality_pass\invalid_candidates"
    }
    "labeled" {
        $imagesDir = Join-Path $repoRoot "dataset_raw\images\train\labeled_from_candidates"
    }
    "wheel_train" {
        $imagesDir = Join-Path $repoRoot "yolo_training\dataset\images\train"
    }
    "wheel_val" {
        $imagesDir = Join-Path $repoRoot "yolo_training\dataset\images\val"
    }
}

if (-not (Test-Path $labelmeExe)) {
    throw "LabelMe was not found at: $labelmeExe"
}

if (-not (Test-Path $labelsFile)) {
    throw "Label list file was not found at: $labelsFile"
}

if (-not (Test-Path $imagesDir)) {
    throw "Image directory was not found at: $imagesDir"
}

Push-Location $repoRoot
try {
    & $labelmeExe $imagesDir --labels $labelsFile
}
finally {
    Pop-Location
}
