<#
.SYNOPSIS
    Package the 41-image corrected 7KP body-end dataset into a zip for Google Colab upload.

.DESCRIPTION
    Reads the YOLO label stems from labels_pose_7kp_bumper_corrected_valid_20260524,
    locates the corresponding images, applies an 80/20 train/val split, writes a
    dataset_pose.yaml, and zips the result to $env:USERPROFILE\Downloads\sdi_7kp_colab_dataset.zip.

    Run from anywhere — the script resolves paths relative to its own location.

.EXAMPLE
    .\package_7kp_colab.ps1
    .\package_7kp_colab.ps1 -ValCount 9   # use 9 val images instead of 8
#>
param(
    [int]$ValCount = 8,
    [string]$OutputZip = "$env:USERPROFILE\Downloads\sdi_7kp_colab_dataset.zip"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir      = $PSScriptRoot                                          # …/yolo_training
$LabelDir       = Join-Path $ScriptDir "side_view_dataset\labels_pose_7kp_bumper_corrected_valid_20260524"
$StagingRoot    = Join-Path $ScriptDir "side_view_dataset\_colab_staging"

# Image search roots (ordered — first match wins)
$ImageRoots = @(
    (Join-Path $ScriptDir "side_view_dataset\pose_dataset\images\train"),
    (Join-Path $ScriptDir "side_view_dataset\subsets\pose_candidates_300\images"),
    (Join-Path $ScriptDir "side_view_dataset\subsets\stanford_raw_side_review_low\images"),
    (Join-Path $ScriptDir "side_view_dataset\cars_train")
)

# ── Collect label stems ────────────────────────────────────────────────────
$LabelFiles = Get-ChildItem -Path $LabelDir -Filter "*.txt" | Sort-Object BaseName
if ($LabelFiles.Count -eq 0) {
    Write-Error "No label files found in $LabelDir"
    exit 1
}
Write-Host "Found $($LabelFiles.Count) label files"

# ── Locate images ─────────────────────────────────────────────────────────
$Pairs = [System.Collections.Generic.List[hashtable]]::new()
$Missing = [System.Collections.Generic.List[string]]::new()

foreach ($lf in $LabelFiles) {
    $stem = $lf.BaseName
    $found = $null
    foreach ($root in $ImageRoots) {
        $candidate = Join-Path $root "$stem.jpg"
        if (Test-Path $candidate) { $found = $candidate; break }
        $candidate = Join-Path $root "$stem.png"
        if (Test-Path $candidate) { $found = $candidate; break }
    }
    if ($null -eq $found) {
        # broader fallback search
        $hits = Get-ChildItem -Recurse -Path $ScriptDir -Filter "$stem.jpg" -ErrorAction SilentlyContinue |
                    Select-Object -First 1 -ExpandProperty FullName
        if ($hits) { $found = $hits }
    }
    if ($null -ne $found) {
        $Pairs.Add(@{ Stem = $stem; Image = $found; Label = $lf.FullName })
    } else {
        $Missing.Add($stem)
    }
}

if ($Missing.Count -gt 0) {
    Write-Warning "Could not locate images for $($Missing.Count) stems:"
    $Missing | ForEach-Object { Write-Warning "  $_" }
}

$Total = $Pairs.Count
Write-Host "Matched $Total image/label pairs"
if ($Total -lt 4) { Write-Error "Too few pairs ($Total) to split"; exit 1 }

# ── Train / val split (deterministic — last N = val) ──────────────────────
$ValCount  = [math]::Min($ValCount, [int]($Total * 0.25))
$ValCount  = [math]::Max($ValCount, 1)
$TrainCount = $Total - $ValCount

$TrainPairs = $Pairs | Select-Object -First $TrainCount
$ValPairs   = $Pairs | Select-Object -Last  $ValCount

Write-Host "Split: train=$TrainCount  val=$ValCount"

# ── Build staging tree ────────────────────────────────────────────────────
if (Test-Path $StagingRoot) { Remove-Item -Recurse -Force $StagingRoot }

$Dirs = @(
    "$StagingRoot\images\train",
    "$StagingRoot\images\val",
    "$StagingRoot\labels\train",
    "$StagingRoot\labels\val"
)
$Dirs | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

function Copy-Pair {
    param($pair, [string]$split)
    $ext = [IO.Path]::GetExtension($pair.Image)
    Copy-Item -Path $pair.Image -Destination "$StagingRoot\images\$split\$($pair.Stem)$ext"
    Copy-Item -Path $pair.Label -Destination "$StagingRoot\labels\$split\$($pair.Stem).txt"
}

$TrainPairs | ForEach-Object { Copy-Pair $_ "train" }
$ValPairs   | ForEach-Object { Copy-Pair $_ "val"   }

# ── dataset_pose.yaml ─────────────────────────────────────────────────────
$Yaml = @"
# 7KP body-end side-view pose dataset — packaged for Google Colab
# Keypoints: front_wheel_center, front_wheel_ground, rear_wheel_center,
#             rear_wheel_ground, ground_ref, front_bumper, rear_bumper
path: /content/sdi_7kp_dataset
train: images/train
val:   images/val

kpt_shape: [7, 3]
# flip_idx maps: fw_c<->rw_c, fw_g<->rw_g, front_bumper<->rear_bumper, ground_ref stays
flip_idx: [2, 3, 0, 1, 4, 6, 5]

nc: 1
names:
  0: vehicle
"@
$Yaml | Set-Content -Path "$StagingRoot\dataset_pose.yaml" -Encoding UTF8

# ── Manifest ──────────────────────────────────────────────────────────────
$Manifest = "stem,split`n"
$TrainPairs | ForEach-Object { $Manifest += "$($_.Stem),train`n" }
$ValPairs   | ForEach-Object { $Manifest += "$($_.Stem),val`n" }
$Manifest | Set-Content -Path "$StagingRoot\manifest.csv" -Encoding UTF8

Write-Host "Staging at: $StagingRoot"

# ── Zip using Python so entries use forward slashes (Linux-compatible) ────
if (Test-Path $OutputZip) { Remove-Item -Force $OutputZip }
$StagingRootFwd = $StagingRoot.Replace('\', '/')
python -c @"
import zipfile, pathlib, os
staging = pathlib.Path(r'$StagingRoot')
out     = r'$OutputZip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in staging.rglob('*'):
        if f.is_file():
            arcname = f.relative_to(staging).as_posix()  # forward slashes
            zf.write(f, arcname)
print('Zip written:', out)
"@
Write-Host ""
Write-Host "Done. Upload this file to Colab:"
Write-Host "  $OutputZip"
Write-Host ""
Write-Host "Train: $TrainCount  Val: $ValCount  Total: $Total"
