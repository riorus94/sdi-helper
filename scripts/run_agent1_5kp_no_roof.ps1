param(
    [string]$ImageDir = "dataset_raw/images/train/side",
    [string]$OutputDir = "yolo_training/side_view_dataset/labelme_json_stanford_screening"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectRoot = (Resolve-Path (Join-Path $repoRoot "..")).Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$priorityConfig = Join-Path $repoRoot "config\agent1_keypoint_priority_5kp_no_roof_apex.json"
$logDir = Join-Path $repoRoot "logs"
$logFile = Join-Path $logDir "agent1_labeling_scheduler.log"
$lockFile = Join-Path $logDir "agent1_labeling.lock"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (Test-Path $lockFile) {
    Add-Content -Path $logFile -Value "[$(Get-Date -Format s)] SKIP: labeling still running (lock exists)."
    exit 0
}

New-Item -ItemType File -Path $lockFile -Force | Out-Null

try {
    $resolvedImageDir = Join-Path $repoRoot $ImageDir
    $resolvedOutputDir = Join-Path $repoRoot $OutputDir

    Add-Content -Path $logFile -Value "[$(Get-Date -Format s)] START labeling 5kp no roof_apex"

    & $pythonExe (Join-Path $repoRoot "scripts\suggest_keypoints.py") `
        --image-dir $resolvedImageDir `
        --output $resolvedOutputDir `
        --priority-config $priorityConfig `
        --phase-only phase1 `
        --overwrite

    if ($LASTEXITCODE -ne 0) {
        Add-Content -Path $logFile -Value "[$(Get-Date -Format s)] FAIL labeling exit=$LASTEXITCODE"
        exit $LASTEXITCODE
    }

    Add-Content -Path $logFile -Value "[$(Get-Date -Format s)] DONE labeling"
}
finally {
    Remove-Item -Path $lockFile -ErrorAction SilentlyContinue
}
