param(
    [string]$PoseLabelDir = "yolo_training/side_view_dataset/labels_pose_5kp_no_roof",
    [string]$RunName = "side_view_pose_5kp_no_roof_auto"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectRoot = (Resolve-Path (Join-Path $repoRoot "..")).Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$logDir = Join-Path $repoRoot "logs"
$logFile = Join-Path $logDir "pose_training_scheduler.log"
$lockFile = Join-Path $logDir "pose_training.lock"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (Test-Path $lockFile) {
    Add-Content -Path $logFile -Value "[$(Get-Date -Format s)] SKIP: training still running (lock exists)."
    exit 0
}

New-Item -ItemType File -Path $lockFile -Force | Out-Null

try {
    $resolvedLabelDir = Join-Path $repoRoot $PoseLabelDir

    if (-not (Test-Path $resolvedLabelDir)) {
        Add-Content -Path $logFile -Value "[$(Get-Date -Format s)] FAIL: label dir not found: $resolvedLabelDir"
        Write-Error "Pose label dir not found: $resolvedLabelDir"
        exit 1
    }

    Add-Content -Path $logFile -Value "[$(Get-Date -Format s)] START training 5kp no roof_apex"

    $env:POSE_KEYPOINTS = "ground_ref,front_wheel_center,front_wheel_ground,rear_wheel_center,rear_wheel_ground"
    $env:POSE_LABEL_DIR = $resolvedLabelDir
    $env:POSE_ALLOW_LEGACY_SOURCE = "1"
    $env:POSE_REQUIRE_WHEEL_IMAGES = "0"
    $env:POSE_RUN_NAME = $RunName

    & $pythonExe (Join-Path $repoRoot "yolo_training\train_pose.py")

    if ($LASTEXITCODE -ne 0) {
        Add-Content -Path $logFile -Value "[$(Get-Date -Format s)] FAIL training exit=$LASTEXITCODE"
        exit $LASTEXITCODE
    }

    Add-Content -Path $logFile -Value "[$(Get-Date -Format s)] DONE training"
}
finally {
    Remove-Item -Path $lockFile -ErrorAction SilentlyContinue
}
