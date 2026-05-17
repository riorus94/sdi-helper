$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$labelScript = Join-Path $repoRoot "scripts\run_agent1_5kp_no_roof.ps1"
$trainScript = Join-Path $repoRoot "scripts\run_training_5kp_no_roof.ps1"

$labelTaskName = "SDI-Helper-Agent1-Labeling-10m"
$trainTaskName = "SDI-Helper-Training-20m"

$labelAction = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$labelScript`""
$trainAction = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$trainScript`""

schtasks.exe /Create /F /SC MINUTE /MO 10 /TN $labelTaskName /TR $labelAction | Out-Host
schtasks.exe /Create /F /SC MINUTE /MO 20 /TN $trainTaskName /TR $trainAction | Out-Host

Write-Host ""
Write-Host "Active tasks:"
schtasks.exe /Query /TN $labelTaskName /V /FO LIST | Out-Host
schtasks.exe /Query /TN $trainTaskName /V /FO LIST | Out-Host

Write-Host ""
Write-Host "Manual test now (optional):"
Write-Host "  schtasks /Run /TN $labelTaskName"
Write-Host "  schtasks /Run /TN $trainTaskName"
