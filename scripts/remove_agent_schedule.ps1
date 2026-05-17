$ErrorActionPreference = "Continue"

$labelTaskName = "SDI-Helper-Agent1-Labeling-10m"
$trainTaskName = "SDI-Helper-Training-20m"

schtasks.exe /Delete /F /TN $labelTaskName | Out-Host
schtasks.exe /Delete /F /TN $trainTaskName | Out-Host

Write-Host "Removed scheduler tasks (if they existed)."
