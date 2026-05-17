param(
    [string]$RepoDir = $(if ($env:REPO_DIR) { $env:REPO_DIR } else { 'D:\project\sdi-helper' }),
    [string]$LogDir = $(if ($env:LOG_DIR) { $env:LOG_DIR } else { Join-Path $RepoDir 'logs' }),
    [string]$LogFile = $(if ($env:LOG_FILE) { $env:LOG_FILE } else { Join-Path $LogDir 'scrape-cron.log' })
)

$ErrorActionPreference = 'Stop'

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $RepoDir

# PYTHONUNBUFFERED ensures Python flushes every log line immediately instead of block-buffering.
$env:PYTHONUNBUFFERED = '1'

# Let cmd.exe append both stdout AND stderr directly to the log file at the OS level.
# This bypasses PowerShell's pipeline entirely, which would only forward the first stderr line
# (NativeCommandError limitation in PS 5.1).  The script exit code mirrors Python's exit code.
$pyExe  = "$RepoDir\.venv\Scripts\python.exe"
$pyCode = 'from sdi_helper.interfaces.cli.run_scrape import main; main()'
$separator = "=== $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') ==="
Add-Content -Path $LogFile -Value $separator
cmd /c "`"$pyExe`" -c `"$pyCode`" >> `"$LogFile`" 2>&1"
exit $LASTEXITCODE