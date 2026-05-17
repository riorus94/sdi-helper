Set-Location D:\project\sdi-helper
$sourceDir = 'yolo_training\side_view_scrape\images\quality_pass\valid_candidates'
$jsonDir = 'yolo_training\side_view_dataset\labelme_json'
$batchDir = 'yolo_training\side_view_dataset\annotation_batches\batch_010'
if (-not (Test-Path $jsonDir)) { $annotatedStems = @() } else { $annotatedStems = Get-ChildItem -Path $jsonDir -Filter '*.json' | Select-Object -ExpandProperty BaseName }
$extensions = @('.jpg', '.jpeg', '.png', '.webp')
$candidates = Get-ChildItem -Path $sourceDir | Where-Object { $extensions -contains $_.Extension.ToLower() } | Sort-Object Name
$selected = @()
foreach ($file in $candidates) { if ($selected.Count -lt 10 -and ($annotatedStems -notcontains $file.BaseName)) { $selected += $file } }
if (-not (Test-Path $batchDir)) { New-Item -ItemType Directory -Path $batchDir -Force } else { Get-ChildItem -Path $batchDir | Remove-Item -Force }
foreach ($file in $selected) { Copy-Item -Path $file.FullName -Destination $batchDir }
Write-Output 'selected_count='$selected.Count
Write-Output 'batch_path='$batchDir
foreach ($file in $selected) { Write-Output $file.Name }
