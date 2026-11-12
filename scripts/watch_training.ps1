# Live view of the overnight training, refreshed every 5 seconds.
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\watch_training.ps1
# Ctrl+C stops watching. Training keeps going either way.
$root = Split-Path $PSScriptRoot -Parent
$yoloLog = Join-Path $root "results\detector_training.log"
$yoloScores = Join-Path $root "results\detection\detector\results.csv"
$cnnLog = Join-Path $root "results\global_training.log"
$cnnRuns = Join-Path $root "results\training"
$colourCode = "$([char]27)\[[0-9;]*[A-Za-z]"

function Read-Tail($path, $bytes = 8192) {
    # Only the end of the log, shared with the training process that is still writing it
    if (-not (Test-Path $path)) { return "" }
    $stream = [IO.File]::Open($path, "Open", "Read", "ReadWrite")
    try {
        $stream.Seek([Math]::Max(0, $stream.Length - $bytes), "Begin") | Out-Null
        return (New-Object IO.StreamReader($stream)).ReadToEnd()
    } finally { $stream.Dispose() }
}

while ($true) {
    $lines = @("Overnight training   $(Get-Date -Format 'HH:mm:ss')   (Ctrl+C stops watching, not training)", "")
    $gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null
    $memory = Get-CimInstance Win32_OperatingSystem
    $lines += "GPU use, memory used, memory total: $gpu"
    $lines += "Free virtual memory: {0:N1} GB" -f ($memory.FreeVirtualMemory / 1MB)
    $lines += ""

    $lines += "1. YOLO detector (5 hours)"
    $progress = ((Read-Tail $yoloLog) -replace $colourCode, "") -split "`r|`n" | Where-Object { $_ -match '^\s+\d+/\d+\s' } | Select-Object -Last 1
    if ($progress) { $lines += "   Now: epoch " + $progress.Trim() }
    if (Test-Path $yoloScores) {
        $lines += "   After each epoch, on validation frames:"
        $lines += "   epoch  precision  recall  mAP50  mAP50-95"
        foreach ($row in Import-Csv $yoloScores) {
            $lines += "   {0,5}  {1,9:N3}  {2,6:N3}  {3,5:N3}  {4,8:N3}" -f [int]$row.epoch, [double]$row."metrics/precision(B)", [double]$row."metrics/recall(B)", [double]$row."metrics/mAP50(B)", [double]$row."metrics/mAP50-95(B)"
        }
    }
    $lines += ""

    $lines += "2. Global CNN, 6 runs (about 3 hours)"
    if (Test-Path $cnnLog) {
        Get-Content $cnnLog -Tail 8 | ForEach-Object { $lines += "   $_" }
        $latest = Get-ChildItem $cnnRuns -Filter "cnn_global_*.csv" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
        if ($latest) {
            $row = Import-Csv $latest.FullName | Select-Object -Last 1
            $lines += "   {0}: epoch {1}, validation F1 {2}" -f $latest.BaseName, $row.epoch, $row.val_f1
        }
    } else {
        $lines += "   Starts after YOLO"
    }

    Clear-Host
    $lines | ForEach-Object { Write-Host $_ }
    Start-Sleep -Seconds 5
}
