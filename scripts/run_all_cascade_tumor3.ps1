param(
    [string]$Device = "cuda",
    [string]$PythonExe = "",
    [string]$ExpertClasses = "adenocarcinoma,large.cell.carcinoma,squamous.cell.carcinoma",
    [int]$ExpertTriggerTopK = 2,
    [double]$ExpertMarginThreshold = 0.12,
    [int]$NumWorkers = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "..\LCC_GPU\python.exe"))
}

$mainPy = Join-Path $repoRoot "src\main.py"
$outputsDir = Join-Path $repoRoot "outputs"
$logDir = Join-Path $outputsDir "cascade_batch_logs"
$null = New-Item -ItemType Directory -Force -Path $logDir

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path $mainPy)) {
    throw "main.py not found: $mainPy"
}

$jobs = @(
    @{ Name = "v2.0_pretrained_ce"; Args = @("--pretrained", "--loss", "cross_entropy") }
    @{ Name = "v2.1_pretrained_ce_cosine"; Args = @("--pretrained", "--loss", "cross_entropy", "--scheduler", "cosine") }
    @{ Name = "v2.2_pretrained_ce_ls"; Args = @("--pretrained", "--loss", "cross_entropy", "--label-smoothing", "0.1") }
    @{ Name = "v2.3_pretrained_ce_ls_cosine"; Args = @("--pretrained", "--loss", "cross_entropy", "--label-smoothing", "0.1", "--scheduler", "cosine") }
    @{ Name = "v2.4_pretrained_ce_ls_cosine_weightedce"; Args = @("--pretrained", "--loss", "cross_entropy", "--label-smoothing", "0.1", "--scheduler", "cosine", "--class-weighting", "balanced") }
    @{ Name = "v3.0_pretrained_focal_ls_cosine"; Args = @("--pretrained", "--loss", "focal", "--label-smoothing", "0.1", "--scheduler", "cosine") }
    @{ Name = "v3.1_pretrained_focal_ls_cosine_mixup"; Args = @("--pretrained", "--loss", "focal", "--label-smoothing", "0.1", "--scheduler", "cosine", "--mixup-alpha", "0.2") }
    @{ Name = "v3.2_pretrained_focal_ls_cosine_cutmix"; Args = @("--pretrained", "--loss", "focal", "--label-smoothing", "0.1", "--scheduler", "cosine", "--cutmix-alpha", "0.5") }
    @{ Name = "v3.3_pretrained_focal_ls_cosine_se"; Args = @("--pretrained", "--loss", "focal", "--label-smoothing", "0.1", "--scheduler", "cosine", "--feature-attention", "se") }
    @{ Name = "v3.4_pretrained_focal_ls_cosine_cbam"; Args = @("--pretrained", "--loss", "focal", "--label-smoothing", "0.1", "--scheduler", "cosine", "--feature-attention", "cbam") }
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "run_all_cascade_tumor3_$timestamp.log"
$marginText = $ExpertMarginThreshold.ToString([System.Globalization.CultureInfo]::InvariantCulture)

Start-Transcript -Path $logPath | Out-Null
try {
    Write-Host "Repo root: $repoRoot"
    Write-Host "Python: $PythonExe"
    Write-Host "Expert classes: $ExpertClasses"
    Write-Host "Trigger: top-$ExpertTriggerTopK, margin <= $marginText"
    Write-Host "Num workers: $NumWorkers"

    foreach ($job in $jobs) {
        $mainName = $job.Name
        $mainDir = Join-Path $outputsDir $mainName
        $mainCkpt = Join-Path $mainDir "best_model.pt"
        $expertOut = Join-Path $outputsDir ("expert_tumor3_" + $mainName)
        $expertCkpt = Join-Path $expertOut "best_model.pt"
        $cascadeOut = Join-Path $outputsDir ("cascade_" + $mainName)
        $cascadeSummary = Join-Path $cascadeOut "metrics_summary.json"

        if (-not (Test-Path $mainCkpt)) {
            throw "Main checkpoint not found: $mainCkpt"
        }

        if (Test-Path $expertCkpt) {
            Write-Host ""
            Write-Host "========== SKIP EXPERT (checkpoint exists): $mainName =========="
            Write-Host $expertCkpt
        } else {
            Write-Host ""
            Write-Host "========== TRAIN EXPERT: $mainName =========="
            $expertCmd = @(
                $mainPy,
                "--run-mode", "expert",
                "--expert-classes", $ExpertClasses,
                "--device", $Device,
                "--epochs", "25",
                "--batch-size", "16",
                "--image-size", "224",
                "--lr", "3e-4",
                "--weight-decay", "1e-4",
                "--num-workers", $NumWorkers.ToString()
            ) + $job.Args + @(
                "--output-dir", $expertOut
            )
            & $PythonExe @expertCmd
            if ($LASTEXITCODE -ne 0) {
                throw "Expert training failed: $mainName"
            }
        }

        if (-not (Test-Path $expertCkpt)) {
            throw "Expert checkpoint not found after training: $expertCkpt"
        }

        if (Test-Path $cascadeSummary) {
            Write-Host ""
            Write-Host "========== SKIP CASCADE (summary exists): $mainName =========="
            Write-Host $cascadeSummary
        } else {
            Write-Host ""
            Write-Host "========== CASCADE EVAL: $mainName =========="
            $cascadeCmd = @(
                $mainPy,
                "--run-mode", "cascade",
                "--device", $Device,
                "--main-checkpoint", $mainCkpt,
                "--expert-checkpoint", $expertCkpt,
                "--expert-trigger-topk", $ExpertTriggerTopK.ToString(),
                "--expert-margin-threshold", $marginText,
                "--output-dir", $cascadeOut
            )
            & $PythonExe @cascadeCmd
            if ($LASTEXITCODE -ne 0) {
                throw "Cascade evaluation failed: $mainName"
            }
        }
    }

    Write-Host ""
    Write-Host "All expert + cascade runs finished."
    Write-Host "Results saved under: $outputsDir"
    Write-Host "Batch log saved to: $logPath"
}
finally {
    Stop-Transcript | Out-Null
}
