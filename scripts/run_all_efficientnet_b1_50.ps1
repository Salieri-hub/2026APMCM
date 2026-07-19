param(
    [string]$Device = "cuda",
    [string]$PythonExe = "",
    [int]$ExpertTriggerTopK = 2,
    [double]$ExpertMarginThreshold = 0.12,
    [int]$NumWorkers = 0,
    [int]$Epochs = 25,
    [int]$BatchSize = 16,
    [int]$ImageSize = 240,
    [string]$ModelName = "efficientnet_b1"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "..\LCC_GPU\python.exe"))
} elseif (-not [System.IO.Path]::IsPathRooted($PythonExe)) {
    $cwdCandidate = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $PythonExe))
    $repoCandidate = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $PythonExe))
    if (Test-Path $cwdCandidate) {
        $PythonExe = $cwdCandidate
    } else {
        $PythonExe = $repoCandidate
    }
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
    @{ Name = "v2.0_pretrained_ce_b1"; Args = @("--pretrained", "--loss", "cross_entropy") }
    @{ Name = "v2.1_pretrained_ce_cosine_b1"; Args = @("--pretrained", "--loss", "cross_entropy", "--scheduler", "cosine") }
    @{ Name = "v2.2_pretrained_ce_ls_b1"; Args = @("--pretrained", "--loss", "cross_entropy", "--label-smoothing", "0.1") }
    @{ Name = "v2.3_pretrained_ce_ls_cosine_b1"; Args = @("--pretrained", "--loss", "cross_entropy", "--label-smoothing", "0.1", "--scheduler", "cosine") }
    @{ Name = "v2.4_pretrained_ce_ls_cosine_weightedce_b1"; Args = @("--pretrained", "--loss", "cross_entropy", "--label-smoothing", "0.1", "--scheduler", "cosine", "--class-weighting", "balanced") }
    @{ Name = "v3.0_pretrained_focal_ls_cosine_b1"; Args = @("--pretrained", "--loss", "focal", "--label-smoothing", "0.1", "--scheduler", "cosine") }
    @{ Name = "v3.1_pretrained_focal_ls_cosine_mixup_b1"; Args = @("--pretrained", "--loss", "focal", "--label-smoothing", "0.1", "--scheduler", "cosine", "--mixup-alpha", "0.2") }
    @{ Name = "v3.2_pretrained_focal_ls_cosine_cutmix_b1"; Args = @("--pretrained", "--loss", "focal", "--label-smoothing", "0.1", "--scheduler", "cosine", "--cutmix-alpha", "0.5") }
    @{ Name = "v3.3_pretrained_focal_ls_cosine_se_b1"; Args = @("--pretrained", "--loss", "focal", "--label-smoothing", "0.1", "--scheduler", "cosine", "--feature-attention", "se") }
    @{ Name = "v3.4_pretrained_focal_ls_cosine_cbam_b1"; Args = @("--pretrained", "--loss", "focal", "--label-smoothing", "0.1", "--scheduler", "cosine", "--feature-attention", "cbam") }
)

$expertTumor3 = "adenocarcinoma,large.cell.carcinoma,squamous.cell.carcinoma"
$expertPairs = @(
    @{ Tag = "ad_lc"; Classes = "adenocarcinoma,large.cell.carcinoma"; Label = "adenocarcinoma vs large.cell.carcinoma" }
    @{ Tag = "ad_sq"; Classes = "adenocarcinoma,squamous.cell.carcinoma"; Label = "adenocarcinoma vs squamous.cell.carcinoma" }
    @{ Tag = "lc_sq"; Classes = "large.cell.carcinoma,squamous.cell.carcinoma"; Label = "large.cell.carcinoma vs squamous.cell.carcinoma" }
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "run_all_efficientnet_b1_50_$timestamp.log"
$marginText = $ExpertMarginThreshold.ToString([System.Globalization.CultureInfo]::InvariantCulture)

function Invoke-CodexPython {
    param(
        [string[]]$CommandArgs,
        [string]$FailureMessage
    )

    & $PythonExe @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

Start-Transcript -Path $logPath | Out-Null
try {
    Write-Host "Repo root: $repoRoot"
    Write-Host "Python: $PythonExe"
    Write-Host "Model: $ModelName"
    Write-Host "Image size: $ImageSize"
    Write-Host "Epochs: $Epochs"
    Write-Host "Batch size: $BatchSize"
    Write-Host "Num workers: $NumWorkers"
    Write-Host "Trigger: top-$ExpertTriggerTopK, margin <= $marginText"
    Write-Host "Single-model runs: $($jobs.Count)"
    Write-Host "Tumor3 cascade runs: $($jobs.Count)"
    Write-Host "Pairwise cascade runs: $($jobs.Count * $expertPairs.Count)"
    Write-Host "Formal experiment outputs to produce: $($jobs.Count * 5)"
    Write-Host "Expert training runs to produce: $($jobs.Count * 4)"

    foreach ($job in $jobs) {
        $runName = $job.Name
        $mainOut = Join-Path $outputsDir $runName
        $mainCkpt = Join-Path $mainOut "best_model.pt"
        $mainSummary = Join-Path $mainOut "metrics_summary.json"

        if (Test-Path $mainSummary) {
            Write-Host ""
            Write-Host "========== SKIP MAIN (summary exists): $runName =========="
            Write-Host $mainSummary
        } else {
            Write-Host ""
            Write-Host "========== TRAIN MAIN: $runName =========="
            $mainCmd = @(
                $mainPy,
                "--run-mode", "single",
                "--model-name", $ModelName,
                "--device", $Device,
                "--epochs", $Epochs.ToString(),
                "--batch-size", $BatchSize.ToString(),
                "--image-size", $ImageSize.ToString(),
                "--lr", "3e-4",
                "--weight-decay", "1e-4",
                "--num-workers", $NumWorkers.ToString()
            ) + $job.Args + @(
                "--output-dir", $mainOut
            )
            Invoke-CodexPython -CommandArgs $mainCmd -FailureMessage "Main training failed: $runName"
        }

        if (-not (Test-Path $mainCkpt)) {
            throw "Main checkpoint not found after training: $mainCkpt"
        }

        $tumor3ExpertOut = Join-Path $outputsDir ("expert_tumor3_" + $runName)
        $tumor3ExpertCkpt = Join-Path $tumor3ExpertOut "best_model.pt"
        $tumor3CascadeOut = Join-Path $outputsDir ("cascade_" + $runName)
        $tumor3CascadeSummary = Join-Path $tumor3CascadeOut "metrics_summary.json"

        if (Test-Path $tumor3ExpertCkpt) {
            Write-Host ""
            Write-Host "========== SKIP TUMOR3 EXPERT (checkpoint exists): $runName =========="
            Write-Host $tumor3ExpertCkpt
        } else {
            Write-Host ""
            Write-Host "========== TRAIN TUMOR3 EXPERT: $runName =========="
            $tumor3ExpertCmd = @(
                $mainPy,
                "--run-mode", "expert",
                "--model-name", $ModelName,
                "--expert-classes", $expertTumor3,
                "--device", $Device,
                "--epochs", $Epochs.ToString(),
                "--batch-size", $BatchSize.ToString(),
                "--image-size", $ImageSize.ToString(),
                "--lr", "3e-4",
                "--weight-decay", "1e-4",
                "--num-workers", $NumWorkers.ToString()
            ) + $job.Args + @(
                "--output-dir", $tumor3ExpertOut
            )
            Invoke-CodexPython -CommandArgs $tumor3ExpertCmd -FailureMessage "Tumor3 expert training failed: $runName"
        }

        if (-not (Test-Path $tumor3ExpertCkpt)) {
            throw "Tumor3 expert checkpoint not found after training: $tumor3ExpertCkpt"
        }

        if (Test-Path $tumor3CascadeSummary) {
            Write-Host ""
            Write-Host "========== SKIP TUMOR3 CASCADE (summary exists): $runName =========="
            Write-Host $tumor3CascadeSummary
        } else {
            Write-Host ""
            Write-Host "========== RUN TUMOR3 CASCADE: $runName =========="
            $tumor3CascadeCmd = @(
                $mainPy,
                "--run-mode", "cascade",
                "--device", $Device,
                "--main-checkpoint", $mainCkpt,
                "--expert-checkpoint", $tumor3ExpertCkpt,
                "--expert-trigger-topk", $ExpertTriggerTopK.ToString(),
                "--expert-margin-threshold", $marginText,
                "--output-dir", $tumor3CascadeOut
            )
            Invoke-CodexPython -CommandArgs $tumor3CascadeCmd -FailureMessage "Tumor3 cascade failed: $runName"
        }

        foreach ($pair in $expertPairs) {
            $pairTag = $pair.Tag
            $pairClasses = $pair.Classes
            $pairLabel = $pair.Label

            $pairExpertOut = Join-Path $outputsDir ("expert_pair_" + $pairTag + "_" + $runName)
            $pairExpertCkpt = Join-Path $pairExpertOut "best_model.pt"
            $pairCascadeOut = Join-Path $outputsDir ("cascade_pair_" + $pairTag + "_" + $runName)
            $pairCascadeSummary = Join-Path $pairCascadeOut "metrics_summary.json"

            if (Test-Path $pairExpertCkpt) {
                Write-Host ""
                Write-Host "========== SKIP PAIR EXPERT (checkpoint exists): $runName / $pairLabel =========="
                Write-Host $pairExpertCkpt
            } else {
                Write-Host ""
                Write-Host "========== TRAIN PAIR EXPERT: $runName / $pairLabel =========="
                $pairExpertCmd = @(
                    $mainPy,
                    "--run-mode", "expert",
                    "--model-name", $ModelName,
                    "--expert-classes", $pairClasses,
                    "--device", $Device,
                    "--epochs", $Epochs.ToString(),
                    "--batch-size", $BatchSize.ToString(),
                    "--image-size", $ImageSize.ToString(),
                    "--lr", "3e-4",
                    "--weight-decay", "1e-4",
                    "--num-workers", $NumWorkers.ToString()
                ) + $job.Args + @(
                    "--output-dir", $pairExpertOut
                )
                Invoke-CodexPython -CommandArgs $pairExpertCmd -FailureMessage "Pair expert training failed: $runName / $pairLabel"
            }

            if (-not (Test-Path $pairExpertCkpt)) {
                throw "Pair expert checkpoint not found after training: $pairExpertCkpt"
            }

            if (Test-Path $pairCascadeSummary) {
                Write-Host ""
                Write-Host "========== SKIP PAIR CASCADE (summary exists): $runName / $pairLabel =========="
                Write-Host $pairCascadeSummary
            } else {
                Write-Host ""
                Write-Host "========== RUN PAIR CASCADE: $runName / $pairLabel =========="
                $pairCascadeCmd = @(
                    $mainPy,
                    "--run-mode", "cascade",
                    "--device", $Device,
                    "--main-checkpoint", $mainCkpt,
                    "--expert-checkpoint", $pairExpertCkpt,
                    "--expert-trigger-topk", $ExpertTriggerTopK.ToString(),
                    "--expert-margin-threshold", $marginText,
                    "--output-dir", $pairCascadeOut
                )
                Invoke-CodexPython -CommandArgs $pairCascadeCmd -FailureMessage "Pair cascade failed: $runName / $pairLabel"
            }
        }
    }

    Write-Host ""
    Write-Host "All EfficientNet-B1 formal experiments finished."
    Write-Host "Results saved under: $outputsDir"
    Write-Host "Batch log saved to: $logPath"
}
finally {
    Stop-Transcript | Out-Null
}
