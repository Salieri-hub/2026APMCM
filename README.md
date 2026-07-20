# 2026APMCM Problem 2 Baseline

## Overview

This project targets APMCM Problem 2: four-class lung cancer pathology image classification.

Current default backbone:
- `EfficientNet-B2`
- default image size: `256`
- training modes: `single / expert / cascade`

Historical completed formal results are still the `B0` line. The `B2` line is the current code default and is prepared for a new batch of `50` formal experiments.

## Dataset Task

Main task classes:
- `adenocarcinoma`
- `large.cell.carcinoma`
- `normal`
- `squamous.cell.carcinoma`

Supported experiment modes:
- `single`: four-class main model
- `expert`: subset expert model, usually two-class or three-class
- `cascade`: four-class main model plus expert refinement

## Current Defaults

- `--model-name efficientnet_b2`
- `--image-size 256`
- `--device auto/cpu/cuda`
- `--loss cross_entropy` or `focal`
- `--scheduler none/cosine/plateau`
- `--feature-attention none/se/cbam`
- `--class-weighting none/balanced/manual`
- `--mixup-alpha` and `--cutmix-alpha`

## Output Layout

All experiment artifacts are now stored under two shared folders inside `outputs/`:

- `outputs/weights/<experiment_name>/best_model.pt`
- `outputs/results/<experiment_name>/metrics_summary.json`
- `outputs/results/<experiment_name>/test_predictions.csv`
- `outputs/results/<experiment_name>/valid_confusion_matrix.csv`
- `outputs/results/<experiment_name>/test_confusion_matrix.csv`
- cascade runs also add:
  - `valid_cascade_predictions.csv`
  - `test_cascade_predictions.csv`

Important:
- you still pass `--output-dir .\outputs\<experiment_name>`
- the code automatically routes weights into `outputs/weights/<experiment_name>/`
- the code automatically routes reports into `outputs/results/<experiment_name>/`

## Quick Start

If you are in `2026APMCM`:

```powershell
..\LCC_GPU\python.exe .\src\main.py --device cuda --pretrained --model-name efficientnet_b2
```

Example main + expert + cascade:

```powershell
..\LCC_GPU\python.exe .\src\main.py --device cuda --pretrained --model-name efficientnet_b2 --loss focal --label-smoothing 0.1 --scheduler cosine --feature-attention cbam --output-dir .\outputs\v3.4_pretrained_focal_ls_cosine_cbam_b2
..\LCC_GPU\python.exe .\src\main.py --run-mode expert --model-name efficientnet_b2 --expert-classes adenocarcinoma,large.cell.carcinoma,squamous.cell.carcinoma --device cuda --pretrained --loss focal --label-smoothing 0.1 --scheduler cosine --feature-attention cbam --output-dir .\outputs\expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam_b2
..\LCC_GPU\python.exe .\src\main.py --run-mode cascade --device cuda --main-checkpoint .\outputs\weights\v3.4_pretrained_focal_ls_cosine_cbam_b2\best_model.pt --expert-checkpoint .\outputs\weights\expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam_b2\best_model.pt --output-dir .\outputs\cascade_v3.4_pretrained_focal_ls_cosine_cbam_b2
```

## Batch Run

Script files:
- `scripts/run_all_efficientnet_b2_50.ps1`
- `scripts/run_all_efficientnet_b2_50.cmd`

If you are in `2026APMCM`:

```powershell
.\scripts\run_all_efficientnet_b2_50.cmd
```

or

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_all_efficientnet_b2_50.ps1" -PythonExe "..\LCC_GPU\python.exe"
```

The batch script produces:
- `10` single-model runs
- `10` tumor3 cascade runs
- `30` pairwise cascade runs
- total formal outputs: `50`

## Historical Best Verified Results

Best historical single model from completed `B0` experiments:
- run: `v3.4_pretrained_focal_ls_cosine_cbam`
- test accuracy: `86.35%`
- test macro F1: `0.8646`

Best historical cascade from completed `B0` experiments:
- run: `cascade_v3.4_pretrained_focal_ls_cosine_cbam`
- test accuracy: `87.62%`
- test macro F1: `0.8773`

## Documents

- `doc/problem2_baseline.md`
- `doc/literature_review.md`
- `doc/ablation_results.md`
- `doc/progress_report_2026-07-19.md`
- `AI_CONTEXT.md`
- `TODO.md`
