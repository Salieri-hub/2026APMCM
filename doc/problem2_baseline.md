# Problem 2 Baseline Plan

## 1. Task Definition

This project solves a four-class lung pathology image classification task for APMCM Problem 2.

Classes:
- `adenocarcinoma`
- `large.cell.carcinoma`
- `normal`
- `squamous.cell.carcinoma`

The main evaluation target is the full four-class task. Expert models are auxiliary subset classifiers used in cascade mode.

## 2. Current Baseline Line

The current code baseline has been migrated to:
- backbone: `EfficientNet-B2`
- default image size: `256`
- optimizer: `AdamW`
- losses: `cross_entropy` and `focal`
- schedulers: `none`, `cosine`, `plateau`
- optional additions: `label smoothing`, `MixUp`, `CutMix`, `SE`, `CBAM`

## 3. Supported Run Modes

### 3.1 Single

Four-class main model training and evaluation.

### 3.2 Expert

Subset expert model training.

Supported subset types:
- tumor3 expert:
  - `adenocarcinoma,large.cell.carcinoma,squamous.cell.carcinoma`
- pair experts:
  - `adenocarcinoma,large.cell.carcinoma`
  - `adenocarcinoma,squamous.cell.carcinoma`
  - `large.cell.carcinoma,squamous.cell.carcinoma`

### 3.3 Cascade

Main model predicts first. The expert branch is invoked only when:
1. the main model top-k classes all stay inside the expert class subset
2. `top1 - top2 <= expert_margin_threshold`

The expert branch does not hard override the main model. It refines probability mass inside the expert subset and then returns a final four-class decision.

## 4. Formal B2 Experiment Plan

This round keeps the same formal experiment conditions as the previous non-CPU, pretrained line, and reruns them on `B2`.

Excluded:
- `v1.0_scratch_ce_cpu`
- `v1.1_scratch_ce_cuda`

Formal outputs to produce:
- `10` single-model runs
- `10` tumor3 cascade runs
- `30` pairwise cascade runs
- total: `50`

## 5. Output Layout

All artifacts are stored in shared folders under `outputs/`.

Weights:
- `outputs/weights/<experiment_name>/best_model.pt`

Other results:
- `outputs/results/<experiment_name>/metrics_summary.json`
- `outputs/results/<experiment_name>/test_predictions.csv`
- `outputs/results/<experiment_name>/valid_confusion_matrix.csv`
- `outputs/results/<experiment_name>/test_confusion_matrix.csv`
- cascade runs also save:
  - `valid_cascade_predictions.csv`
  - `test_cascade_predictions.csv`

The command line still uses:
- `--output-dir .\outputs\<experiment_name>`

The code automatically maps that experiment name into the shared `weights/` and `results/` trees.

## 6. Commands

If you are in `2026APMCM`:

```powershell
.\scripts\run_all_efficientnet_b2_50.cmd
```

or

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_all_efficientnet_b2_50.ps1" -PythonExe "..\LCC_GPU\python.exe"
```

Manual cascade example:

```powershell
..\LCC_GPU\python.exe .\src\main.py --run-mode cascade --device cuda --main-checkpoint .\outputs\weights\v3.4_pretrained_focal_ls_cosine_cbam_b2\best_model.pt --expert-checkpoint .\outputs\weights\expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam_b2\best_model.pt --output-dir .\outputs\cascade_v3.4_pretrained_focal_ls_cosine_cbam_b2
```

## 7. Historical Reference Results

Completed historical best single model from `B0`:
- `v3.4_pretrained_focal_ls_cosine_cbam`
- test accuracy: `86.35%`
- macro F1: `0.8646`

Completed historical best cascade from `B0`:
- `cascade_v3.4_pretrained_focal_ls_cosine_cbam`
- test accuracy: `87.62%`
- macro F1: `0.8773`

The current `B2` line is prepared but not yet formally rerun.
