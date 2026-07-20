# Progress Report 2026-07-19

## 1. Summary

The repository has been migrated from the user-restored `EfficientNet-B1` baseline to a new `EfficientNet-B2` baseline.

This migration keeps the existing training, expert, and cascade framework, while updating the default backbone and output organization.

## 2. Completed Changes

### 2.1 Backbone Migration

- default `--model-name`: `efficientnet_b2`
- default `--image-size`: `256`
- local pretrained weight loading added for `B2`
- compatibility with historical `B0/B1` checkpoints preserved

### 2.2 Output Layout Migration

All new experiments now write into shared folders under `outputs/`.

Weights:
- `outputs/weights/<experiment_name>/best_model.pt`

Other results:
- `outputs/results/<experiment_name>/metrics_summary.json`
- `outputs/results/<experiment_name>/...csv files...`

This replaces the older per-experiment nested layout.

### 2.3 Batch Script Migration

Created:
- `scripts/run_all_efficientnet_b2_50.ps1`
- `scripts/run_all_efficientnet_b2_50.cmd`

The batch script covers:
- `10` single-model runs
- `10` tumor3 cascade runs
- `30` pairwise cascade runs
- total: `50`

## 3. Current Reference Results

Fully completed historical best single model from `B0`:
- `v3.4_pretrained_focal_ls_cosine_cbam`
- test accuracy: `86.35%`
- macro F1: `0.8646`

Fully completed historical best cascade from `B0`:
- `cascade_v3.4_pretrained_focal_ls_cosine_cbam`
- test accuracy: `87.62%`
- macro F1: `0.8773`

## 4. Current Status

The `B2` code path and documents are ready.

Not yet completed:
- the actual `50` formal `B2` runs
- `B2` vs `B0` result comparison
- post-run docx summaries

## 5. Run Command

If you are in `2026APMCM`:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_all_efficientnet_b2_50.ps1" -PythonExe "..\LCC_GPU\python.exe"
```

or

```powershell
.\scripts\run_all_efficientnet_b2_50.cmd
```

## 6. Next Step

Run the `B2` batch experiments, then generate the new cross-backbone comparison documents.
