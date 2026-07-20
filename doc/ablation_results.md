# Historical Ablation Results

## Scope

This file summarizes the historical completed `B0` ablation line.

That completed line contains:
- `12` original non-B2 experiments
- later cascade extensions documented elsewhere

The current repository default has now been migrated to `B2`, but the `B2` formal rerun has not been executed yet.

## Historical Best B0 Single Model

- run: `v3.4_pretrained_focal_ls_cosine_cbam`
- backbone: `efficientnet_b0`
- configuration:
  - pretrained
  - focal loss
  - label smoothing
  - cosine scheduler
  - CBAM
- test accuracy: `86.35%`
- test macro F1: `0.8646`

## Historical Best B0 Cascade

- run: `cascade_v3.4_pretrained_focal_ls_cosine_cbam`
- main model: `v3.4_pretrained_focal_ls_cosine_cbam`
- expert model: `expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam`
- trigger: `top-k=2`, `margin <= 0.12`
- test accuracy: `87.62%`
- test macro F1: `0.8773`

## Historical Lessons

1. Pretraining produced the largest single improvement.
2. `label smoothing` and `focal loss` improved generalization more reliably than plain weighted CE.
3. Lightweight attention modules remained useful on this task.
4. Expert cascade helped in some confusion-heavy settings, but not every cascade variant beat its paired single model.

## Current B2 Status

The codebase is now prepared for a new `B2` rerun under the same non-CPU, pretrained conditions.

Formal B2 plan:
- `10` single-model runs
- `10` tumor3 cascade runs
- `30` pairwise cascade runs
- total: `50`

Output layout for the new line:
- `outputs/weights/<experiment_name>/`
- `outputs/results/<experiment_name>/`
