# AI Context

## Current Position

The repository code path has been migrated to `EfficientNet-B2`.

Current default behavior:
- backbone: `efficientnet_b2`
- default image size: `256`
- supported modes: `single / expert / cascade`
- local pretrained cache is enabled
- output layout is split into shared `weights/` and `results/` trees

## Output Convention

All new experiments must follow this layout:
- `outputs/weights/<experiment_name>/best_model.pt`
- `outputs/results/<experiment_name>/metrics_summary.json`
- `outputs/results/<experiment_name>/...other csv files...`

Do not use the old flat layout `outputs/<experiment_name>/weights/...` anymore.

## Historical Completed Results

The fully completed and verified formal experiments are still the historical `B0` line.

Best verified single model:
- run: `v3.4_pretrained_focal_ls_cosine_cbam`
- backbone: `efficientnet_b0`
- test accuracy: `86.35%`
- test macro F1: `0.8646`

Best verified cascade:
- run: `cascade_v3.4_pretrained_focal_ls_cosine_cbam`
- main model: `v3.4_pretrained_focal_ls_cosine_cbam`
- expert model: `expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam`
- trigger: `top-k=2`, `margin <= 0.12`
- test accuracy: `87.62%`
- test macro F1: `0.8773`

## Current B2 Goal

Run `50` formal `B2` experiments, excluding:
- `v1.0_scratch_ce_cpu`
- `v1.1_scratch_ce_cuda`

The `50` formal outputs are:
- `10` single-model runs
- `10` tumor3 cascade runs
- `30` pairwise cascade runs

Related scripts:
- `scripts/run_all_efficientnet_b2_50.ps1`
- `scripts/run_all_efficientnet_b2_50.cmd`

## Model Logic

Main model:
- four-class classifier
- `EfficientNet-B2` backbone

Expert model:
- two-class or three-class subset classifier
- same backbone family

Cascade trigger logic:
1. main model predicts first
2. expert can only trigger if main top-k classes all stay inside the expert subset
3. expert also requires `top1 - top2 <= expert_margin_threshold`
4. expert output does not hard-replace the main model; it redistributes probability mass inside the expert subset

## Literature Path

When writing formal documentation, use the paper folder path:
- `..\相关论文(1)`

## Next Actions

1. Run the `B2` batch script.
2. Compare `B2` against completed `B0` results.
3. If `B2` wins, update Word summaries and thesis-style experiment analysis.
