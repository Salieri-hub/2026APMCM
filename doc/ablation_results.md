# Ablation Results

## Scope

This file summarizes the completed `12` formal single-model runs on the historical `EfficientNet-B0` line only.

- It does not include the later `10` three-tumor cascade runs.
- It does not include the later `30` pairwise cascade runs.
- It does not include the new pending `EfficientNet-B1` rerun.
- For the full completed B0 `52`-run comparison, use the Word summaries in `doc/`.

## Naming Rule

- Historical B0 single-model folders follow `vX.Y_<change>`.
- The new B1 rerun keeps the same base experiment names and appends `_b1`.
- `v1.x` is the scratch + CE family.
- `v2.x` is the pretrained + CE family.
- `v3.x` is the pretrained + focal + attention family.
- `ablation_smoke_mixup_cbam` is only a smoke test and is excluded from the formal comparison.

## Summary

| Run folder | Based on | What changed | Acc | Macro F1 | Note |
| --- | --- | --- | ---: | ---: | --- |
| `v1.0_scratch_ce_cpu` | none | initial scratch CE CPU baseline | 39.68% | 0.4427 | starting point |
| `v1.1_scratch_ce_cuda` | `v1.0_scratch_ce_cpu` | switch CPU to CUDA only | 40.63% | 0.4488 | hardware check only |
| `v2.0_pretrained_ce` | `v1.1_scratch_ce_cuda` | enable pretrained weights | 76.19% | 0.7721 | biggest single gain |
| `v2.1_pretrained_ce_cosine` | `v2.0_pretrained_ce` | add cosine scheduler | 77.14% | 0.7957 | mild gain |
| `v2.2_pretrained_ce_ls` | `v2.0_pretrained_ce` | add label smoothing | 83.81% | 0.8416 | best CE regularizer |
| `v2.3_pretrained_ce_ls_cosine` | `v2.2_pretrained_ce_ls` | add cosine on top of label smoothing | 77.46% | 0.7894 | stacking did not help |
| `v2.4_pretrained_ce_ls_cosine_weightedce` | `v2.3_pretrained_ce_ls_cosine` | add balanced weighted CE | 73.33% | 0.7547 | over-corrects class balance |
| `v3.0_pretrained_focal_ls_cosine` | `v2.3_pretrained_ce_ls_cosine` | replace CE with focal loss | 79.37% | 0.7988 | better hard-sample focus |
| `v3.1_pretrained_focal_ls_cosine_mixup` | `v3.0_pretrained_focal_ls_cosine` | add MixUp | 71.75% | 0.7320 | too aggressive here |
| `v3.2_pretrained_focal_ls_cosine_cutmix` | `v3.0_pretrained_focal_ls_cosine` | add CutMix | 73.97% | 0.7546 | also degrades |
| `v3.3_pretrained_focal_ls_cosine_se` | `v3.0_pretrained_focal_ls_cosine` | add extra SE block | 77.14% | 0.7884 | weak structural gain |
| `v3.4_pretrained_focal_ls_cosine_cbam` | `v3.0_pretrained_focal_ls_cosine` | add CBAM attention | 86.35% | 0.8646 | best completed single model |

## Interpretation

- `v1.1_scratch_ce_cuda` only changes hardware. Its small gain shows that speed alone does not solve the classification problem.
- `v2.0_pretrained_ce` is the most important step. Transfer learning lifts test accuracy by `36.51` points over the scratch CUDA run.
- `v2.2_pretrained_ce_ls` is much stronger than `v2.1_pretrained_ce_cosine`, so overconfidence is a bigger problem than the raw learning-rate schedule.
- `v2.3_pretrained_ce_ls_cosine` does not beat `v2.2_pretrained_ce_ls`, so label smoothing and cosine do not create a stable stacking gain here.
- `v2.4_pretrained_ce_ls_cosine_weightedce` drops further, which indicates standard balanced weights are too blunt for this dataset.
- `v3.0_pretrained_focal_ls_cosine` helps hard-sample learning, but is still not the best completed single-model configuration.
- `v3.1_pretrained_focal_ls_cosine_mixup` and `v3.2_pretrained_focal_ls_cosine_cutmix` both hurt performance. In this small medical-image setting, aggressive mixed augmentation likely destroys subtle lesion cues.
- `v3.3_pretrained_focal_ls_cosine_se` is weaker than expected because `EfficientNet-B0` already contains internal SE-style channel attention.
- `v3.4_pretrained_focal_ls_cosine_cbam` is the best completed single-model result.

## Conclusions

- Pretraining is the largest improvement.
- Label smoothing is the strongest low-cost regularizer in the CE family.
- Standard balanced weighted CE, MixUp, and CutMix are not helpful in the current small-sample setting.
- Extra SE is weaker than CBAM and looks redundant on top of `EfficientNet-B0`.
- Best completed single-model run: `v3.4_pretrained_focal_ls_cosine_cbam`.
- Best completed overall B0 result is no longer in this file; after adding expert cascade, the completed global best becomes `cascade_v3.4_pretrained_focal_ls_cosine_cbam` with `87.62%` test accuracy and `0.8773` Macro F1.
- The next step is the B1 rerun, which will reuse the same formal experiment definitions with `_b1` suffixes.
