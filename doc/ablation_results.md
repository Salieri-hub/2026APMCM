# Ablation Results

## Naming Rule

- Formal experiment folders follow `vX.Y_<change>`.
- `v1.x` is the scratch + CE family.
- `v2.x` is the pretrained + CE family.
- `v3.x` is the pretrained + focal + label smoothing + cosine family.
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
| `v3.4_pretrained_focal_ls_cosine_cbam` | `v3.0_pretrained_focal_ls_cosine` | add CBAM attention | 86.35% | 0.8646 | best overall |

## Interpretation

- `v1.1_scratch_ce_cuda` only changes hardware. Its small gain shows that better speed alone does not solve the classification problem.
- `v2.0_pretrained_ce` is the most important step. Transfer learning gives the model useful visual priors and lifts test accuracy by `36.51` points over the scratch CUDA run.
- `v2.2_pretrained_ce_ls` is much stronger than `v2.1_pretrained_ce_cosine`, so overconfidence is a bigger problem than the raw learning-rate schedule.
- `v2.3_pretrained_ce_ls_cosine` does not beat `v2.2_pretrained_ce_ls`, so label smoothing and cosine do not create a positive stacking effect here.
- `v2.4_pretrained_ce_ls_cosine_weightedce` drops further, which indicates standard balanced weights are too blunt for this dataset.
- `v3.0_pretrained_focal_ls_cosine` helps hard-sample learning, so it beats the CE family with the same backbone settings.
- `v3.1_pretrained_focal_ls_cosine_mixup` and `v3.2_pretrained_focal_ls_cosine_cutmix` both hurt performance. In this small medical-image setting, aggressive mixed augmentation likely destroys subtle lesion cues.
- `v3.3_pretrained_focal_ls_cosine_se` is weaker than expected because `EfficientNet-B0` already contains SE-style channel attention.
- `v3.4_pretrained_focal_ls_cosine_cbam` is the best result. Compared with `v3.3_pretrained_focal_ls_cosine_se`, it improves accuracy by `9.21` points and Macro F1 by `0.0762`, which means spatial attention is more useful than another channel-only recalibration block here.

## Conclusions

- Pretraining is the largest improvement.
- Label smoothing is the strongest low-cost regularizer in the CE family.
- Standard balanced weighted CE, MixUp, and CutMix are not helpful in the current small-sample setting.
- Extra SE is weaker than CBAM and looks redundant on top of `EfficientNet-B0`.
- Best run: `v3.4_pretrained_focal_ls_cosine_cbam`.
