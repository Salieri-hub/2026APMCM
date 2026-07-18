# Ablation Results

## Setup

All runs use the same dataset split, `EfficientNet-B0`, `batch_size=16`, `image_size=224`, `lr=3e-4`, `weight_decay=1e-4`, and `seed=42`.

## Summary

| Run | Key change | Acc | Macro F1 | Note |
| --- | --- | ---: | ---: | --- |
| `ablation_scratch_ce` | scratch CE | 39.68% | 0.4427 | legacy CPU baseline |
| `ablation_pretrained_ce` | + pretrained | 76.19% | 0.7721 | biggest single gain |
| `ablation_pretrained_ce_ls_cosine` | + label smoothing + cosine | 77.46% | 0.7894 | historical best CE run |
| `ablation_pretrained_ce_ls_cosine_weightedce` | + balanced weights | 73.33% | 0.7547 | hurts overall |
| `ablation_pretrained_focal_ls_cosine` | + focal loss | 79.37% | 0.7988 | historical best before ablations |
| `ablation_scratch_ce_cuda` | scratch CE on CUDA | 40.63% | 0.4488 | same idea as legacy baseline |
| `ablation_pretrained_ce_ls` | + label smoothing only | 83.81% | 0.8416 | best simple regularizer |
| `ablation_pretrained_ce_cosine` | + cosine only | 77.14% | 0.7957 | mild gain |
| `ablation_pretrained_focal_ls_cosine_mixup` | + MixUp | 71.75% | 0.7320 | too aggressive here |
| `ablation_pretrained_focal_ls_cosine_cutmix` | + CutMix | 73.97% | 0.7546 | also degrades |
| `ablation_pretrained_focal_ls_cosine_se` | + extra SE | 77.14% | 0.7884 | weak improvement |
| `ablation_pretrained_focal_ls_cosine_cbam` | + CBAM | 86.35% | 0.8646 | best overall |

## Attention Analysis

- `ablation_pretrained_focal_ls_cosine_se` reaches `77.14%` accuracy and `0.7884` macro F1. It does not beat the focal baseline, so the extra SE block looks mostly redundant on top of `EfficientNet-B0`.
- `ablation_pretrained_focal_ls_cosine_cbam` reaches `86.35%` accuracy and `0.8646` macro F1. Compared with SE, it improves accuracy by `9.21` points and macro F1 by `0.0762`.
- Per-class tradeoff: CBAM greatly lifts `adenocarcinoma` recall, keeps `large.cell.carcinoma` and `normal` strong, but lowers `squamous.cell.carcinoma` recall.

## Conclusions

- Pretraining is the largest improvement.
- Label smoothing is the strongest low-cost regularizer here.
- `balanced` class weighting, MixUp, and CutMix are not helpful in this small-sample setting.
- Extra SE is weaker than CBAM and likely redundant on top of `EfficientNet-B0`.
- Best run: `ablation_pretrained_focal_ls_cosine_cbam`.
