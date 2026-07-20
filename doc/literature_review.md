# Literature Review Notes

## Source Folder

Use the local paper folder:
- `..\相关论文(1)`

## Practical Takeaways Used by This Project

### 1. Transfer Learning

The literature consistently supports using pretrained ImageNet backbones for small pathology datasets.

Project use:
- `--pretrained`
- current default backbone family: `EfficientNet`

### 2. EfficientNet Family

The literature supports lightweight-to-medium EfficientNet backbones as strong baselines for histopathology image classification.

Project evolution:
- historical completed line: `EfficientNet-B0`
- reverted user baseline before this turn: `EfficientNet-B1`
- current default line after this update: `EfficientNet-B2`

### 3. Loss and Regularization

Common useful strategies from related papers:
- `label smoothing`
- `focal loss`
- class reweighting in some settings
- stronger augmentation such as `MixUp` and `CutMix`

Project support:
- `cross_entropy`
- `focal`
- `label smoothing`
- `balanced/manual` class weighting
- `MixUp`
- `CutMix`

### 4. Learning Rate Scheduling

The literature generally favors nontrivial learning-rate schedules over fixed learning rates.

Project support:
- `none`
- `cosine`
- `plateau`

### 5. Attention Modules

Related papers often use lightweight attention modules to refine discriminative local features.

Project support:
- `SE`
- `CBAM`

### 6. Expert or Cascade Logic

For classes with strong visual confusion, a secondary specialist classifier is a reasonable design.

Project support:
- tumor3 expert branch
- pairwise expert branches
- cascade triggering based on top-k containment and margin threshold

## Current Engineering Decision

The current code path keeps the methodology aligned with the literature while maintaining a controlled engineering ablation setup:
- same task definition
- same main training pipeline
- same expert and cascade framework
- backbone upgraded to `EfficientNet-B2`

## Output Convention

All new B2 experiments use:
- `outputs/weights/<experiment_name>/`
- `outputs/results/<experiment_name>/`
