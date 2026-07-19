# APMCM 2026 B题阶段进度汇报

截至 `2026-07-19`，本项目已完成问题一的数据统计框架、问题二 baseline 搭建、GPU 环境验证，以及 `12` 组正式实验和 `1` 组 smoke test。当前工作重点已经从“先把模型跑通”转向“围绕当前最优结构继续提升泛化能力，并整理可直接用于论文与汇报的材料”。

## 1. 当前整体进展

- 问题一：题意分析、数据规模统计和准确率公式计算已完成，后续需要整理成正式论文式表述。
- 问题二：已完成 `EfficientNet-B0` 四分类 baseline，实现 `CUDA/CPU` 自动切换、AMP、学习率调度、`Focal Loss`、类别加权、`MixUp/CutMix`、`SE/CBAM` 等增量实验能力。
- 环境方面：`..\LCC_GPU` 已验证可正常进行 CUDA 训练，当前主机可识别 `NVIDIA GeForce RTX 4060 Laptop GPU`。
- 当前阶段判断：已经有可复现、可比较的稳定 baseline，但还不是最终提交方案，后续仍要围绕误分类边界继续优化。

## 2. 数据与 baseline 情况

当前数据集划分如下：

| 划分 | 样本数 |
| --- | ---: |
| 训练集 | 613 |
| 验证集 | 72 |
| 测试集 | 315 |

四个类别分别为：

- `adenocarcinoma`
- `large.cell.carcinoma`
- `normal`
- `squamous.cell.carcinoma`

当前 baseline 主体配置为：

- 主干网络：`EfficientNet-B0`
- 优化器：`AdamW`
- 默认轮数：`25`
- 输入尺寸：`224 x 224`
- 可选损失：`CrossEntropyLoss / Focal Loss`
- 可选策略：`pretrained`、`label smoothing`、`cosine scheduler`、类别加权、`MixUp`、`CutMix`、`SE`、`CBAM`

## 3. 实验命名规则

- 正式实验目录统一采用 `vX.Y_<change>` 规则。
- `v1.x` 表示 scratch + CE 系列。
- `v2.x` 表示 pretrained + CE 系列。
- `v3.x` 表示 pretrained + focal + label smoothing + cosine 系列。
- `ablation_smoke_mixup_cbam` 仅用于流程验证，不纳入正式比较。

## 4. 12组正式实验结果汇总

| 实验目录 | 基于哪组 | 新增改动 | 验证集准确率 | 测试集准确率 | Macro F1 |
| --- | --- | --- | ---: | ---: | ---: |
| `v1.0_scratch_ce_cpu` | 无 | scratch + CE + CPU | 62.50% | 39.68% | 0.4427 |
| `v1.1_scratch_ce_cuda` | `v1.0_scratch_ce_cpu` | 仅改为 CUDA | 69.44% | 40.63% | 0.4488 |
| `v2.0_pretrained_ce` | `v1.1_scratch_ce_cuda` | 开启 pretrained | 91.67% | 76.19% | 0.7721 |
| `v2.1_pretrained_ce_cosine` | `v2.0_pretrained_ce` | 加 cosine scheduler | 90.28% | 77.14% | 0.7957 |
| `v2.2_pretrained_ce_ls` | `v2.0_pretrained_ce` | 加 label smoothing | 97.22% | 83.81% | 0.8416 |
| `v2.3_pretrained_ce_ls_cosine` | `v2.2_pretrained_ce_ls` | 在 `v2.2` 上再加 cosine | 90.28% | 77.46% | 0.7894 |
| `v2.4_pretrained_ce_ls_cosine_weightedce` | `v2.3_pretrained_ce_ls_cosine` | 再加 balanced weighted CE | 87.50% | 73.33% | 0.7547 |
| `v3.0_pretrained_focal_ls_cosine` | `v2.3_pretrained_ce_ls_cosine` | 将 CE 改为 focal loss | 90.28% | 79.37% | 0.7988 |
| `v3.1_pretrained_focal_ls_cosine_mixup` | `v3.0_pretrained_focal_ls_cosine` | 加 MixUp | 86.11% | 71.75% | 0.7320 |
| `v3.2_pretrained_focal_ls_cosine_cutmix` | `v3.0_pretrained_focal_ls_cosine` | 加 CutMix | 91.67% | 73.97% | 0.7546 |
| `v3.3_pretrained_focal_ls_cosine_se` | `v3.0_pretrained_focal_ls_cosine` | 加额外 SE 模块 | 93.06% | 77.14% | 0.7884 |
| `v3.4_pretrained_focal_ls_cosine_cbam` | `v3.0_pretrained_focal_ls_cosine` | 加 CBAM 模块 | 93.06% | 86.35% | 0.8646 |

## 5. 阶段性结论

- `pretrained` 是最关键的一步，测试集准确率相对 `v1.1_scratch_ce_cuda` 从 `40.63%` 提升到 `76.19%`。
- 在 CE 系列中，`label smoothing` 的收益明显强于单独加入 `cosine scheduler`，说明当前问题更偏向过拟合与过度自信，而不是单纯学习率退火不足。
- `v2.3_pretrained_ce_ls_cosine` 没有超过 `v2.2_pretrained_ce_ls`, 说明这两个改动在当前数据集上没有形成稳定叠加收益。
- `v2.4_pretrained_ce_ls_cosine_weightedce` 明显下降，说明标准 `balanced weighted CE` 在当前场景下干预过强。
- `v3.0_pretrained_focal_ls_cosine` 说明 focal loss 对难分类样本有一定帮助，但还不是最佳方案。
- `v3.1_pretrained_focal_ls_cosine_mixup` 和 `v3.2_pretrained_focal_ls_cosine_cutmix` 均下降，说明强混合增强会破坏当前小样本医学图像中的关键纹理信息。
- `v3.3_pretrained_focal_ls_cosine_se` 收益有限，而 `v3.4_pretrained_focal_ls_cosine_cbam` 明显优于它，说明空间注意力比额外通道重标定更有效。
- 当前最优结果为 `v3.4_pretrained_focal_ls_cosine_cbam`，测试集准确率 `86.35%`，`Macro F1 = 0.8646`。

## 6. 当前最优实验的重点解读

`v3.4_pretrained_focal_ls_cosine_cbam` 的核心配置是：

```text
pretrained + focal loss(gamma=2) + label smoothing(0.1) + cosine scheduler + CBAM
```

该实验的各类测试集召回率为：

- `adenocarcinoma`：`90.83%`
- `large.cell.carcinoma`：`90.20%`
- `normal`：`98.15%`
- `squamous.cell.carcinoma`：`71.11%`

这说明当前模型整体指标已经较强，但鳞癌召回率仍然偏低，仍然是下一阶段最需要解决的问题。

## 7. 下一步计划

- 继续围绕 `v3.4_pretrained_focal_ls_cosine_cbam` 尝试采样策略或手动类别权重，优先拉回鳞癌召回率。
- 重点验证 `Focal Loss + 手动权重` 是否能在保住腺癌和大细胞癌收益的同时，减少鳞癌漏判。
- 开展误差分析，重点检查“鳞癌 -> 腺癌 / 大细胞癌”的高频误判样本。
- 将问题一统计结论与问题二实验结果整理成论文可直接使用的正式文字。

## 8. 口头汇报压缩版

1. 目前我已经完成了问题二 baseline、GPU 环境和 `12` 组正式对比实验，项目已经从“先跑通模型”进入“围绕最优结构继续优化”的阶段。
2. 这一轮实验里最有效的三个改动是 `pretrained`、`label smoothing` 和 `CBAM`，其中迁移学习带来的提升最大。
3. 当前最优实验目录是 `v3.4_pretrained_focal_ls_cosine_cbam`，测试集准确率达到 `86.35%`，`Macro F1` 达到 `0.8646`。
4. 下一步我会重点解决鳞癌召回率偏低的问题，继续尝试采样策略、手动权重和误差分析。
