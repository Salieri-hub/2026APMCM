# 问题二分析与 Baseline 方案

## 1. 题目二需要完成什么

问题二要求围绕四分类肺部 CT 影像识别任务完成三部分内容：

1. 说明训练集、验证集、测试集在模型训练流程中的作用。
2. 给出分类准确率公式，并计算测试集 `315` 张中正确识别 `279` 张时的准确率。
3. 搭建一个可运行的肺癌影像分类模型，并讨论如何降低腺癌与鳞状细胞癌的误诊率。

## 2. 数据集作用

- 训练集：用于更新模型参数，让模型学习四类图像的判别特征。
- 验证集：用于调参、比较不同模型配置，并作为保存最佳模型或选定最佳级联方案的依据。
- 测试集：只在模型训练结束后使用，用于评估模型的泛化能力，不能参与训练和调参。

## 3. 准确率公式

分类准确率定义为：

```text
Accuracy = 正确分类样本数 / 总样本数 × 100%
```

当测试集总数为 `315`，正确识别 `279` 张时：

```text
Accuracy = 279 / 315 × 100% = 88.57%
```

## 4. 当前实现的实验管线

当前项目不再只是“单个 baseline 模型”，而是一个可复用的实验管线，入口统一为 `src/main.py`。主要支持三种模式：

1. `single`
   - 训练和评估完整四分类主模型。
2. `expert`
   - 只针对指定类别子集训练专家模型，可做二分类或三分类。
3. `cascade`
   - 先运行四分类主模型，再按触发规则调用专家模型，对局部高混淆类别做二次判别，最终仍输出四分类结果。

当前默认 backbone 为 `EfficientNet-B0`，主模型和专家模型都沿用这一主干，只是最终分类头类别数不同。

## 5. 数据与标签

默认读取的数据目录为：

```text
../附件/Data/train
../附件/Data/valid
../附件/Data/test
```

脚本会自动处理目录命名不完全一致的问题，并统一映射到以下四类：

1. `adenocarcinoma`
2. `large.cell.carcinoma`
3. `normal`
4. `squamous.cell.carcinoma`

当前样本规模：

- 训练集：`613`
- 验证集：`72`
- 测试集：`315`

## 6. 当前主模型基线配置

当前主线 baseline 配置为：

- 框架：PyTorch + timm
- 主干网络：`EfficientNet-B0`
- 优化器：`AdamW`
- 默认训练轮数：`25`
- 输入尺寸：`224 x 224`
- 损失函数：`CrossEntropyLoss / Focal Loss`
- 学习率调度：`none / cosine / plateau`
- 可选正则：`label smoothing`
- 可选数据增强：`MixUp / CutMix`
- 可选结构增强：`SE / CBAM`
- 设备策略：`--device auto`，有可用 CUDA 时优先使用 GPU

## 7. 级联模式原理

当前 `cascade` 模式采用“主模型先判，再按条件触发专家模型”的方式：

1. 主模型先输出四类概率。
2. 如果主模型 `top-k` 预测全部落在专家模型负责的类别子集内，则该样本具备触发资格。
3. 若主模型 `top1 - top2` 的置信差值不大于阈值 `--expert-margin-threshold`，说明该样本在局部边界上较难分，此时触发专家模型。
4. 专家模型只在自己负责的类别子集内重新细分概率，最后再和主模型概率融合，输出最终四分类结果。

当前默认触发参数为：

- `expert_trigger_topk = 2`
- `expert_margin_threshold = 0.12`

## 8. 已完成的正式实验范围

截至 `2026-07-19`，项目已经完成三组正式实验：

1. `12` 组单模型实验
2. `10` 组三肿瘤专家级联实验
3. `30` 组两两肿瘤专家级联实验

合计 `52` 组正式结果。

单模型目录命名规则：

- `v1.x`：scratch + CE
- `v2.x`：pretrained + CE
- `v3.x`：pretrained + focal / attention

级联目录命名规则：

- `expert_tumor3_*`：三肿瘤专家模型
- `expert_pair_*`：两两肿瘤专家模型
- `cascade_*`：三肿瘤专家级联结果
- `cascade_pair_*`：两两肿瘤专家级联结果

## 9. 当前最佳单模型结果

当前最佳单模型为：

```text
v3.4_pretrained_focal_ls_cosine_cbam
```

其核心配置为：

```text
pretrained + focal loss(gamma=2) + label smoothing(0.1) + cosine scheduler + CBAM
```

对应结果：

- 最佳验证集准确率：`93.06%`
- 测试集准确率：`86.35%`
- 测试集 `macro F1`：`0.8646`
- 测试集 `weighted F1`：`0.8647`

测试集各类别召回率：

- `adenocarcinoma`：`90.83%`
- `large.cell.carcinoma`：`90.20%`
- `normal`：`98.15%`
- `squamous.cell.carcinoma`：`71.11%`

## 10. 当前最佳全局结果

当前 `52` 组实验中的全局最佳结果为：

```text
cascade_v3.4_pretrained_focal_ls_cosine_cbam
```

其组成方式为：

- 四分类主模型：`v3.4_pretrained_focal_ls_cosine_cbam`
- 三肿瘤专家模型：`expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam`
- 触发策略：`top-k=2`，`margin <= 0.12`

对应结果：

- 测试集准确率：`87.62%`
- 测试集 `macro F1`：`0.8773`

相对于最佳单模型：

- 测试集准确率提升 `1.27` 个百分点
- 测试集 `macro F1` 提升约 `0.0127`

测试集级联统计：

- `expert_invocations = 17`
- `expert_changed_predictions = 8`
- `expert_corrected_predictions = 5`
- `expert_hurt_predictions = 1`

这说明当前级联机制并不是“大面积重写主模型输出”，而是在少量高混淆样本上做针对性修正。

## 11. 消融实验结论

从 `12` 组单模型结果看：

1. `pretrained` 是收益最大的单步改动。
2. `label smoothing` 是 CE 系列里最有效的低成本正则项。
3. `balanced class-weighted CE` 在当前数据上整体不如不加权。
4. `MixUp` 和 `CutMix` 在当前小样本医学图像设定下会破坏细粒度病灶线索。
5. `CBAM` 明显优于额外 `SE`，是当前最佳结构增量。

从 `40` 组级联结果看：

1. 三肿瘤专家级联整体更稳，`10` 组里有 `8` 组优于对应单模型。
2. 两两肿瘤专家级联中，`large.cell.carcinoma` 与 `squamous.cell.carcinoma` 这一支平均效果最好。
3. 不是所有级联都优于主模型，说明触发规则和专家模型质量仍然是性能上限的重要决定因素。

## 12. 相关论文借鉴

项目参考的论文位于上级目录 `..\相关论文(1)`。当前已经落地并被实验部分验证的思路主要包括：

1. 迁移学习：`pretrained` 已验证为当前最关键改进。
2. `EfficientNet` 路线：当前所有正式实验统一使用 `EfficientNet-B0` 作为 backbone。
3. 轻量注意力机制：`CBAM` 已证明优于额外 `SE`。
4. 难例聚焦与边界平滑：`Focal Loss`、`label smoothing` 已被纳入正式实验并保留为当前主线。

## 13. 后续优化方向

1. 在当前最佳 `v3.4` 主模型和三肿瘤专家模型上微调触发阈值。
2. 对最佳单模型和最佳级联模型补充误差分析与高混淆样本检查。
3. 若后续还需继续提高结果，优先优化专家触发逻辑和误差修正机制，而不是立刻大规模更换 backbone。

## 14. 相关文件

- `doc/ablation_results.md`：`12` 组单模型消融总结
- `doc/progress_report_2026-07-19.md`：阶段进展汇总
- `doc/all_52_experiments_comparison_20260719.docx`：`52` 组总对比 Word 文档
- `doc/outputs_实验版本说明与消融对比_20260719.docx`：各版本与消融分析 Word 文档
