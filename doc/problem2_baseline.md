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

## 5. 当前 backbone 状态

需要区分“历史结果 backbone”和“当前代码默认 backbone”：

1. 历史正式结果 backbone
   - `EfficientNet-B0`
   - 已完成 `52` 组正式实验
2. 当前代码默认 backbone
   - `EfficientNet-B1`
   - 默认输入尺寸 `240`
   - 准备重新执行除 CPU baseline 与非迁移 GPU baseline 之外的其余 `50` 组正式实验

因此，当前项目不是删掉了 B0，而是保留 B0 已完成结果，同时把新的默认主线切到 B1。

## 6. 数据与标签

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

## 7. 当前代码默认主线配置

当前默认主线配置为：

- 框架：PyTorch + timm
- 默认主干网络：`EfficientNet-B1`
- 默认输入尺寸：`240 x 240`
- 优化器：`AdamW`
- 默认训练轮数：`25`
- 损失函数：`CrossEntropyLoss / Focal Loss`
- 学习率调度：`none / cosine / plateau`
- 可选正则：`label smoothing`
- 可选数据增强：`MixUp / CutMix`
- 可选结构增强：`SE / CBAM`
- 设备策略：`--device auto`，有可用 CUDA 时优先使用 GPU

补充说明：

- 如果显式指定 `--model-name efficientnet_b0`，默认输入尺寸会回到 `224`
- 当前 `main.py` 对旧 B0 checkpoint 仍然兼容

## 8. 级联模式原理

当前 `cascade` 模式采用“主模型先判，再按条件触发专家模型”的方式：

1. 主模型先输出四类概率。
2. 如果主模型 `top-k` 预测全部落在专家模型负责的类别子集内，则该样本具备触发资格。
3. 若主模型 `top1 - top2` 的置信差值不大于阈值 `--expert-margin-threshold`，说明该样本在局部边界上较难分，此时触发专家模型。
4. 专家模型只在自己负责的类别子集内重新细分概率，最后再和主模型概率融合，输出最终四分类结果。

当前默认触发参数为：

- `expert_trigger_topk = 2`
- `expert_margin_threshold = 0.12`

## 9. 已完成的历史正式实验范围

截至 `2026-07-19`，项目已经完成三组历史正式实验：

1. `12` 组单模型实验
2. `10` 组三肿瘤专家级联实验
3. `30` 组两两肿瘤专家级联实验

合计 `52` 组历史正式结果，统一基于 `EfficientNet-B0`。

## 10. 当前已验证最佳单模型结果

当前“已完成并验证”的最佳单模型为：

```text
v3.4_pretrained_focal_ls_cosine_cbam
```

其核心配置为：

```text
EfficientNet-B0 + pretrained + focal loss(gamma=2) + label smoothing(0.1) + cosine scheduler + CBAM
```

对应结果：

- 最佳验证集准确率：`93.06%`
- 测试集准确率：`86.35%`
- 测试集 `macro F1`：`0.8646`
- 测试集 `weighted F1`：`0.8647`

## 11. 当前已验证最佳全局结果

当前已完成的 `52` 组历史实验中的全局最佳结果为：

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

测试集级联统计：

- `expert_invocations = 17`
- `expert_changed_predictions = 8`
- `expert_corrected_predictions = 5`
- `expert_hurt_predictions = 1`

## 12. 新一轮 B1 正式实验范围

当前计划重新执行的 B1 正式实验共 `50` 组，排除以下两个历史实验：

1. `v1.0_scratch_ce_cpu`
2. `v1.1_scratch_ce_cuda`

保留并重跑的其余实验包括：

1. `10` 组迁移学习单模型实验
2. `10` 组三肿瘤专家级联实验
3. `30` 组两两肿瘤专家级联实验

新的 B1 正式实验命名规则为：在原正式实验名后追加 `_b1`

例如：

- `v2.0_pretrained_ce_b1`
- `v3.4_pretrained_focal_ls_cosine_cbam_b1`
- `cascade_v3.4_pretrained_focal_ls_cosine_cbam_b1`
- `cascade_pair_lc_sq_v3.4_pretrained_focal_ls_cosine_cbam_b1`

## 13. B1 一键运行脚本

已新增：

- `scripts/run_all_efficientnet_b1_50.ps1`
- `scripts/run_all_efficientnet_b1_50.cmd`
- `scripts/run_all_cascade_tumor3.ps1`
- `scripts/run_all_cascade_tumor_pairs.ps1`

一条命令启动：

```powershell
.\scripts\run_all_efficientnet_b1_50.cmd
```

该脚本会自动完成：

1. `10` 组 B1 单模型训练
2. `10` 个三肿瘤专家模型训练与 `10` 次三肿瘤级联
3. `30` 个两两专家模型训练与 `30` 次两两级联
4. 自动跳过已完成目录，支持续跑

补充说明：

- 新的 B1 主线会把预训练权重缓存到项目内 `2026APMCM/.cache/weights`
- 旧的两个级联脚本已经固定为 `EfficientNet-B0` 历史复现脚本，不会跟随代码默认 backbone 一起切到 B1

## 14. 相关论文借鉴

项目参考的论文位于上级目录 `..\相关论文(1)`。当前已经落地并被实验部分验证的思路主要包括：

1. 迁移学习：`pretrained` 已验证为当前最关键改进
2. `EfficientNet` 路线：当前主线从 B0 迁移到 B1，属于同一结构族的渐进放大
3. 轻量注意力机制：`CBAM` 已证明优于额外 `SE`
4. 难例聚焦与边界平滑：`Focal Loss`、`label smoothing` 已被纳入正式实验主线

## 15. 后续优化方向

1. 先完成 B1 的 `50` 组正式实验
2. 对比 B1 与历史 B0 在单模型和级联两条主线上的收益差异
3. 若 B1 表现更好，再继续微调触发阈值并补充误差分析
