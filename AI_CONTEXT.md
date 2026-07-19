# AI_CONTEXT

## 当前项目目标

项目目标是完成 APMCM 2026 B题“肺癌疾病诊断图像识别与分类问题”的阶段性求解，当前重点包括：

1. 梳理问题一的数据统计分析要求并形成可写入论文的结论。
2. 搭建问题二的可运行四分类主模型，并开展系统化消融实验。
3. 在主模型基础上训练专家模型，并通过级联触发机制降低肿瘤亚型之间的误诊率。
4. 将实验结果沉淀为可直接用于论文、答辩和汇报的文档材料。
5. 在保留 B0 历史结果的前提下，把统一 backbone 从 `EfficientNet-B0` 迁移到 `EfficientNet-B1`，重新执行除 CPU baseline 与非迁移 GPU baseline 以外的其余 `50` 组正式实验。

## 当前整体进度

当前已完成题目理解、数据结构核对、`src/main.py` 主流程搭建、GPU 环境准备、B0 单模型正式消融、B0 专家模型训练和 B0 级联实验扩展。项目已从“先把 baseline 跑通”进入“保留已完成 B0 结论，同时切换到 B1 主线重新跑正式实验”的阶段。

截至 `2026-07-19`：

- 已完成并保留的历史正式结果：
  - `12` 组单模型实验
  - `10` 组三肿瘤专家级联实验
  - `30` 组两两肿瘤专家级联实验
  - 共 `52` 组正式结果
- 上述 `52` 组结果统一基于 `EfficientNet-B0`
- 当前代码默认 backbone 已切换到 `EfficientNet-B1`
- `EfficientNet-B1` 的 `50` 组正式实验批量脚本已写好，但本轮尚未等待其全部运行完成

## 本次已完成内容

- 阅读赛题 PDF，完成问题一任务拆解和核心计算结论。
- 核对项目外部目录 `..\附件\Data` 下 `train/valid/test` 的目录结构与类别映射。
- 在 `src/main.py` 中实现四分类训练、验证、测试主流程。
- 将 `src/main.py` 扩展为支持三种模式：
  - `single`：四分类主模型训练与评估
  - `expert`：类别子集专家模型训练
  - `cascade`：主模型 + 专家模型级联评估
- 在训练脚本中加入：
  - `CUDA/CPU` 自动切换
  - AMP 混合精度
  - `label smoothing`
  - 学习率调度器
  - `Focal Loss`
  - 类别加权 `CrossEntropy`
  - `MixUp / CutMix`
  - `SE / CBAM`
- 建立并验证 `..\LCC_GPU` CUDA 环境。
- 完成 B0 `v1.0` 到 `v3.4` 的 `12` 组单模型正式实验。
- 完成 B0 `10` 组三肿瘤专家级联实验。
- 完成 B0 `30` 组两两肿瘤专家级联实验。
- 核查 `40` 个专家模型训练日志，确认主要问题为过拟合而非欠拟合。
- 将代码默认 backbone 改为 `EfficientNet-B1`，同时保持对旧 B0 checkpoint 的兼容。
- 增加 B1 默认输入尺寸逻辑：
  - `B0 -> 224`
  - `B1 -> 240`
- 为 B1 正式实验增加新的输出命名规则：
  - 在原正式实验名后追加 `_b1`
- 为避免 Windows 对 `huggingface` 系统缓存目录的权限和路径问题，B1 预训练权重改为下载到项目内 `2026APMCM/.cache/weights`
- 新增 B1 批量脚本：
  - `scripts/run_all_efficientnet_b1_50.ps1`
  - `scripts/run_all_efficientnet_b1_50.cmd`
- 将旧的 B0 级联批处理脚本固定为：
  - `model_name=efficientnet_b0`
  - `image_size=224`
  以保证其仍然复现历史 B0 结果，而不是误切到 B1

## 已修改模块

- `src/main.py`
- `README.md`
- `AI_CONTEXT.md`
- `TODO.md`
- `doc/problem2_baseline.md`
- `doc/literature_review.md`
- `doc/ablation_results.md`
- `doc/progress_report_2026-07-19.md`
- `scripts/run_all_efficientnet_b1_50.ps1`
- `scripts/run_all_efficientnet_b1_50.cmd`
- `scripts/run_all_cascade_tumor3.ps1`
- `scripts/run_all_cascade_tumor_pairs.ps1`
- `scripts/generate_version_comparison_docx.py`
- `scripts/generate_outputs_version_ablation_docx.py`
- `scripts/generate_52_experiment_comparison_docx.py`
- `scripts/generate_ablation_results_cn_docx.py`
- `..\LCC_GPU\*`

## 当前模型 / 算法状态

当前代码默认主线配置：

- 框架：PyTorch + timm
- 默认主干网络：`efficientnet_b1`
- 默认输入尺寸：`240`
- 主模型任务：四分类
- 专家模型任务：二分类或三分类子任务
- 专家模型 backbone：与主模型保持一致
- 优化器：`AdamW`
- 损失函数：支持 `CrossEntropyLoss / Focal Loss`
- 学习率调度：支持 `none / cosine / plateau`
- 结构增量：支持 `SE / CBAM`
- 数据增强：支持 `MixUp / CutMix`
- 设备策略：`auto -> cuda if available else cpu`

级联触发逻辑：

1. 先由四分类主模型输出 4 类概率。
2. 只有当主模型 `top-k` 预测全部落在专家类别子集内时，专家模型才有资格触发。
3. 当 `top1 - top2` 的置信差值不大于 `--expert-margin-threshold` 时，认为主模型在该局部边界上不够确定，触发专家模型二次判别。
4. 触发后不是直接覆盖主模型，而是对专家子集内的概率做重新分配，再输出最终四分类结果。

## 当前已验证最佳结果

当前“已完成并验证”的最佳单模型结果仍来自 B0 历史实验：

- 目录：`outputs/v3.4_pretrained_focal_ls_cosine_cbam`
- backbone：`efficientnet_b0`
- 配置：`pretrained + focal loss(gamma=2) + label smoothing(0.1) + cosine scheduler + CBAM`
- 最佳验证集准确率：`93.06%`
- 测试集准确率：`86.35%`
- 测试集 `macro F1`：`0.8646`

当前“已完成并验证”的全局最佳结果仍来自 B0 历史级联实验：

- 目录：`outputs/cascade_v3.4_pretrained_focal_ls_cosine_cbam`
- 主模型：`v3.4_pretrained_focal_ls_cosine_cbam`
- 专家模型：`expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam`
- 触发参数：`top-k=2`，`margin <= 0.12`
- 测试集准确率：`87.62%`
- 测试集 `macro F1`：`0.8773`
- 测试集触发统计：
  - `expert_invocations = 17`
  - `expert_changed_predictions = 8`
  - `expert_corrected_predictions = 5`
  - `expert_hurt_predictions = 1`

## 当前分析结论

1. `pretrained` 是最重要的单步提升。
2. 在单模型体系中，`label smoothing` 是当前最有效的低成本正则项。
3. `balanced class-weighted CE`、`MixUp`、`CutMix` 在当前小样本设置下整体无益。
4. `CBAM` 明显优于额外 `SE`，是当前最有效的结构增量。
5. 三肿瘤专家级联整体比两两肿瘤专家级联更稳，平均收益更一致。
6. 从训练日志看，绝大多数专家模型并不存在典型欠拟合，主要问题是过拟合和验证波动。
7. 基于文献与当前结果，`EfficientNet-B1` 是合理的下一轮统一 backbone。

## 当前存在的问题

1. 验证集只有 `72` 张，导致验证准确率和最佳轮次存在较明显波动。
2. 单模型体系下鳞癌召回率仍是主要短板之一。
3. 不是所有级联版本都优于单模型，说明当前触发规则仍有优化空间。
4. B1 代码和批量脚本已就绪，但其 `50` 组正式实验结果尚未全部生成完成。

## 文献调研结论

基于 `..\相关论文(1)` 的整理，当前可以视为已经被实验验证或部分验证的结论如下：

1. 迁移学习是该任务中收益最大的低成本配置。
2. `EfficientNet` 作为统一 backbone 是合理路线，适合在固定主干上做消融。
3. 对于肿瘤亚型混淆问题，损失设计和注意力模块比盲目增加训练轮数更有效。
4. 文献中的 `EfficientNet` 渐进扩展路线支持从 `B0` 升级到 `B1` 继续重跑正式实验。

## 下一步计划

1. 运行 `scripts/run_all_efficientnet_b1_50.cmd` 或 `scripts/run_all_efficientnet_b1_50.ps1`，完成 B1 的 `50` 组正式实验。
2. 等 B1 批量结果生成后，比较 B1 与历史 B0 在单模型和级联两条主线上的差异。
3. 若 B1 表现更好，再同步更新 Word 汇总与论文分析文案。

## 需要提醒后续协作者的事项

- 后续回答和修改应优先参考本文件，其次是 `README.md` 和 `doc/problem2_baseline.md`
- 正式文档中关于论文来源的路径应写为 `..\相关论文(1)`，不要再写旧的 `..\相关论文`
- 当前代码默认 backbone 已经是 `EfficientNet-B1`
- 历史 B0 结果目录不要覆盖
- 新一轮 B1 正式结果统一使用 `_b1` 后缀目录
