# AI_CONTEXT

## 当前项目目标

项目目标是完成 APMCM 2026 B题“肺癌疾病诊断图像识别与分类问题”的阶段性求解，当前重点包括：

1. 梳理问题一的数据统计分析要求并形成可写入论文的结论。
2. 搭建问题二的可运行四分类主模型，并开展系统化消融实验。
3. 在主模型基础上训练专家模型，并通过级联触发机制降低肿瘤亚型之间的误诊率。
4. 将实验结果沉淀为可直接用于论文、答辩和汇报的文档材料。

## 当前整体进度

当前已完成题目理解、数据结构核对、`src/main.py` 主流程搭建、GPU 环境准备、单模型正式消融、专家模型训练和级联实验扩展。项目已从“先把 baseline 跑通”进入“围绕最佳主模型和专家级联机制做总结与针对性优化”的阶段。

截至 `2026-07-19`，已完成：

- `12` 组单模型正式实验
- `10` 组三肿瘤专家级联实验
- `30` 组两两肿瘤专家级联实验
- 共 `52` 组正式结果对比
- 对应的 `40` 个专家模型训练
- 多份 Word/Markdown 总结文档与批处理脚本

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
- 完成 `v1.0` 到 `v3.4` 的 `12` 组单模型正式消融实验。
- 完成 `10` 组三肿瘤专家模型训练与级联评估。
- 完成 `30` 组两两肿瘤专家模型训练与级联评估。
- 编写批处理脚本：
  - `scripts/run_all_cascade_tumor3.ps1`
  - `scripts/run_all_cascade_tumor_pairs.ps1`
- 生成并更新文档：
  - `README.md`
  - `doc/problem2_baseline.md`
  - `doc/ablation_results.md`
  - `doc/progress_report_2026-07-19.md`
  - 多个 `.docx` 汇总文档与生成脚本

## 已修改模块

- `src/main.py`
- `README.md`
- `AI_CONTEXT.md`
- `TODO.md`
- `doc/problem2_baseline.md`
- `doc/literature_review.md`
- `doc/ablation_results.md`
- `doc/progress_report_2026-07-19.md`
- `scripts/run_all_cascade_tumor3.ps1`
- `scripts/run_all_cascade_tumor_pairs.ps1`
- `scripts/generate_version_comparison_docx.py`
- `scripts/generate_outputs_version_ablation_docx.py`
- `scripts/generate_52_experiment_comparison_docx.py`
- `scripts/generate_ablation_results_cn_docx.py`
- `..\LCC_GPU\*`

## 当前模型 / 算法状态

当前主线配置：

- 框架：PyTorch + timm
- 主干网络：`efficientnet_b0`
- 主模型任务：四分类
- 专家模型任务：二分类或三分类子任务
- 专家模型 backbone：同样使用 `efficientnet_b0`
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

## 当前最佳结果

当前最佳单模型：

- 目录：`outputs/v3.4_pretrained_focal_ls_cosine_cbam`
- 配置：`pretrained + focal loss(gamma=2) + label smoothing(0.1) + cosine scheduler + CBAM`
- 最佳验证集准确率：`93.06%`
- 测试集准确率：`86.35%`
- 测试集 `macro F1`：`0.8646`

当前全局最佳结果：

- 目录：`outputs/cascade_v3.4_pretrained_focal_ls_cosine_cbam`
- 主模型：`v3.4_pretrained_focal_ls_cosine_cbam`
- 专家模型：三肿瘤专家模型 `expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam`
- 触发参数：`top-k=2`，`margin <= 0.12`
- 测试集准确率：`87.62%`
- 测试集 `macro F1`：`0.8773`
- 测试集触发统计：
  - `expert_invocations = 17`
  - `expert_changed_predictions = 8`
  - `expert_corrected_predictions = 5`
  - `expert_hurt_predictions = 1`

当前最佳两两肿瘤专家级联结果：

- 目录：`outputs/cascade_pair_lc_sq_v3.4_pretrained_focal_ls_cosine_cbam`
- 测试集准确率：`86.67%`
- 测试集 `macro F1`：`0.8681`

## 当前分析结论

1. `pretrained` 是最重要的单步提升，显著强于单纯从 CPU 切到 CUDA。
2. 在单模型体系中，`label smoothing` 是当前最有效的低成本正则项。
3. `balanced class-weighted CE`、`MixUp`、`CutMix` 在当前小样本设置下整体无益。
4. `CBAM` 明显优于额外 `SE`，是当前最有效的结构增量。
5. 三肿瘤专家级联整体比两两肿瘤专家级联更稳，平均收益更一致。
6. 两两肿瘤专家中，`large.cell.carcinoma` vs `squamous.cell.carcinoma` 这一支平均效果最好。
7. 从训练日志看，绝大多数专家模型并不存在典型欠拟合，主要问题是过拟合和验证波动。

## 当前存在的问题

1. 验证集只有 `72` 张，导致验证准确率和最佳轮次存在较明显波动。
2. 单模型体系下鳞癌召回率仍是主要短板之一。
3. 不是所有级联版本都优于单模型，说明当前触发规则仍有优化空间。
4. 专家模型多数已把训练集学得很深，但泛化优势不稳定，提示需要继续控制过拟合。
5. Windows 本地批量训练时若 DataLoader 进程数过高，容易触发页面文件不足问题。

## 文献调研结论

基于 `..\相关论文(1)` 的整理，当前可以视为已经被实验验证或部分验证的结论如下：

1. 迁移学习是该任务中收益最大的低成本配置。
2. `EfficientNet` 作为统一 backbone 是合理路线，适合在固定主干上做消融。
3. 对于肿瘤亚型混淆问题，损失设计和注意力模块比盲目增加训练轮数更有效。
4. 文献中的“轻量注意力 + 迁移学习 + 局部边界增强”思路，与当前项目中的 `CBAM + focal + cascade` 路线是一致的。

## 下一步计划

1. 以 `v3.4` 主模型为核心，继续分析是否需要微调 `expert_margin_threshold` 和 `expert_trigger_topk`。
2. 对最佳单模型和最佳级联模型补充误差分析，重点检查高频混淆样本。
3. 如需继续扩展结构实验，优先围绕当前最佳主模型做小范围修改，而不是重新开大量 backbone 试验。
4. 将 `52` 组实验的结论整理为论文正文所需的实验分析段落。

## 需要提醒后续协作者的事项

- 后续回答和修改应优先参考本文件，其次是 `README.md` 和 `doc/problem2_baseline.md`。
- 正式文档中关于论文来源的路径应写为 `..\相关论文(1)`，不要再写旧的 `..\相关论文`。
- 后续 GPU 训练默认使用 `..\LCC_GPU\python.exe`。
- `outputs` 中已经存在完整的 `12 + 10 + 30` 组正式结果，新增实验不要覆盖这些目录。
- 如果继续训练或新增结果，必须同步更新 `AI_CONTEXT.md`、`TODO.md` 和至少一份对外说明文档。
