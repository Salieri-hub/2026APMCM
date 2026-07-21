# AI Context

## 当前状态

仓库默认代码路径已经从 `EfficientNet-B3` 切换为 `EfficientNet-B4`。

当前进度如下：

- `B0` 到 `B4` 的单模型、专家模型和级联实验结果均已落盘到 `outputs/results/`
- 默认主干网络：`efficientnet_b4`
- 默认输入尺寸：`320`
- 支持模式：`single`、`expert`、`cascade`
- 本地预训练缓存已支持 `B4`
- 输出目录继续采用共享的 `weights/` 与 `results/` 结构
- `src/main.py` 已完成拆分，当前仅保留入口，核心实现位于 `src/lcc/`

当前 `B4` 本地预训练权重约定路径为：

- `.cache/weights/efficientnet_b4.ra2_in1k/model.safetensors`

如果本地文件不存在，代码会回退到 `timm` 的预训练加载流程。

## 输出规范

所有新实验统一使用以下输出结构：

- `outputs/weights/<experiment_name>/best_model.pt`
- `outputs/results/<experiment_name>/metrics_summary.json`
- `outputs/results/<experiment_name>/...其他 csv 文件...`

不要再使用旧的平铺目录形式 `outputs/<experiment_name>/weights/...`。

## 当前最佳结果

以下结果均以四分类完整测试集（`test=315`）为准，不将专家子集实验与全任务结果混排。

当前最佳单模型：

- 实验名：`v3.2_pretrained_focal_ls_cosine_cutmix_b4`
- 主干网络：`efficientnet_b4`
- 关键配置：`pretrained + focal + label_smoothing + cosine + cutmix`
- 测试集准确率：`94.29%`
- 测试集 Macro F1：`0.9415`

当前最佳级联结果（全仓库 overall best）：

- 实验名：`cascade_pair_ad_sq_v3.2_pretrained_focal_ls_cosine_cutmix_b4`
- 主模型：`v3.2_pretrained_focal_ls_cosine_cutmix_b4`
- 专家模型：`expert_pair_ad_sq_v3.2_pretrained_focal_ls_cosine_cutmix_b4`
- 专家子集：`adenocarcinoma` / `squamous.cell.carcinoma`
- 触发条件：`top-k=2`，`margin <= 0.12`
- 测试集准确率：`94.60%`
- 测试集 Macro F1：`0.9439`

## B4 批量脚本

`B4` 的 `50` 组正式实验已经跑完，相关脚本保留用于复现或重新执行：

- `scripts/run_all_efficientnet_b4_50.ps1`
- `scripts/run_all_efficientnet_b4_50.cmd`

这 `50` 组正式输出包括：

- `10` 组单模型实验
- `10` 组三分类肿瘤专家级联实验
- `30` 组两两专家级联实验

## 代码结构

- `src/main.py`：程序入口，仅调用 `lcc.cli.main()`
- `src/lcc/cli.py`：命令行参数与模式分发
- `src/lcc/data.py`、`models.py`、`losses.py`、`train.py`：数据、模型、损失与训练流程
- `src/lcc/cascade.py`、`reporting.py`、`runtime.py`：级联推理、结果导出与运行时工具

## 模型逻辑

主模型：

- 四分类分类器
- 默认主干为 `EfficientNet-B4`

专家模型：

- 二分类或三分类子集分类器
- 与主模型保持同一主干家族

级联触发逻辑：

1. 先由主模型完成第一次预测。
2. 只有当主模型 top-k 类别全部落在专家子集内时，专家模型才有资格触发。
3. 同时要求 `top1 - top2 <= expert_margin_threshold`。
4. 专家模型不会直接硬替换主模型输出，而是在专家子集内部重新分配概率质量。

## 论文资料路径

撰写正式文档时，请使用本地论文目录：

- `..\相关论文(1)`

## 下一步

1. 将当前最佳 `B4` 结果继续同步到最终 Word 报告与论文正文。
2. 整理 `B0` 到 `B4` 的横向对比摘要与图表。
3. 如需继续优化，再调 `expert_margin_threshold`、`top-k` 与专家触发策略。
