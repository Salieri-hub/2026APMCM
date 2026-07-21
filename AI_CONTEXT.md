# AI Context

## 当前状态

仓库默认代码路径已经从 `EfficientNet-B2` 切换为 `EfficientNet-B3`。

当前默认行为如下：

- 主干网络：`efficientnet_b3`
- 默认输入尺寸：`288`
- 支持模式：`single`、`expert`、`cascade`
- 本地预训练缓存已支持 `B3`
- 输出目录继续采用共享的 `weights/` 与 `results/` 结构

当前 `B3` 本地预训练权重约定路径为：

- `.cache/weights/efficientnet_b3.ra2_in1k/model.safetensors`

如果本地文件不存在，代码会回退到 `timm` 的预训练加载流程。

## 输出规范

所有新实验统一使用以下输出结构：

- `outputs/weights/<experiment_name>/best_model.pt`
- `outputs/results/<experiment_name>/metrics_summary.json`
- `outputs/results/<experiment_name>/...其他 csv 文件...`

不要再使用旧的平铺目录形式 `outputs/<experiment_name>/weights/...`。

## 历史完整结果

当前已经完整跑通并验证的正式结果仍然是历史 `B0` 线路。

历史单模型最佳结果：

- 实验名：`v3.4_pretrained_focal_ls_cosine_cbam`
- 主干网络：`efficientnet_b0`
- 测试集准确率：`86.35%`
- 测试集 Macro F1：`0.8646`

历史级联最佳结果：

- 实验名：`cascade_v3.4_pretrained_focal_ls_cosine_cbam`
- 主模型：`v3.4_pretrained_focal_ls_cosine_cbam`
- 专家模型：`expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam`
- 触发条件：`top-k=2`，`margin <= 0.12`
- 测试集准确率：`87.62%`
- 测试集 Macro F1：`0.8773`

## 当前 B3 目标

执行 `50` 组正式 `B3` 实验，不包含：

- `v1.0_scratch_ce_cpu`
- `v1.1_scratch_ce_cuda`

这 `50` 组正式输出包括：

- `10` 组单模型实验
- `10` 组三分类肿瘤专家级联实验
- `30` 组两两专家级联实验

对应脚本如下：

- `scripts/run_all_efficientnet_b3_50.ps1`
- `scripts/run_all_efficientnet_b3_50.cmd`

## 模型逻辑

主模型：

- 四分类分类器
- 默认主干为 `EfficientNet-B3`

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

1. 运行 `B3` 批量脚本。
2. 对比 `B3` 与历史 `B0` 的结果。
3. 如果 `B3` 效果更优，再更新 Word 摘要与论文式实验分析。
