# 历史消融结果汇总

## 范围说明

本文件用于汇总已经完整完成的历史 `B0` 消融线路。

该完成线路包括：

- `12` 组早期原始实验
- 以及后续在其他文档中记录的级联扩展实验

当前仓库默认线路已经切换为 `B4`，但 `B4` 的正式 `50` 组复现实验尚未执行完毕。

## 历史 B0 单模型最佳结果

- 实验名：`v3.4_pretrained_focal_ls_cosine_cbam`
- 主干网络：`efficientnet_b0`
- 配置：预训练、`focal loss`、`label smoothing`、`cosine scheduler`、`CBAM`
- 测试集准确率：`86.35%`
- 测试集 Macro F1：`0.8646`

## 历史 B0 级联最佳结果

- 实验名：`cascade_v3.4_pretrained_focal_ls_cosine_cbam`
- 主模型：`v3.4_pretrained_focal_ls_cosine_cbam`
- 专家模型：`expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam`
- 触发条件：`top-k=2`，`margin <= 0.12`
- 测试集准确率：`87.62%`
- 测试集 Macro F1：`0.8773`

## 历史经验

1. 预训练带来的提升最稳定，也是单项收益最大的改动。
2. `label smoothing` 与 `focal loss` 的泛化收益通常比普通加权交叉熵更稳定。
3. 轻量注意力模块在该任务上仍然有价值。
4. 专家级联对高混淆类别有帮助，但并不是所有级联变体都能稳定超过对应的单模型。

## 当前 B4 状态

代码库已经完成新一轮 `B4` 复现实验的准备工作，实验条件继续沿用此前非 CPU、预训练的正式设置。

正式 `B4` 计划如下：

- `10` 组单模型实验
- `10` 组三分类肿瘤专家级联实验
- `30` 组两两专家级联实验
- 总计：`50`

新线路的输出目录如下：

- `outputs/weights/<experiment_name>/`
- `outputs/results/<experiment_name>/`
