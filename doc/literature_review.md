# 文献综述笔记

## 论文目录

请使用本地论文目录：

- `..\相关论文(1)`

## 本项目采用的核心文献结论

### 1. 迁移学习

相关文献普遍支持：对于规模较小的病理图像数据集，使用 ImageNet 预训练主干通常优于从零训练。

本项目的对应做法：

- 使用 `--pretrained`
- 当前默认主干家族为 `EfficientNet`

### 2. EfficientNet 系列

文献普遍认为，轻量到中等规模的 EfficientNet 主干适合作为组织病理图像分类的强基线。

本项目的演进路径为：

- 历史完整正式结果：`EfficientNet-B0`
- 用户此前恢复的默认基线：`EfficientNet-B1`
- 中间迁移版本：`EfficientNet-B2`
- 当前默认线路：`EfficientNet-B3`

### 3. 损失函数与正则化

相关论文中较常见且有效的策略包括：

- `label smoothing`
- `focal loss`
- 部分类别不平衡场景下的类别重加权
- 更强的数据增强，例如 `MixUp` 与 `CutMix`

本项目已经支持：

- `cross_entropy`
- `focal`
- `label smoothing`
- `balanced/manual` 类别加权
- `MixUp`
- `CutMix`

### 4. 学习率调度

文献通常不建议始终使用固定学习率，非平凡调度策略更容易取得稳定收益。

本项目已经支持：

- `none`
- `cosine`
- `plateau`

### 5. 注意力模块

相关论文经常通过轻量注意力模块增强局部判别特征。

本项目已经支持：

- `SE`
- `CBAM`

### 6. 专家模型与级联逻辑

对于视觉上容易混淆的类别，引入第二阶段专家分类器是合理的工程方案。

本项目已经支持：

- 三分类肿瘤专家分支
- 两两专家分支
- 基于 top-k 包含关系和 margin 阈值的级联触发机制

## 当前工程决策

当前代码路径在方法层面仍然与文献结论保持一致，同时维持可控的工程消融框架：

- 任务定义不变
- 主训练流程不变
- 专家模型与级联框架不变
- 默认主干升级为 `EfficientNet-B3`

## 输出规范

所有新的 `B3` 实验统一使用：

- `outputs/weights/<experiment_name>/`
- `outputs/results/<experiment_name>/`
