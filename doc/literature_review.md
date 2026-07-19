# 相关论文阅读与可迁移思路

## 1. 说明

本文档用于整理项目上级目录 `..\相关论文(1)` 中参考论文的初步阅读结论。目标不是逐篇复现论文，而是提炼对当前肺部 CT 四分类任务最有价值、最容易落地的优化方向，并与本项目现有实验结果和下一轮 B1 计划对应起来。

## 2. 总体结论

对当前项目最有帮助的思路可以概括为六点：

1. 优先使用迁移学习。随机初始化的 CNN 在当前小样本四分类任务上明显偏弱。
2. 保持统一 backbone，更有利于开展可信的消融实验。
3. 在小样本医学图像场景中，轻量注意力机制通常比大幅改写架构更稳妥。
4. 对肿瘤亚型间混淆问题，损失函数、边界平滑和专家式细分往往比单纯增加 epoch 更有效。
5. 评价模型时不能只看准确率，应联合 `macro F1`、各类召回率和混淆矩阵。
6. `EfficientNet` 的渐进放大路线支持从 `B0` 升级到 `B1`，这是当前最自然的下一轮主线实验。

## 3. 论文拆解

### 3.1 Leveraging Transfer Learning and Attention Mechanisms for a Computed Tomography Lung Cancer Classification Model

- 任务相关性：高，直接面向肺癌 CT 分类
- 关键词：迁移学习、注意力机制、微调训练
- 对本项目的启发：
  - `pretrained` 应作为首要实验方向
  - 注意力模块可以在现有 CNN 框架中做增量验证
  - 若模型出现过度自信，应考虑 `label smoothing`

### 3.2 Lung-EffNet

- 任务相关性：高，与当前项目都围绕 `EfficientNet` 路线展开
- 关键词：`EfficientNet`、迁移学习、数据增强
- 对本项目的启发：
  - 当前 `EfficientNet-B0` 路线可以保留为历史基线
  - 下一轮 backbone 升级优先尝试 `EfficientNet-B1`
  - 数据增强在医学图像任务上需要谨慎，不是越强越好

### 3.3 Classification of lung cancer subtypes on CT images with synthetic pathological priors

- 任务相关性：高，直接关注肺癌亚型分类
- 关键词：亚型边界、先验信息、细粒度区分
- 对本项目的启发：
  - 单模型对肿瘤亚型边界仍不够稳
  - 如果无法直接引入病理先验，至少应通过损失函数设计、误差分析或专家模型去强化局部边界学习
  - 当前专家级联路线，是对“亚型内再细分”思路的工程化近似

### 3.4 CCT Lightweight compact convolutional transformer for lung disease CT image classification

- 任务相关性：中，任务不完全相同，但方法设计有借鉴价值
- 关键词：轻量卷积 Transformer、全局上下文
- 对本项目的启发：
  - 当前误判模式提示模型可能过度依赖局部纹理
  - 轻量上下文增强是合理方向，但第一步应优先试兼容当前框架的小改动，例如 `CBAM`

### 3.5 EfficientNet with attention and global blocks for accurate pulmonary disease detection in chest CT scans

- 任务相关性：中，疾病集合不同，但结构设计可迁移
- 关键词：`EfficientNet`、注意力、全局块
- 对本项目的启发：
  - `EfficientNet` 主线可自然扩展为“backbone + attention”的版本
  - 如果继续做结构优化，应优先选择兼容当前代码的小模块，而不是直接推翻现有训练流程

### 3.6 Emerging computational intelligence based techniques for lung cancer diagnosis and classification on chest CT scan images

- 任务相关性：中，偏综述
- 关键词：方法组合、可解释性、混合路线
- 对本项目的启发：
  - 高性能方案通常不是单一技巧，而是迁移学习、损失设计、注意力和分析方法的组合
  - 论文写作时可以用这类综述支撑“为什么选择这些优化路线”的方法论说明

## 4. 文献启发与当前实验的对应关系

截至 `2026-07-19`，当前项目的实验结果已经对部分文献启发给出了明确验证：

1. `pretrained` 是收益最大的单步改进
2. `label smoothing` 是有效的低成本边界正则
3. `CBAM` 是当前最有效的结构增量
4. 额外 `SE` 在 `EfficientNet-B0` 上收益有限
5. 标准 `balanced weighted CE`、`MixUp`、`CutMix` 在当前小样本设定下不适合作为主线方案
6. 专家级联对肿瘤亚型边界修正是有价值的

## 5. 对当前仓库的具体建议

按投入产出比排序，当前最值得继续做的是：

1. 保留历史 B0 `52` 组结果作为参考基线
2. 以同一结构族把 backbone 从 `B0` 升级到 `B1`
3. 用 B1 重跑除 CPU baseline 和非迁移 GPU baseline 之外的其余 `50` 组正式实验
4. 等 B1 结果完成后，再比较 B0 与 B1 在单模型和级联两条主线上的收益差异

## 6. 当前结论

对于本项目，最现实、最值得优先执行的路线不是推翻现有框架，而是在统一 `EfficientNet` 主线下逐步叠加：

1. 预训练迁移学习
2. 更合理的损失与正则设计
3. 轻量注意力模块
4. 面向高混淆类别的专家级联机制
5. 从 `B0` 到 `B1` 的渐进式 backbone 升级

这也是当前代码已经切换到 `EfficientNet-B1` 的主要依据。
