# 相关论文阅读与可迁移思路

## 1. 说明

本文件用于整理项目上级目录 `..\相关论文` 中参考文献的初步阅读结论，目标不是复现全部论文细节，而是提炼对当前四分类肺部 CT baseline 最有价值、最容易落地的优化方向。

当前归纳主要基于论文标题、摘要、方法结构与任务设定的对照，服务于本仓库下一阶段的实验设计。

## 2. 总体结论

对当前项目最有帮助的思路可以压缩为四点：

1. 优先使用迁移学习。随机初始化的 CNN baseline 在小样本四分类任务上通常偏弱。
2. 优先试轻量注意力或全局上下文增强，而不是立刻大改为完整 Transformer 流程。
3. 对腺癌、鳞癌高误诊问题，要从损失函数、采样策略和误差分析同时入手。
4. 后续实验汇报不能只看准确率，应同步看 `macro F1`、各类别召回率和混淆矩阵。

## 3. 论文拆解

### 3.1 Leveraging Transfer Learning and Attention Mechanisms for a Computed Tomography Lung Cancer Classification Model

- 任务相关性：高。直接面向肺癌 CT 分类。
- 关键做法：迁移学习、注意力机制、微调训练。
- 对本项目的启发：
  - `--pretrained` 应作为最近一轮 baseline 优化的首选实验。
  - 注意力模块值得在现有 CNN 框架中增量加入。
  - 后续训练配置可考虑加入 `label smoothing` 一类缓解过度自信预测的手段。

### 3.2 Lung-EffNet

- 任务相关性：高。核心路线与当前项目都围绕 `EfficientNet`。
- 关键做法：`EfficientNet` 系列、迁移学习、数据增强。
- 对本项目的启发：
  - 当前 `EfficientNet-B0` 路线本身可继续保留。
  - 在资源允许时，可补试 `EfficientNet-B1`，比较是否优于 `B0`。
  - 数据增强不应只停留在基础随机裁剪与翻转，应增强对易混类别的针对性。

### 3.3 Classification of lung cancer subtypes on CT images with synthetic pathological priors

- 任务相关性：高。直接针对肺癌亚型分类。
- 关键做法：融合额外先验信息，提高亚型可分性。
- 对本项目的启发：
  - 当前单流图像分类器过于朴素，后续应考虑辅助监督或更强先验。
  - 在现阶段无法直接复现病理先验时，可先用类别加权、误差驱动的数据分析和辅助损失做弱化替代。
  - 重点不是继续堆 epoch，而是增强亚型边界学习能力。

### 3.4 CCT Lightweight compact convolutional transformer for lung disease CT image classification

- 任务相关性：中。任务不是完全同一问题，但方法思路有借鉴价值。
- 关键做法：轻量卷积 Transformer，强调全局上下文建模。
- 对本项目的启发：
  - 当前误判模式说明模型可能过度依赖局部纹理。
  - 全局上下文增强是合理方向，但第一步应优先做轻量模块而不是全面重写训练框架。

### 3.5 EfficientNet with attention and global blocks for accurate pulmonary disease detection in chest CT scans

- 任务相关性：中。病种不同，但结构设计可迁移。
- 关键做法：`EfficientNet` + 注意力 / global block。
- 对本项目的启发：
  - 现有 `EfficientNet` 路线可以自然扩展到“backbone + 上下文增强”版本。
  - 如果需要做结构升级，应优先选兼容当前代码的小改动方案。

### 3.6 Emerging computational intelligence based techniques for lung cancer diagnosis and classification on chest CT scan images

- 任务相关性：中。综述性质，适合把握共性结论。
- 关键做法：总结肺癌 CT 分类中的常见方法组合。
- 对本项目的启发：
  - 高性能方案往往不是裸 CNN，而是迁移学习、注意力、损失设计、可解释性分析的组合。
  - 论文写作时可以用它来支撑“为什么选择这些优化路线”。

## 4. 对当前仓库的具体改造建议

按投入产出比排序，建议依次做：

1. `pretrained EfficientNet-B0` 复现实验。
2. 加入学习率调度器与 `label smoothing`。
3. 尝试类别加权 `CrossEntropyLoss`、`Focal Loss` 或采样策略。
4. 在当前 backbone 上试验 `SE` / `CBAM` 等轻量注意力模块。
5. 基于混淆矩阵做误差样本分析，并补充 `Grad-CAM`。
6. 如 CPU 试验周期过长，再考虑切换 GPU 或缩小搜索范围。

## 5. 实验记录建议

后续每次实验至少记录以下项目：

- 模型结构与是否预训练
- 损失函数与采样策略
- 学习率、轮数、batch size
- 验证集准确率
- 测试集准确率
- 测试集 `macro F1`
- 各类别召回率
- 关键混淆项是否改善

## 6. 当前结论

对于本项目，最现实、最值得优先执行的路线不是直接推翻现有 baseline，而是在 `EfficientNet` 主线上逐步加入：

1. 预训练迁移
2. 更合理的优化目标与损失设计
3. 轻量注意力 / 全局上下文增强
4. 更细的误差分析与可解释性检查
## Ablation Implications

- Pretraining remains the strongest transferable improvement for this dataset.
- Label smoothing is the best low-cost regularizer in the current runs.
- MixUp and CutMix do not help under the present small-sample setting.
- Extra CBAM is the strongest structural add-on; `EfficientNet-B0` already contains internal SE blocks.
