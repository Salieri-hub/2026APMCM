# 问题二分析与 Baseline 方案

## 1. 题目二需要完成什么

问题二要求围绕四分类肺部 CT 影像识别任务完成三部分内容：

1. 说明训练集、验证集、测试集在模型训练流程中的作用。
2. 给出分类准确率公式，并计算测试集 `315` 张中正确识别 `279` 张时的准确率。
3. 搭建一个可运行的肺癌影像分类模型，并讨论如何降低腺癌与鳞状细胞癌的误诊率。

## 2. 数据集作用

- 训练集：用于更新模型参数，让模型学习四类图像的判别特征。
- 验证集：用于调参、比较不同模型配置，并作为保存最佳模型的依据。
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

## 4. 已实现的 Baseline

当前 baseline 采用如下配置：

- 框架：PyTorch + timm
- 主干网络：EfficientNet-B0
- 损失函数：CrossEntropyLoss
- 优化器：AdamW
- 默认训练轮数：25 epoch
- 输入尺寸：224 x 224
- 代码入口：`src/main.py`

默认读取的数据目录为：

```text
../附件/Data/train
../附件/Data/valid
../附件/Data/test
```

脚本会自动处理训练集、验证集、测试集目录命名不完全一致的问题，并统一映射到以下四类：

1. `adenocarcinoma`
2. `large.cell.carcinoma`
3. `normal`
4. `squamous.cell.carcinoma`

## 5. 运行方式

在项目根目录执行：

```powershell
..\LCC\Scripts\python.exe .\src\main.py
```

常见运行方式：

```powershell
..\LCC\Scripts\python.exe .\src\main.py --epochs 20
..\LCC\Scripts\python.exe .\src\main.py --epochs 30
..\LCC\Scripts\python.exe .\src\main.py --pretrained
```

路径说明：

- 当前项目代码位于 `B题` 目录内。
- 数据集位于项目外部同级目录 `..\附件\Data`。
- 虚拟环境位于项目外部同级目录 `..\LCC`。
- `src/main.py` 默认会优先使用 `..\附件\Data`，兼容当前目录布局。

## 6. 输出结果

脚本会在 `outputs/problem2_baseline` 下生成：

- `best_model.pt`：验证集准确率最高的模型权重
- `metrics_summary.json`：训练过程与最终指标汇总
- `test_predictions.csv`：测试集逐样本预测结果
- `valid_confusion_matrix.csv`：验证集混淆矩阵
- `test_confusion_matrix.csv`：测试集混淆矩阵

## 7. 当前最新实验结果

最近一次已完成检查的运行配置：

- `epochs=20`
- `batch_size=16`
- `image_size=224`
- `lr=3e-4`
- `weight_decay=1e-4`
- `pretrained=false`
- `device=cpu`

对应结果：

- 最佳验证集轮次：`epoch 15`
- 最佳验证集准确率：`62.50%`
- 测试集准确率：`39.68%`
- 测试集 `macro F1`：`0.4427`
- 测试集 `weighted F1`：`0.3663`

测试集各类别召回率：

- `adenocarcinoma`：`6.67%`
- `large.cell.carcinoma`：`96.08%`
- `normal`：`83.33%`
- `squamous.cell.carcinoma`：`25.56%`

## 8. 结果解读

当前 baseline 已经完成了问题二最基本的端到端流程验证，但模型效果还不能作为最终方案提交，主要原因有：

1. 验证集最好成绩为 `62.50%`，但测试集仅 `39.68%`，泛化落差较大。
2. 模型存在明显的类别预测偏置，大量样本被预测成 `large.cell.carcinoma`。
3. 题目重点关注的腺癌与鳞癌误诊问题在当前 baseline 中依然突出。

从测试集混淆情况看：

- 腺癌 `120` 张中仅 `8` 张预测正确，`106` 张被判成了大细胞癌。
- 鳞癌 `90` 张中仅 `23` 张预测正确，`57` 张被判成了大细胞癌。
- 正常肺部样本识别相对稳定。

## 9. 相关论文借鉴

项目上级目录 `..\相关论文` 中的参考文献完成初步阅读后，当前可借鉴的思路主要包括：

1. 迁移学习是四分类肺部 CT 任务中的基础配置。多篇论文都采用预训练 backbone，再针对医学影像数据做微调。
2. 注意力机制和全局上下文增强值得优先尝试。相比直接大改架构，`SE`、`CBAM`、global block 一类轻量模块更适合当前 baseline 增量验证。
3. 对于肺癌亚型间混淆严重的问题，仅靠更长训练通常不足，往往还需要类别加权损失、焦点损失、`label smoothing` 或更针对性的采样策略。
4. 论文中的指标对比通常不只看准确率，还会联合使用 `macro F1`、AUC、各类召回率和混淆矩阵分析。

对当前仓库最有直接借鉴价值的论文包括：

- `Leveraging Transfer Learning and Attention Mechanisms for a Computed Tomography Lung Cancer Classification Model`
  - 启发：优先把预训练与注意力模块结合，而不是继续使用纯随机初始化的 backbone。
- `Lung-EffNet`
  - 启发：`EfficientNet` 系列在肺部 CT 分类中是可行路线，后续可在 `B0` 之外补试 `B1`。
- `Classification of lung cancer subtypes on CT images with synthetic pathological priors`
  - 启发：肺癌亚型分类可结合额外先验或辅助信息，当前项目可先用更容易落地的类别加权、误差分析与辅助监督思路做弱化替代。
- `CCT Lightweight compact convolutional transformer for lung disease CT image classification`
  - 启发：全局上下文有价值，但现阶段应优先试轻量增强模块，而不是直接大规模重写训练框架。

## 10. 后续优化方向

在当前 baseline 基础上，优先考虑以下优化路径：

1. 启用 `--pretrained`，增强小样本场景下的特征提取能力。
2. 引入学习率调度器与 `label smoothing`，降低后期训练波动与过度自信预测。
3. 对腺癌与鳞癌尝试类别加权损失、`Focal Loss` 或采样策略。
4. 在当前 `EfficientNet` baseline 上增量试验轻量注意力模块。
5. 加强针对易混类别的图像增强策略。
6. 基于混淆矩阵开展误差分析，检查被高频误判样本，必要时补充 `Grad-CAM`。
7. 如条件允许，切换到 GPU 环境缩短迭代周期。
