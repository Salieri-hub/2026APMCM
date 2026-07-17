# 问题二分析与 Baseline 方案

## 1. 题目二需要完成什么

问题二要求围绕四分类肺部影像识别任务完成三部分内容：

1. 说明训练集、验证集、测试集在模型训练流程中的作用。
2. 给出分类准确率公式，并计算当测试集 315 张中正确识别 279 张时的准确率。
3. 搭建一个可运行的肺癌影像分类模型，并讨论如何降低腺癌与鳞状细胞癌的误诊率。

## 2. 数据集作用

- 训练集：用于更新模型参数，让模型学习四类图像的判别特征。
- 验证集：用于调参、比较不同模型配置，并作为保存最佳模型的依据。
- 测试集：只在模型训练结束后使用，用于评估泛化能力，不能参与训练和调参。

## 3. 准确率公式

分类准确率定义为：

```text
Accuracy = 正确分类样本数 / 总样本数 × 100%
```

当测试集总数为 315，正确识别 279 张时：

```text
Accuracy = 279 / 315 × 100% = 88.57%
```

## 4. Baseline 模型说明

本项目的 baseline 使用以下配置：

- 框架：PyTorch + timm
- 主干网络：EfficientNet-B0
- 损失函数：CrossEntropyLoss
- 优化器：AdamW
- 默认训练轮数：25 epoch
- 输入尺寸：224 x 224

代码入口为 `src/main.py`，默认读取：

```text
附件/Data/train
附件/Data/valid
附件/Data/test
```

脚本会自动处理训练集、验证集、测试集目录名称不完全一致的问题，并统一映射到以下 4 个类别：

1. `adenocarcinoma`
2. `large.cell.carcinoma`
3. `normal`
4. `squamous.cell.carcinoma`

## 5. 运行方式

在项目根目录执行：

```powershell
.\LCC\Scripts\python.exe .\src\main.py
```

如果需要把训练轮数改到 20 或 30：

```powershell
.\LCC\Scripts\python.exe .\src\main.py --epochs 20
.\LCC\Scripts\python.exe .\src\main.py --epochs 30
```

## 6. 输出结果

运行完成后，脚本会在 `outputs/problem2_baseline` 下生成：

- `best_model.pt`：验证集准确率最高的模型权重
- `metrics_summary.json`：训练过程和最终指标汇总
- `test_predictions.csv`：测试集逐样本预测结果
- `valid_confusion_matrix.csv`：验证集混淆矩阵
- `test_confusion_matrix.csv`：测试集混淆矩阵

## 7. 降低腺癌和鳞癌误诊率的优化方向

可在 baseline 基础上继续优化：

1. 使用预训练权重，提高小样本场景下的特征提取能力。
2. 针对腺癌与鳞癌增加更有针对性的增强策略，扩充边界样本。
3. 对易混类别引入类别加权或焦点损失，降低少数难分类样本被忽略的风险。
4. 针对混淆矩阵做误差分析，筛查被错分最多的样本并进行清洗或重标注。
5. 采用测试时增强、模型集成或更高分辨率输入，进一步提升稳定性。
