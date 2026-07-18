# APMCM 2026 B题

肺癌疾病诊断图像识别与分类问题的本地实验项目。

## 项目目标

本项目当前围绕题目中的两个任务推进：

1. 问题一：完成数据集规模、比例和分层抽样合理性的统计分析。
2. 问题二：搭建四分类肺部 CT 图像识别 baseline，并输出验证集与测试集结果。

## 当前状态

- 问题一的题意分析已完成，核心统计结论已经整理。
- 问题二 baseline 已实现，入口为 `src/main.py`。
- 已在项目外部的 `..\LCC` 虚拟环境下完成一次 `20 epoch` 训练与评估。
- 最新 baseline 为 `EfficientNet-B0 + CrossEntropyLoss + AdamW`。
- 已完成一轮相关文献初步阅读与方法归纳，后续优化方向已明确。

最新一次已验证运行结果：

- 最佳验证集轮次：`epoch 15`
- 最佳验证集准确率：`62.50%`
- 测试集准确率：`39.68%`
- 当前最明显问题：模型对 `large.cell.carcinoma` 预测偏置明显，腺癌和鳞癌召回率偏低。

## 文献调研结论

结合上级目录 `..\相关论文` 中的参考文献，当前最值得优先借鉴的思路有：

1. 优先启用迁移学习。现有 baseline 未使用预训练权重，而多篇肺部 CT 分类论文都将 transfer learning 作为小样本场景的基础配置。
2. 在 CNN backbone 上加入轻量注意力或全局上下文模块。相比直接换成更重的架构，`SE`、`CBAM`、global block 一类模块更适合当前仓库按增量方式验证。
3. 针对腺癌和鳞癌的高误诊率，应同时尝试类别加权损失、`Focal Loss`、`label smoothing` 或采样策略，而不是只增加训练轮数。
4. 后续实验比较不应只看准确率，还应重点记录 `macro F1`、各类召回率和混淆矩阵变化。

## 目录结构

```text
.
├─ src/
│  └─ main.py
├─ doc/
│  └─ problem2_baseline.md
├─ outputs/
│  └─ problem2_baseline/
├─ README.md
├─ AI_CONTEXT.md
├─ TODO.md
└─ 肺癌疾病诊断图像识别与分类问题.pdf

../
├─ 附件/
│  └─ Data/
├─ LCC/
└─ 相关论文/
```

## 环境依赖

推荐直接使用项目外部同级目录下的 `..\LCC` 虚拟环境。

当前已验证的核心依赖：

- `torch==2.13.0+cpu`
- `torchvision==0.28.0+cpu`
- `timm==1.0.28`
- `numpy==2.2.6`
- `pillow==12.3.0`
- `scikit-learn==1.7.2`
- `scikit-image==0.25.2`

说明：

- 当前环境是 CPU 版 `torch`，训练速度较慢。
- 如果后续需要显著提速，应切换到可用的 GPU 版 PyTorch。

## 数据格式

当前数据位于项目外部目录 `..\附件\Data`，采用三层结构：

```text
../附件/Data/
├─ train/
├─ valid/
└─ test/
```

每个划分目录下按类别子文件夹存储图像。脚本会自动将不同命名形式统一映射为以下四类：

1. `adenocarcinoma`
2. `large.cell.carcinoma`
3. `normal`
4. `squamous.cell.carcinoma`

当前样本数：

- 训练集：`613`
- 验证集：`72`
- 测试集：`315`

## 使用方式

直接在项目根目录运行：

```powershell
..\LCC\Scripts\python.exe .\src\main.py
```

常用参数：

```powershell
..\LCC\Scripts\python.exe .\src\main.py --epochs 20
..\LCC\Scripts\python.exe .\src\main.py --epochs 30
..\LCC\Scripts\python.exe .\src\main.py --pretrained
```

主要参数说明：

- `--epochs`：训练轮数，默认 `25`
- `--batch-size`：批大小，默认 `16`
- `--image-size`：输入尺寸，默认 `224`
- `--lr`：学习率，默认 `3e-4`
- `--weight-decay`：权重衰减，默认 `1e-4`
- `--pretrained`：启用 `timm` 预训练权重

说明：

- `src/main.py` 默认会优先查找项目外部同级目录 `..\附件\Data`。
- 如果外部路径不存在，脚本才会回退到项目内部的 `.\附件\Data`。

## 输出结果

运行完成后，结果会写入 `outputs/problem2_baseline`：

- `best_model.pt`：验证集最佳权重
- `metrics_summary.json`：完整训练记录与指标
- `test_predictions.csv`：测试集逐样本预测结果
- `valid_confusion_matrix.csv`：验证集混淆矩阵
- `test_confusion_matrix.csv`：测试集混淆矩阵

## 相关文档

- 问题二 baseline 说明见 `doc/problem2_baseline.md`
- 相关论文阅读与可迁移思路见 `doc/literature_review.md`
- 阶段进度见 `AI_CONTEXT.md`
- 待办事项见 `TODO.md`
