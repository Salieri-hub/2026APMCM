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
- 设备策略：`--device auto`，有可用 CUDA 时优先使用 GPU
- CUDA 优化：自动混合精度 AMP、`pin_memory`、自动 `num_workers`
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
..\LCC_GPU\python.exe .\src\main.py
```

常见运行方式：

```powershell
..\LCC_GPU\python.exe .\src\main.py --epochs 20
..\LCC_GPU\python.exe .\src\main.py --epochs 30
..\LCC_GPU\python.exe .\src\main.py --pretrained
..\LCC_GPU\python.exe .\src\main.py --device cuda --pretrained --batch-size 32 --num-workers 4
..\LCC_GPU\python.exe .\src\main.py --device cuda --no-amp
```

路径说明：

- 当前项目代码位于 `B题` 目录内。
- 数据集位于项目外部同级目录 `..\附件\Data`。
- 推荐 GPU 虚拟环境位于项目外部同级目录 `..\LCC_GPU`。
- `src/main.py` 默认会优先使用 `..\附件\Data`，兼容当前目录布局。
- 当前机器已经检测到 `NVIDIA GeForce RTX 4060 Laptop GPU`，且截至 `2026-07-18`，`..\LCC_GPU` 已验证 `torch==2.13.0+cu130`、`torchvision==0.28.0+cu130` 可正常识别 CUDA。

## 6. 输出结果

脚本会在输出目录下生成：

- CPU 默认输出：`outputs/problem2_baseline`
- GPU 默认输出：`outputs/problem2_baseline_gpu`

- `best_model.pt`：验证集准确率最高的模型权重
- `metrics_summary.json`：训练过程与最终指标汇总
- `test_predictions.csv`：测试集逐样本预测结果
- `valid_confusion_matrix.csv`：验证集混淆矩阵
- `test_confusion_matrix.csv`：测试集混淆矩阵

## 7. 当前最新实验结果与 GPU 状态

截至 `2026-07-18`，当前最新已完成检查的运行配置为 GPU 版本：

- `epochs=25`
- `batch_size=16`
- `image_size=224`
- `lr=3e-4`
- `weight_decay=1e-4`
- `pretrained=true`
- `device=cuda`

对应结果：

- 最佳验证集轮次：`epoch 23`
- 最佳验证集准确率：`91.67%`
- 测试集准确率：`76.19%`
- 测试集 `macro F1`：`0.7721`
- 测试集 `weighted F1`：`0.7656`

当前 GPU 状态：

- 主机显卡：`NVIDIA GeForce RTX 4060 Laptop GPU`
- 驱动版本：`581.80`
- `nvidia-smi` 可正常识别 GPU
- 已验证环境：`..\LCC_GPU`
- 已验证依赖：`torch==2.13.0+cu130`、`torchvision==0.28.0+cu130`
- 已完成一次正式 `25 epoch` GPU baseline 训练
- 当前结论：代码与环境都已具备 GPU 运行条件，且 `pretrained` 显著优于旧 CPU 非预训练 baseline

测试集各类别召回率：

- `adenocarcinoma`：`71.67%`
- `large.cell.carcinoma`：`92.16%`
- `normal`：`98.15%`
- `squamous.cell.carcinoma`：`60.00%`

## 8. 结果解读

当前 baseline 已经从“可运行”提升到了“有明显竞争力”的阶段，但模型效果还不能直接视为最终方案，主要原因有：

1. 虽然测试集准确率已提升到 `76.19%`，但 `large.cell.carcinoma` 仍存在过预测。
2. 题目重点关注的腺癌与鳞癌误诊问题已经明显缓解，但鳞癌召回率仍有提升空间。
3. 当前最优模型仍只基于 `EfficientNet-B0 + CrossEntropyLoss + AdamW`，还没有加入更针对性的损失与采样优化。

从测试集混淆情况看：

- 腺癌 `120` 张中有 `86` 张预测正确，仍有 `25` 张被判成了大细胞癌。
- 鳞癌 `90` 张中有 `54` 张预测正确，仍有 `19` 张被判成了大细胞癌、`16` 张被判成腺癌。
- 正常肺部样本识别非常稳定，`54` 张中有 `53` 张预测正确。

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

1. 在当前 `GPU + pretrained` 基线之上引入学习率调度器与 `label smoothing`。
2. 对腺癌与鳞癌尝试类别加权损失、`Focal Loss` 或采样策略。
3. 在当前 `EfficientNet` baseline 上增量试验轻量注意力模块。
4. 加强针对易混类别的图像增强策略。
5. 基于混淆矩阵开展误差分析，重点检查鳞癌与大细胞癌的混淆样本，必要时补充 `Grad-CAM`。
