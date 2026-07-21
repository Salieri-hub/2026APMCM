# `src` 代码运行说明

本文档面向评审老师，说明当前版本 `src/` 代码的用途、运行环境、数据组织方式，以及三种运行模式的执行方法。

## 1. 代码功能概述

当前程序用于 APMCM B 题 Problem 2 的肺癌病理图像分类，支持以下三种模式：

- `single`：四分类主模型训练与评估
- `expert`：对子类别子集训练专家模型
- `cascade`：使用“主模型 + 专家模型”进行级联推理评估

程序入口文件为：

```text
src/main.py
```

从仓库根目录运行时，统一使用：

```powershell
python .\src\main.py [参数]
```

如果本机有多个 Python 环境，请将上面的 `python` 替换为实际解释器完整路径。

## 2. 运行前准备

### 2.1 建议目录

以下命令默认在项目根目录 `2026APMCM/` 下执行。

### 2.2 依赖安装

项目根目录已提供 `requirements.txt`。建议先执行：

```powershell
pip install -r .\requirements.txt
```

当前 `src/lcc/models.py` 还会用到 `safetensors`。如果安装完依赖后运行时报错 `No module named 'safetensors'`，请补充安装：

```powershell
pip install safetensors
```

说明：

- `requirements.txt` 中给出的是已验证的 CUDA 版 PyTorch 依赖。
- 如果评审机器只使用 CPU，也可以自行安装与本机兼容的 `torch`、`torchvision`，再安装 `timm`、`numpy`、`Pillow`、`scikit-learn`、`safetensors`。

### 2.3 快速检查程序是否可启动

```powershell
python .\src\main.py --help
```

如果能正常打印参数说明，说明程序入口可用。

## 3. 数据目录要求

建议评审时始终显式传入 `--data-dir`，不要依赖程序内部的默认路径探测。

程序要求数据目录按以下结构组织：

```text
数据目录/
  train/
    adenocarcinoma/
    large.cell.carcinoma/
    normal/
    squamous.cell.carcinoma/
  valid/
    adenocarcinoma/
    large.cell.carcinoma/
    normal/
    squamous.cell.carcinoma/
  test/
    adenocarcinoma/
    large.cell.carcinoma/
    normal/
    squamous.cell.carcinoma/
```

每个类别文件夹中放图像文件，支持：

- `.png`
- `.jpg`
- `.jpeg`

程序识别的标准类别为：

- `adenocarcinoma`
- `large.cell.carcinoma`
- `normal`
- `squamous.cell.carcinoma`

其中 `adenocarcinoma`、`large.cell.carcinoma`、`squamous.cell.carcinoma` 这三类文件夹允许以标准类名开头再附加后缀；`normal` 建议直接命名为 `normal`。

## 4. 三种运行模式的使用方法

### 4.1 `single`：四分类主模型

这是默认模式，可不写 `--run-mode single`。

最小示例：

```powershell
python .\src\main.py --data-dir "D:\Data" --output-dir .\outputs\demo_single
```

推荐示例：

```powershell
python .\src\main.py `
  --run-mode single `
  --data-dir "D:\Data" `
  --model-name efficientnet_b4 `
  --pretrained `
  --device cuda `
  --epochs 25 `
  --batch-size 16 `
  --loss focal `
  --label-smoothing 0.1 `
  --scheduler cosine `
  --feature-attention cbam `
  --output-dir .\outputs\v3.4_pretrained_focal_ls_cosine_cbam_b4
```

用途说明：

- 训练一个四分类模型
- 在验证集上选择最佳模型
- 在测试集上输出预测结果和混淆矩阵

### 4.2 `expert`：专家模型

该模式只在指定类别子集上训练模型。类别用逗号分隔。

二分类专家示例：

```powershell
python .\src\main.py `
  --run-mode expert `
  --data-dir "D:\Data" `
  --model-name efficientnet_b4 `
  --expert-classes adenocarcinoma,squamous.cell.carcinoma `
  --pretrained `
  --device cuda `
  --epochs 25 `
  --batch-size 16 `
  --loss focal `
  --label-smoothing 0.1 `
  --scheduler cosine `
  --output-dir .\outputs\expert_ad_sq_demo
```

三分类专家示例：

```powershell
python .\src\main.py `
  --run-mode expert `
  --data-dir "D:\Data" `
  --model-name efficientnet_b4 `
  --expert-classes adenocarcinoma,large.cell.carcinoma,squamous.cell.carcinoma `
  --pretrained `
  --device cuda `
  --output-dir .\outputs\expert_tumor3_demo
```

用途说明：

- 从四分类数据中筛出指定类别
- 只在该子集上训练一个专家模型
- 该专家模型可供后续 `cascade` 模式使用

### 4.3 `cascade`：级联推理

该模式不再训练，而是读取两个现成的检查点：

- 一个主模型检查点
- 一个专家模型检查点

示例：

```powershell
python .\src\main.py `
  --run-mode cascade `
  --data-dir "D:\Data" `
  --device cuda `
  --main-checkpoint .\outputs\weights\v3.4_pretrained_focal_ls_cosine_cbam_b4\best_model.pt `
  --expert-checkpoint .\outputs\weights\expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam_b4\best_model.pt `
  --expert-trigger-topk 2 `
  --expert-margin-threshold 0.12 `
  --output-dir .\outputs\cascade_v3.4_pretrained_focal_ls_cosine_cbam_b4
```

用途说明：

- 主模型先对样本做四分类预测
- 当主模型满足触发条件时，再调用专家模型细分判断
- 最终输出级联后的验证集和测试集结果

## 5. 关键参数说明

以下是当前版本最常用的参数：

- `--data-dir`：数据目录，建议每次显式指定
- `--output-dir`：实验输出目录，建议写成 `.\outputs\实验名`
- `--run-mode`：`single`、`expert`、`cascade`
- `--model-name`：默认 `efficientnet_b4`
- `--pretrained`：使用预训练权重
- `--device`：`auto`、`cpu`、`cuda`
- `--epochs`：训练轮数，默认 `25`
- `--batch-size`：默认 `16`
- `--image-size`：输入尺寸；`efficientnet_b4` 默认 `320`
- `--loss`：`cross_entropy` 或 `focal`
- `--label-smoothing`：标签平滑系数
- `--scheduler`：`none`、`cosine`、`plateau`
- `--feature-attention`：`none`、`se`、`cbam`
- `--class-weighting`：`none`、`balanced`、`manual`
- `--mixup-alpha`：大于 `0` 时启用 MixUp
- `--cutmix-alpha`：大于 `0` 时启用 CutMix
- `--expert-classes`：专家模型对应的类别子集
- `--main-checkpoint`：级联模式主模型权重路径
- `--expert-checkpoint`：级联模式专家模型权重路径
- `--expert-trigger-topk`：触发专家模型的 top-k 规则
- `--expert-margin-threshold`：主模型 top1 与 top2 概率差阈值；若设为负数则关闭该过滤条件

## 6. 输出结果在哪里

无论是训练还是级联，只要传入：

```powershell
--output-dir .\outputs\实验名
```

程序都会自动整理为以下结构：

```text
outputs/
  weights/
    实验名/
      best_model.pt
  results/
    实验名/
      metrics_summary.json
      test_predictions.csv
      valid_confusion_matrix.csv
      test_confusion_matrix.csv
```

其中：

- `best_model.pt`：最佳模型权重
- `metrics_summary.json`：本次运行的完整摘要，包括参数、数据集规模、验证集和测试集指标
- `test_predictions.csv`：测试集逐样本预测结果
- `valid_confusion_matrix.csv`、`test_confusion_matrix.csv`：混淆矩阵

如果是 `cascade` 模式，还会额外生成：

- `valid_cascade_predictions.csv`
- `test_cascade_predictions.csv`

## 7. 批量脚本如何使用

项目中已提供 B2、B3、B4 的批量脚本，位于：

- `scripts/run_all_efficientnet_b2_50.ps1`
- `scripts/run_all_efficientnet_b3_50.ps1`
- `scripts/run_all_efficientnet_b4_50.ps1`

其中 `.cmd` 文件只是对 `.ps1` 的一层包装。当前 `.cmd` 默认写死了 `..\LCC_GPU\python.exe`，因此在其他机器上更推荐直接运行 `.ps1`，并手动传入 Python 解释器绝对路径。

示例：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_all_efficientnet_b4_50.ps1" `
  -PythonExe "C:\Python310\python.exe"
```

脚本会自动完成：

- 主模型训练
- 三分类专家模型训练
- 二分类专家模型训练
- 对应的级联评估

## 8. 评审时最建议的三条命令

如果只需要快速看懂并验证程序，可按下面顺序执行。

### 8.1 查看全部参数

```powershell
python .\src\main.py --help
```

### 8.2 运行一次四分类主模型

```powershell
python .\src\main.py `
  --data-dir "D:\Data" `
  --pretrained `
  --device cuda `
  --output-dir .\outputs\review_single_demo
```

### 8.3 在已有主模型和专家模型基础上运行一次级联

```powershell
python .\src\main.py `
  --run-mode cascade `
  --data-dir "D:\Data" `
  --device cuda `
  --main-checkpoint .\outputs\weights\review_single_demo\best_model.pt `
  --expert-checkpoint .\outputs\weights\expert_ad_sq_demo\best_model.pt `
  --output-dir .\outputs\review_cascade_demo
```

## 9. 常见说明

- 如果机器没有可用 GPU，请把 `--device cuda` 改为 `--device cpu`。
- 如果首次启用 `--pretrained`，程序可能会自动下载预训练权重到 `.cache/` 目录。
- `cascade` 模式必须先准备好主模型和专家模型的 `best_model.pt`。
- 为了避免不同机器上的默认路径差异，评审时最稳妥的做法是始终显式传入 `--data-dir` 与 `--output-dir`。

## 10. 一句话总结

对评审老师而言，最简单的理解方式是：

- `single`：训练一个四分类模型
- `expert`：训练一个只区分部分类别的专家模型
- `cascade`：先用四分类模型判断，再在必要时交给专家模型细分

只要数据目录按本文档组织，并从项目根目录执行 `python .\src\main.py ...`，即可复现当前版本 `src` 代码的完整运行流程。
