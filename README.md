# 2026APMCM B题 Problem 2 基线说明

## 项目概述

本项目用于求解 APMCM B 题 Problem 2：肺癌病理图像四分类任务。

当前代码默认基线为：

- 主干网络：`EfficientNet-B3`
- 默认输入尺寸：`288`
- 支持模式：`single`、`expert`、`cascade`

已经完整验证的历史正式结果仍然是 `B0` 线路；当前仓库默认运行线路已经切换为 `B3`，并准备执行新一轮 `50` 组正式实验。

## 数据任务

主任务类别如下：

- `adenocarcinoma`
- `large.cell.carcinoma`
- `normal`
- `squamous.cell.carcinoma`

支持的实验模式如下：

- `single`：四分类主模型
- `expert`：二分类或三分类子集专家模型
- `cascade`：主模型加专家模型的级联推理

## 当前默认参数

- `--model-name efficientnet_b3`
- `--image-size 288`
- `--device auto/cpu/cuda`
- `--loss cross_entropy` 或 `focal`
- `--scheduler none/cosine/plateau`
- `--feature-attention none/se/cbam`
- `--class-weighting none/balanced/manual`
- `--mixup-alpha` 与 `--cutmix-alpha`

## 输出目录规范

所有实验产物统一写入 `outputs/` 下的共享目录：

- `outputs/weights/<experiment_name>/best_model.pt`
- `outputs/results/<experiment_name>/metrics_summary.json`
- `outputs/results/<experiment_name>/test_predictions.csv`
- `outputs/results/<experiment_name>/valid_confusion_matrix.csv`
- `outputs/results/<experiment_name>/test_confusion_matrix.csv`

级联实验还会额外保存：

- `outputs/results/<experiment_name>/valid_cascade_predictions.csv`
- `outputs/results/<experiment_name>/test_cascade_predictions.csv`

命令行仍然使用：

- `--output-dir .\outputs\<experiment_name>`

代码会自动将权重写入 `outputs/weights/<experiment_name>/`，并将结果写入 `outputs/results/<experiment_name>/`。

## 快速开始

如果当前目录位于 `2026APMCM`：

```powershell
..\LCC_GPU\python.exe .\src\main.py --device cuda --pretrained --model-name efficientnet_b3
```

主模型、专家模型和级联模型的手动运行示例如下：

```powershell
..\LCC_GPU\python.exe .\src\main.py --device cuda --pretrained --model-name efficientnet_b3 --loss focal --label-smoothing 0.1 --scheduler cosine --feature-attention cbam --output-dir .\outputs\v3.4_pretrained_focal_ls_cosine_cbam_b3
..\LCC_GPU\python.exe .\src\main.py --run-mode expert --model-name efficientnet_b3 --expert-classes adenocarcinoma,large.cell.carcinoma,squamous.cell.carcinoma --device cuda --pretrained --loss focal --label-smoothing 0.1 --scheduler cosine --feature-attention cbam --output-dir .\outputs\expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam_b3
..\LCC_GPU\python.exe .\src\main.py --run-mode cascade --device cuda --main-checkpoint .\outputs\weights\v3.4_pretrained_focal_ls_cosine_cbam_b3\best_model.pt --expert-checkpoint .\outputs\weights\expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam_b3\best_model.pt --output-dir .\outputs\cascade_v3.4_pretrained_focal_ls_cosine_cbam_b3
```

## 批量运行 50 组正式实验

脚本文件：

- `scripts/run_all_efficientnet_b3_50.ps1`
- `scripts/run_all_efficientnet_b3_50.cmd`

如果当前目录位于 `2026APMCM`：

```powershell
.\scripts\run_all_efficientnet_b3_50.cmd
```

或者：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_all_efficientnet_b3_50.ps1" -PythonExe "..\LCC_GPU\python.exe"
```

批量脚本会产出：

- `10` 组单模型实验
- `10` 组三分类肿瘤专家级联实验
- `30` 组两两专家级联实验
- 正式输出总数：`50`

## 历史已验证最佳结果

历史 `B0` 单模型最佳结果：

- 实验名：`v3.4_pretrained_focal_ls_cosine_cbam`
- 测试集准确率：`86.35%`
- 测试集 Macro F1：`0.8646`

历史 `B0` 级联最佳结果：

- 实验名：`cascade_v3.4_pretrained_focal_ls_cosine_cbam`
- 测试集准确率：`87.62%`
- 测试集 Macro F1：`0.8773`

## 文档索引

- `doc/problem2_baseline.md`
- `doc/literature_review.md`
- `doc/ablation_results.md`
- `doc/progress_report_2026-07-19.md`
- `AI_CONTEXT.md`
- `TODO.md`
