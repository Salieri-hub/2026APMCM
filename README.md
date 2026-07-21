# 2026APMCM B题 Problem 2 基线说明

## 项目概述

本项目用于求解 APMCM B 题 Problem 2：肺癌病理图像四分类任务。

当前代码默认基线为：

- 主干网络：`EfficientNet-B4`
- 默认输入尺寸：`320`
- 支持模式：`single`、`expert`、`cascade`

当前进度如下：

- `B0` 到 `B4` 的单模型、专家模型和级联实验均已完成并归档到 `outputs/results/`
- 仓库默认运行线路仍为 `B4`
- `src/main.py` 已拆分为轻量入口，核心逻辑位于 `src/lcc/`

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

- `--model-name efficientnet_b4`
- `--image-size 320`
- `--device auto/cpu/cuda`
- `--loss cross_entropy` 或 `focal`
- `--scheduler none/cosine/plateau`
- `--feature-attention none/se/cbam`
- `--class-weighting none/balanced/manual`
- `--mixup-alpha` 与 `--cutmix-alpha`

## 当前最佳结果

以下结果均来自 `outputs/results/`，并以四分类完整测试集（`test=315`）为准。

当前最佳单模型：

- 实验名：`v3.2_pretrained_focal_ls_cosine_cutmix_b4`
- 主干网络：`efficientnet_b4`
- 测试集准确率：`94.29%`
- 测试集 Macro F1：`0.9415`

当前最佳级联结果（全仓库 overall best）：

- 实验名：`cascade_pair_ad_sq_v3.2_pretrained_focal_ls_cosine_cutmix_b4`
- 主模型：`v3.2_pretrained_focal_ls_cosine_cutmix_b4`
- 专家模型：`expert_pair_ad_sq_v3.2_pretrained_focal_ls_cosine_cutmix_b4`
- 触发条件：`top-k=2`，`margin <= 0.12`
- 测试集准确率：`94.60%`
- 测试集 Macro F1：`0.9439`

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

## 代码结构

- `src/main.py`：程序入口，仅调用 `lcc.cli.main()`
- `src/lcc/cli.py`：命令行参数与模式分发
- `src/lcc/config.py`、`constants.py`：配置与常量
- `src/lcc/data.py`、`models.py`、`losses.py`、`train.py`：数据、模型、损失与训练流程
- `src/lcc/cascade.py`、`reporting.py`、`runtime.py`：级联推理、结果导出与运行时工具

## 快速开始

如果当前目录位于 `2026APMCM`：

```powershell
..\LCC_GPU\python.exe .\src\main.py --device cuda --pretrained --model-name efficientnet_b4
```

主模型、专家模型和级联模型的手动运行示例如下：

```powershell
..\LCC_GPU\python.exe .\src\main.py --device cuda --pretrained --model-name efficientnet_b4 --loss focal --label-smoothing 0.1 --scheduler cosine --feature-attention cbam --output-dir .\outputs\v3.4_pretrained_focal_ls_cosine_cbam_b4
..\LCC_GPU\python.exe .\src\main.py --run-mode expert --model-name efficientnet_b4 --expert-classes adenocarcinoma,large.cell.carcinoma,squamous.cell.carcinoma --device cuda --pretrained --loss focal --label-smoothing 0.1 --scheduler cosine --feature-attention cbam --output-dir .\outputs\expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam_b4
..\LCC_GPU\python.exe .\src\main.py --run-mode cascade --device cuda --main-checkpoint .\outputs\weights\v3.4_pretrained_focal_ls_cosine_cbam_b4\best_model.pt --expert-checkpoint .\outputs\weights\expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam_b4\best_model.pt --output-dir .\outputs\cascade_v3.4_pretrained_focal_ls_cosine_cbam_b4
```

## B4 批量脚本（已完成，可复现）

脚本文件：

- `scripts/run_all_efficientnet_b4_50.ps1`
- `scripts/run_all_efficientnet_b4_50.cmd`

如果当前目录位于 `2026APMCM`：

```powershell
.\scripts\run_all_efficientnet_b4_50.cmd
```

或者：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_all_efficientnet_b4_50.ps1" -PythonExe "..\LCC_GPU\python.exe"
```

该批量脚本已经完成过一轮正式运行，产出：

- `10` 组单模型实验
- `10` 组三分类肿瘤专家级联实验
- `30` 组两两专家级联实验
- 正式输出总数：`50`

## 文档索引

- `doc/problem2_baseline.md`
- `doc/literature_review.md`
- `doc/ablation_results.md`
- `doc/progress_report_2026-07-19.md`
- `AI_CONTEXT.md`
- `TODO.md`
