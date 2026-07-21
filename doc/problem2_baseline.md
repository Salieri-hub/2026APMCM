# Problem 2 基线方案

## 1. 任务定义

本项目解决 APMCM Problem 2 的肺癌病理图像四分类任务。

类别如下：

- `adenocarcinoma`
- `large.cell.carcinoma`
- `normal`
- `squamous.cell.carcinoma`

主评价目标是完整四分类任务。专家模型用于特定易混类别子集，并在级联模式中辅助主模型。

## 2. 当前基线线路

当前仓库默认基线已经切换为：

- 主干网络：`EfficientNet-B4`
- 默认输入尺寸：`320`
- 优化器：`AdamW`
- 损失函数：`cross_entropy` 与 `focal`
- 学习率调度：`none`、`cosine`、`plateau`
- 可选增强项：`label smoothing`、`MixUp`、`CutMix`、`SE`、`CBAM`

## 3. 支持的运行模式

### 3.1 Single

用于四分类主模型训练与评估。

### 3.2 Expert

用于训练类别子集专家模型。

当前支持的专家子集包括：

- 三分类肿瘤专家：`adenocarcinoma,large.cell.carcinoma,squamous.cell.carcinoma`
- 两两专家一：`adenocarcinoma,large.cell.carcinoma`
- 两两专家二：`adenocarcinoma,squamous.cell.carcinoma`
- 两两专家三：`large.cell.carcinoma,squamous.cell.carcinoma`

### 3.3 Cascade

级联模式先由主模型预测，再决定是否进入专家分支。触发条件如下：

1. 主模型的 top-k 类别必须全部落在专家子集内。
2. `top1 - top2 <= expert_margin_threshold`。

专家模型不会直接硬覆盖主模型输出，而是只在专家子集内部重新分配概率并返回最终四分类结果。

## 4. 正式 B4 实验计划

本轮实验沿用此前非 CPU、预训练线路的正式设置，并将其整体迁移到 `B4`。

明确排除：

- `v1.0_scratch_ce_cpu`
- `v1.1_scratch_ce_cuda`

计划产出如下：

- `10` 组单模型实验
- `10` 组三分类肿瘤专家级联实验
- `30` 组两两专家级联实验
- 总计：`50`

## 5. 输出结构

所有产物统一保存在 `outputs/` 下的共享目录：

- `outputs/weights/<experiment_name>/best_model.pt`
- `outputs/results/<experiment_name>/metrics_summary.json`
- `outputs/results/<experiment_name>/test_predictions.csv`
- `outputs/results/<experiment_name>/valid_confusion_matrix.csv`
- `outputs/results/<experiment_name>/test_confusion_matrix.csv`
- 级联实验额外保存：
- `outputs/results/<experiment_name>/valid_cascade_predictions.csv`
- `outputs/results/<experiment_name>/test_cascade_predictions.csv`

命令行仍然使用：

- `--output-dir .\outputs\<experiment_name>`

代码会自动将实验名映射到共享的 `weights/` 与 `results/` 目录。

## 6. 运行命令

如果当前目录位于 `2026APMCM`：

```powershell
.\scripts\run_all_efficientnet_b4_50.cmd
```

或者：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_all_efficientnet_b4_50.ps1" -PythonExe "..\LCC_GPU\python.exe"
```

手动执行级联实验的示例如下：

```powershell
..\LCC_GPU\python.exe .\src\main.py --run-mode cascade --device cuda --main-checkpoint .\outputs\weights\v3.4_pretrained_focal_ls_cosine_cbam_b4\best_model.pt --expert-checkpoint .\outputs\weights\expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam_b4\best_model.pt --output-dir .\outputs\cascade_v3.4_pretrained_focal_ls_cosine_cbam_b4
```

## 7. 历史参考结果

历史 `B0` 单模型最佳结果：

- 实验名：`v3.4_pretrained_focal_ls_cosine_cbam`
- 测试集准确率：`86.35%`
- Macro F1：`0.8646`

历史 `B0` 级联最佳结果：

- 实验名：`cascade_v3.4_pretrained_focal_ls_cosine_cbam`
- 测试集准确率：`87.62%`
- Macro F1：`0.8773`

当前 `B4` 线路已经准备完成，但还没有完成正式 `50` 组复现实验。
