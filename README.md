# APMCM 2026 B题

肺癌疾病诊断图像识别与分类问题的本地实验项目。

## 项目目标

本项目当前围绕题目中的两个任务推进：

1. 问题一：完成数据集规模、比例和分层抽样合理性的统计分析。
2. 问题二：搭建四分类肺部 CT 图像识别模型，并系统比较不同优化策略。
3. 在四分类主模型基础上，引入专家模型与级联触发机制，尽量降低三种肿瘤亚型之间的误诊率。

## 当前状态

- 问题一的题意分析、数据统计口径和准确率公式计算已完成。
- `src/main.py` 已从单一 baseline 脚本扩展为完整实验管线，支持 `single`、`expert`、`cascade` 三种运行模式。
- 已完成 `12` 组单模型正式实验、`10` 组三肿瘤专家级联实验和 `30` 组两两肿瘤专家级联实验，共 `52` 组正式结果对比。
- 当前最佳单模型为 `v3.4_pretrained_focal_ls_cosine_cbam`：
  - 测试集准确率：`86.35%`
  - 测试集 `macro F1`：`0.8646`
- 当前全局最佳结果为 `cascade_v3.4_pretrained_focal_ls_cosine_cbam`：
  - 测试集准确率：`87.62%`
  - 测试集 `macro F1`：`0.8773`
- 三肿瘤专家级联在 `10` 个主模型版本中有 `8` 个优于各自对应的单模型；两两肿瘤专家级联在 `30` 组中有 `21` 组优于各自对应的单模型。
- 当前最稳定的结构主线是：`pretrained EfficientNet-B0 + label smoothing / focal loss + CBAM + selective cascade`。

## 文献调研结论

结合上级目录 `..\相关论文(1)` 中的参考文献，当前已验证或值得优先借鉴的思路有：

1. 迁移学习是小样本肺部 CT 分类中最重要的低成本改进项。
2. `EfficientNet` 路线适合作为当前项目的统一 backbone，便于控制变量和开展消融实验。
3. `label smoothing`、`Focal Loss`、轻量注意力模块比单纯增加训练轮数更能改善泛化。
4. `CBAM` 在当前数据上明显优于额外 `SE`，说明空间注意力比继续叠加通道重标定更有效。
5. 为了解决腺癌、鳞癌和大细胞癌之间的边界混淆，仅靠单模型不够，专家级联是值得保留的工程化路线。

## 目录结构

```text
.
├─ src/
│  └─ main.py
├─ scripts/
│  ├─ run_all_cascade_tumor3.ps1
│  ├─ run_all_cascade_tumor_pairs.ps1
│  ├─ generate_version_comparison_docx.py
│  ├─ generate_outputs_version_ablation_docx.py
│  ├─ generate_52_experiment_comparison_docx.py
│  └─ generate_ablation_results_cn_docx.py
├─ doc/
│  ├─ problem2_baseline.md
│  ├─ literature_review.md
│  ├─ ablation_results.md
│  └─ progress_report_2026-07-19.md
├─ outputs/
│  ├─ v1.0_scratch_ce_cpu/
│  ├─ v3.4_pretrained_focal_ls_cosine_cbam/
│  ├─ expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam/
│  ├─ cascade_v3.4_pretrained_focal_ls_cosine_cbam/
│  ├─ cascade_pair_lc_sq_v3.4_pretrained_focal_ls_cosine_cbam/
│  └─ cascade_batch_logs/
├─ README.md
├─ AI_CONTEXT.md
├─ TODO.md
└─ 肺癌疾病诊断图像识别与分类问题.pdf

../
├─ 附件/
│  └─ Data/
├─ LCC/
├─ LCC310/
├─ LCC_GPU/
└─ 相关论文(1)/
```

## 环境依赖

推荐优先使用已经准备好的 `..\LCC_GPU` CUDA 环境。

截至 `2026-07-19`，本地已验证可用的核心依赖为：

- `torch==2.13.0+cu130`
- `torchvision==0.28.0+cu130`
- `timm==1.0.28`
- `numpy==2.2.6`
- `pillow==12.3.0`
- `scikit-learn==1.7.2`
- `scikit-image==0.25.2`

说明：

- `..\LCC_GPU` 已在本机验证通过：`torch.cuda.is_available() == True`。
- 当前主机可识别 GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`。
- `..\LCC` 目录不再作为推荐训练环境；正式 GPU 实验优先使用 `..\LCC_GPU`。
- `outputs` 中的正式结果目录可能来自不同 CUDA 运行环境，但统一以各自目录下的 `metrics_summary.json` 为准。

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

在项目根目录运行：

```powershell
..\LCC_GPU\python.exe .\src\main.py
```

常用运行方式：

```powershell
..\LCC_GPU\python.exe .\src\main.py --device cuda --pretrained
..\LCC_GPU\python.exe .\src\main.py --device cuda --pretrained --loss focal --label-smoothing 0.1 --scheduler cosine --feature-attention cbam --output-dir .\outputs\v3.4_pretrained_focal_ls_cosine_cbam
..\LCC_GPU\python.exe .\src\main.py --run-mode expert --expert-classes adenocarcinoma,large.cell.carcinoma,squamous.cell.carcinoma --device cuda --pretrained --loss focal --label-smoothing 0.1 --scheduler cosine --feature-attention cbam --output-dir .\outputs\expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam
..\LCC_GPU\python.exe .\src\main.py --run-mode cascade --device cuda --main-checkpoint .\outputs\v3.4_pretrained_focal_ls_cosine_cbam\best_model.pt --expert-checkpoint .\outputs\expert_tumor3_v3.4_pretrained_focal_ls_cosine_cbam\best_model.pt --output-dir .\outputs\cascade_v3.4_pretrained_focal_ls_cosine_cbam
```

批量脚本：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_all_cascade_tumor3.ps1" -PythonExe "..\LCC_GPU\python.exe"
powershell -ExecutionPolicy Bypass -File ".\scripts\run_all_cascade_tumor_pairs.ps1" -PythonExe "..\LCC_GPU\python.exe"
```

主要参数说明：

- `--run-mode`：支持 `single / expert / cascade`
- `--model-name`：当前默认 `efficientnet_b0`
- `--pretrained`：启用 `timm` 预训练权重
- `--loss`：支持 `cross_entropy / focal`
- `--label-smoothing`：标签平滑系数
- `--scheduler`：支持 `none / cosine / plateau`
- `--feature-attention`：支持 `none / se / cbam`
- `--class-weighting`：支持 `none / balanced / manual`
- `--mixup-alpha`、`--cutmix-alpha`：混合增强参数
- `--expert-classes`：专家模型负责的类别子集
- `--expert-trigger-topk`、`--expert-margin-threshold`：级联触发阈值

## 输出结果

单模型或专家模型训练完成后，输出目录通常包含：

- `best_model.pt`
- `metrics_summary.json`
- `test_predictions.csv`
- `valid_confusion_matrix.csv`
- `test_confusion_matrix.csv`

级联评估输出目录通常包含：

- `metrics_summary.json`
- `valid_cascade_predictions.csv`
- `test_cascade_predictions.csv`
- `valid_confusion_matrix.csv`
- `test_confusion_matrix.csv`

正式实验目录命名约定：

- `v1.x`：scratch + CE 系列
- `v2.x`：pretrained + CE 系列
- `v3.x`：pretrained + focal / attention 系列
- `expert_tumor3_*`：三肿瘤专家模型
- `expert_pair_*`：两两肿瘤专家模型
- `cascade_*`：三肿瘤专家级联结果
- `cascade_pair_*`：两两肿瘤专家级联结果

## 相关文档

- 问题二实验说明见 `doc/problem2_baseline.md`
- 文献整理见 `doc/literature_review.md`
- 12 组单模型消融汇总见 `doc/ablation_results.md`
- 阶段进展与协作上下文见 `AI_CONTEXT.md`
- 待办事项见 `TODO.md`
- 52 组结果的 Word 汇总见 `doc/all_52_experiments_comparison_20260719.docx`
- 各版本说明与消融分析 Word 汇总见 `doc/outputs_实验版本说明与消融对比_20260719.docx`
