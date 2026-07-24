# 肺癌疾病诊断图像识别与分类问题

本仓库对应《肺癌疾病诊断图像识别与分类问题》的代码与材料。当前 `src/` 目录中的主流程主要面向肺癌图像四分类任务，将输入图像划分为 `adenocarcinoma`、`large.cell.carcinoma`、`normal`、`squamous.cell.carcinoma` 四类；同时提供单模型推理和“主模型 + 专家分支”的级联推理，用于提升易混类别的判别效果。

如果你只需要按评测要求运行代码，直接使用 `src/main.py` 即可。该入口默认读取系统根目录下的 `/testdata`（Windows 通常为 `C:\testdata`），递归扫描全部测试图片，并在当前工作目录生成 `cla_pre.csv`。

## 仓库结构

- `doc/src代码运行说明.docx` / `doc/src代码运行说明.pdf`：原始运行说明
- `src/main.py`：评测/提交入口，负责生成 `cla_pre.csv`
- `src/lcc/submission.py`：单模型与级联推理、CSV 导出逻辑
- `src/lcc/cli.py`：训练与带标签级联评估入口
- `src/outputs/results/.../best_model.pt`：当前仓库附带的默认权重

## 环境要求

- 推荐 Python `3.10+`
- 依赖文件为 `src/requirements.txt`
- 当前压缩包内附带 `venv/`（Python `3.10.0`）；如果不想激活虚拟环境，可将下面示例中的 `python` 替换为：
  - 在仓库根目录执行时：`.\venv\Scripts\python.exe`
  - 在 `src/` 目录执行时：`..\venv\Scripts\python.exe`

安装依赖：

```powershell
cd src
python -m pip install -r .\requirements.txt
```

如果环境使用 `python3` 命令，也可执行：

```powershell
cd src
python3 -m pip install -r .\requirements.txt
```

## 快速开始

### 1. 准备测试数据

默认测试数据目录为系统根目录下的 `/testdata`：

- Windows：`C:\testdata`
- Linux / macOS：`/testdata`

程序会递归读取该目录下的全部测试图片，支持格式：

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tif`
- `.tiff`

如果不想使用默认目录，可以通过 `--input` 显式指定测试集路径。

### 2. 运行评测 / 提交入口

建议先进入 `src/` 目录再执行，这样输出文件会落在 `src/cla_pre.csv`：

```powershell
cd src
```

单模型默认运行：

```powershell
python .\main.py
```

级联默认运行：

```powershell
python .\main.py --run-mode cascade
```

如果环境使用 `python3` 命令，也可以执行：

```powershell
python3 .\main.py
python3 .\main.py --run-mode cascade
```

如果你希望在仓库根目录直接运行，可以使用：

```powershell
python .\src\main.py
python .\src\main.py --run-mode cascade
```

此时输出文件会生成在仓库根目录下的 `.\cla_pre.csv`。

## 输入与输出说明

默认行为如下：

- 输入目录：`/testdata`
- 扫描方式：递归读取全部测试图片
- 输出文件：`./cla_pre.csv`

输出 CSV 的表头为：

```text
image_name,label
```

字段含义如下：

- `image_name`：图片相对于 `testdata` 根目录的相对路径
- `label`：预测类别名称

当前类别标签为：

- `adenocarcinoma`
- `large.cell.carcinoma`
- `normal`
- `squamous.cell.carcinoma`

## 默认模型与结果

### 单模型

`single` 模式默认使用实验 `v3.2_pretrained_focal_ls_cosine_cutmix_b4`。

代码会优先尝试以下权重路径（相对于 `src/`）：

- `outputs/results/v3.2_pretrained_focal_ls_cosine_cutmix_b4/best_model.pt`
- `outputs/weights/v3.2_pretrained_focal_ls_cosine_cutmix_b4/best_model.pt`

根据 `doc/src代码运行说明` 中记录的结果：

- 最佳 epoch：`15`
- 测试集准确率：`0.9428571428571428`，约 `94.29%`

### 级联模型

`cascade` 模式默认组合如下：

- 主模型：`v3.2_pretrained_focal_ls_cosine_cutmix_b4`
- 专家模型：`expert_pair_ad_sq_v3.2_pretrained_focal_ls_cosine_cutmix_b4`

默认触发参数：

- `--expert-trigger-topk 2`
- `--expert-margin-threshold 0.12`

代码会优先尝试以下权重路径（相对于 `src/`）：

- `outputs/results/v3.2_pretrained_focal_ls_cosine_cutmix_b4/best_model.pt`
- `outputs/weights/v3.2_pretrained_focal_ls_cosine_cutmix_b4/best_model.pt`
- `outputs/results/expert_pair_ad_sq_v3.2_pretrained_focal_ls_cosine_cutmix_b4/best_model.pt`
- `outputs/weights/expert_pair_ad_sq_v3.2_pretrained_focal_ls_cosine_cutmix_b4/best_model.pt`

根据 `doc/src代码运行说明` 中记录的结果：

- 级联实验名：`cascade_pair_ad_sq_v3.2_pretrained_focal_ls_cosine_cutmix_b4`
- 测试集准确率：`0.946031746031746`，约 `94.60%`

## `src/main.py` 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--run-mode` | 推理模式，支持 `single` / `cascade` | `single` |
| `--input` | 测试数据目录 | 自动查找 `/testdata` |
| `--checkpoint` | 单模型权重；在 `cascade` 模式下也可作为主模型权重别名使用 | 默认单模型权重 |
| `--main-checkpoint` | 级联主模型权重 | 默认主模型权重 |
| `--expert-checkpoint` | 级联专家模型权重 | 默认专家模型权重 |
| `--output` | 输出 CSV 路径 | `./cla_pre.csv` |
| `--device` | 运行设备，支持 `auto` / `cpu` / `cuda` | `auto` |
| `--amp` | 在 CUDA 推理时启用混合精度 | CUDA 下未显式指定时默认开启 |
| `--no-amp` | 显式关闭混合精度 | - |
| `--expert-trigger-topk` | 级联触发条件之一，要求主模型前 `k` 个类别都落在专家类别集合内 | `2` |
| `--expert-margin-threshold` | 主模型 `top1-top2` 概率差不大于该阈值时触发专家模型 | `0.12` |

兼容别名参数：

- `--testdata-dir` 等价于 `--input`
- `--output-file` 等价于 `--output`

查看当前参数帮助：

```powershell
cd src
python .\main.py --help
```

## 常用运行示例

单模型默认提交：

```powershell
python .\main.py
```

级联默认提交：

```powershell
python .\main.py --run-mode cascade
```

显式指定测试集路径：

```powershell
python .\main.py --input C:\testdata
```

显式指定单模型权重：

```powershell
python .\main.py --checkpoint .\outputs\results\v3.2_pretrained_focal_ls_cosine_cutmix_b4\best_model.pt --output .\cla_pre.csv
```

显式指定级联主模型与专家模型：

```powershell
python .\main.py --run-mode cascade --main-checkpoint .\outputs\results\v3.2_pretrained_focal_ls_cosine_cutmix_b4\best_model.pt --expert-checkpoint .\outputs\results\expert_pair_ad_sq_v3.2_pretrained_focal_ls_cosine_cutmix_b4\best_model.pt --output .\cla_pre.csv
```

CPU 评审示例：

```powershell
python .\main.py --device cpu
python .\main.py --run-mode cascade --device cpu
```

## 训练与带标签级联评估

训练入口为 `src/lcc/cli.py`。由于该文件使用包内相对导入，建议在 `src/` 目录下以模块方式运行：

```powershell
cd src
python -m lcc.cli --help
```

数据目录应指向包含 `train/`、`valid/`、`test/` 三个子目录的数据集根目录。

四分类主模型训练示例：

```powershell
python -m lcc.cli --run-mode single --data-dir <数据集根目录> --output-dir .\outputs\v3.2_pretrained_focal_ls_cosine_cutmix_b4 --model-name efficientnet_b4 --pretrained --loss focal --label-smoothing 0.1 --scheduler cosine --cutmix-alpha 0.5 --epochs 25 --batch-size 16 --device cuda
```

专家分支训练示例：

```powershell
python -m lcc.cli --run-mode expert --data-dir <数据集根目录> --output-dir .\outputs\expert_pair_ad_sq_v3.2_pretrained_focal_ls_cosine_cutmix_b4 --model-name efficientnet_b4 --pretrained --loss focal --label-smoothing 0.1 --scheduler cosine --cutmix-alpha 0.5 --epochs 25 --batch-size 16 --expert-classes adenocarcinoma,squamous.cell.carcinoma --device cuda
```

基于已训练权重执行带标签级联评估：

```powershell
python -m lcc.cli --run-mode cascade --data-dir <数据集根目录> --main-checkpoint .\outputs\weights\<主模型实验名>\best_model.pt --expert-checkpoint .\outputs\weights\<专家模型实验名>\best_model.pt
```

训练输出位置：

- 最优权重：`outputs/weights/<实验名>/best_model.pt`
- 指标汇总：`outputs/results/<实验名>/metrics_summary.json`
- 测试集预测：`outputs/results/<实验名>/test_predictions.csv`

常用训练参数包括：

- `--model-name`：骨干网络，默认 `efficientnet_b4`
- `--pretrained`：启用预训练权重
- `--loss`：损失函数，支持 `cross_entropy` / `focal`
- `--epochs`：训练轮数，默认 `25`
- `--batch-size`：批大小，默认 `16`
- `--image-size`：输入分辨率；B4/B3/B2/B1/B0 默认分别为 `320/288/256/240/224`
- `--lr`：学习率，默认 `3e-4`
- `--weight-decay`：权重衰减，默认 `1e-4`
- `--label-smoothing`：标签平滑，默认 `0.0`
- `--scheduler`：学习率调度器，支持 `none` / `cosine` / `plateau`
- `--mixup-alpha` / `--cutmix-alpha`：MixUp / CutMix 强度
- `--feature-attention`：附加注意力模块，支持 `none` / `se` / `cbam`
- `--device`：运行设备，支持 `auto` / `cpu` / `cuda`
- `--expert-classes`：专家分支使用的类别子集，默认 `adenocarcinoma,squamous.cell.carcinoma`
- `--main-checkpoint` / `--expert-checkpoint`：`cascade` 模式下的主模型 / 专家模型权重

完整参数说明请执行：

```powershell
cd src
python -m lcc.cli --help
```

## 与评审相关的主要文件

- `src/main.py`：评测 / 提交入口
- `src/lcc/submission.py`：单模型与级联推理、`cla_pre.csv` 导出逻辑
- `src/lcc/data.py`：图像预处理
- `src/lcc/models.py`：模型加载与预测
- `src/lcc/runtime.py`：设备与运行时工具
- `src/lcc/cli.py`：训练与带标签级联评估入口
- `src/lcc/train.py`：训练流程
- `src/lcc/cascade.py`：级联推理规则与带标签评估逻辑

## 快速自检

先确认评测入口可用：

```powershell
cd src
python .\main.py --help
```

如需确认训练入口可用：

```powershell
cd src
python -m lcc.cli --help
```

如果命令报错缺少依赖，请先安装：

```powershell
cd src
python -m pip install -r .\requirements.txt
```

## 参考文档

- [题目 PDF](肺癌疾病诊断图像识别与分类问题.pdf)
- [运行说明 PDF](doc/src代码运行说明.pdf)
