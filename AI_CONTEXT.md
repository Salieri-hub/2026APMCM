# AI_CONTEXT

## 当前项目目标

项目目标是完成 APMCM 2026 B题“肺癌疾病诊断图像识别与分类问题”的阶段性求解，当前重点包括：

1. 梳理问题一的数据统计分析要求并形成可写入论文的结论。
2. 搭建问题二的可运行 baseline 分类模型。
3. 基于 baseline 结果定位性能瓶颈，为后续优化提供依据。

## 当前整体进度

当前已完成题目理解、数据结构核对、baseline 搭建、一次 `20 epoch` CPU 实跑验证、一轮相关文献初步阅读、训练脚本的 GPU 化改造、独立 GPU 环境 `..\LCC_GPU` 的建立与验证、一次正式 `25 epoch` GPU baseline 训练，以及一次加入 `label smoothing + cosine scheduler` 的对比实验。项目已进入“围绕当前最优测试结果继续做类别边界优化”阶段。

## 本次已完成内容

- 阅读赛题 PDF，完成问题一的任务拆解和核心计算结论。
- 识别并核对项目外部目录 `..\附件\Data` 下 `train/valid/test` 的实际目录结构与类别映射。
- 在项目外部目录 `..\LCC` 虚拟环境下补齐 baseline 所需依赖。
- 实现 `src/main.py`：
  - 自动读取数据集
  - 统一四类标签
  - 构建 `EfficientNet-B0`
  - 使用 `CrossEntropyLoss + AdamW`
  - 按验证集准确率保存最佳模型
  - 输出测试集预测和混淆矩阵
- 生成并更新问题二说明文档。
- 跑通 `20 epoch` baseline，并完成结果检查。
- 初步阅读项目上级目录 `..\相关论文` 中的参考文献，并结合公开摘要归纳后续可迁移的优化路线。
- 核对本机 GPU 状态，确认当前主机存在 `NVIDIA GeForce RTX 4060 Laptop GPU`，驱动版本 `581.80`。
- 将 `src/main.py` 改为 GPU 优先版本，增加：
  - `--device auto / cpu / cuda`
  - CUDA 自动 AMP 混合精度
  - GPU 场景下自动 `num_workers`
  - `pin_memory`、`non_blocking`、`persistent_workers`
  - GPU 默认独立输出目录 `outputs/problem2_baseline_gpu`
- 基于未改动的 `..\LCC310` 复制生成 `..\LCC_GPU`，并安装 CUDA 版 `torch==2.13.0+cu130`、`torchvision==0.28.0+cu130`。
- 使用 `..\LCC_GPU\python.exe` 完成一次最小 GPU smoke test，确认 CUDA 训练路径可用。
- 使用 `..\LCC_GPU\python.exe .\src\main.py --device cuda --pretrained` 完成一次正式 `25 epoch` 训练。
- 在 `src/main.py` 中加入 `label smoothing`、学习率调度器和每轮学习率记录。
- 使用 `..\LCC_GPU\python.exe .\src\main.py --device cuda --pretrained --label-smoothing 0.1 --scheduler cosine` 完成一轮对比实验。

## 已修改模块

- `src/main.py`
- `README.md`
- `doc/problem2_baseline.md`
- `AI_CONTEXT.md`
- `TODO.md`
- `..\LCC_GPU\*`

## 当前模型 / 算法状态

当前 baseline 配置：

- 框架：PyTorch + timm
- 主干网络：`efficientnet_b0`
- 损失函数：`CrossEntropyLoss`，现支持 `label smoothing`
- 优化器：`AdamW`
- 学习率调度：支持 `none / cosine / plateau`
- 默认轮数：`25`
- 最近一次已验证运行：`25 epoch`
- 设备策略：`auto -> cuda if available else cpu`
- CUDA 优化：AMP、自动 `num_workers`、`pin_memory`
- 预训练：已启用

当前最新对比实验中，测试集表现最好的核心结果：

- 最佳验证集轮次：`epoch 23`
- 最佳验证集准确率：`90.28%`
- 测试集准确率：`77.46%`
- 测试集 `macro F1`：`0.7894`
- 测试集 `weighted F1`：`0.7705`
- 配置：`pretrained + label_smoothing=0.1 + cosine scheduler`
- 总训练耗时：约 `138.9s`

当前运行环境状态：

- 主机 GPU 已存在并可被 `nvidia-smi` 识别。
- 截至 `2026-07-18`，`..\LCC_GPU` 已验证 `torch.cuda.is_available() == True`。
- 当前最新完整训练指标已经来自 GPU 调参实验。

当前模型误差特征：

- `normal` 类识别相对稳定。
- `large.cell.carcinoma` 的过预测相比上一轮有所缓解。
- `squamous.cell.carcinoma` 召回率明显提高，但部分 `adenocarcinoma` 被转移误判为 `squamous.cell.carcinoma`。
- 相比旧 CPU baseline，泛化落差已明显缩小，但腺癌与鳞癌的边界仍未完全稳定。

## 文献调研结论

本轮文献梳理后，当前可直接迁移到本项目的结论如下：

1. `transfer learning` 是当前最应优先落地的低成本优化项，现有 baseline 未启用预训练权重，和文献中的常见做法相比明显偏弱。
2. 注意力机制是比“直接换重模型”更稳妥的下一步，优先级应放在 `SE`、`CBAM` 或轻量 global context 模块，而不是立刻全面改写为 Transformer 方案。
3. 腺癌与鳞癌被误判为大细胞癌的问题，本质上更接近类间边界不清和训练偏置，宜优先尝试类别加权损失、`Focal Loss`、`label smoothing`、采样策略与误差样本分析。
4. 后续实验汇报不应只看验证集准确率，应同步比较 `macro F1`、各类别召回率和混淆矩阵。

## 当前存在的问题

1. `label smoothing + cosine scheduler` 虽然提升了测试集准确率和 `macro F1`，但腺癌召回率下降，需要继续平衡类别取舍。
2. 现阶段尚未加入类别加权、`Focal Loss`、采样策略、注意力模块等更针对性的优化手段。
3. 当前最佳模型仍以验证集准确率为主要保存依据，实验评价维度还可以继续扩展。

## 下一步计划

1. 在当前 `label smoothing + cosine scheduler` 基线上尝试类别加权、`Focal Loss` 或采样策略，优先拉回腺癌召回率。
2. 试验轻量注意力模块，如 `SE`、`CBAM` 或其他全局上下文增强模块。
3. 基于混淆矩阵与高频误判样本开展误差分析，重点检查“腺癌 -> 鳞癌”的新误判路径。
4. 视结果决定是否保留 `label smoothing`，或改为“类别加权 + scheduler”的组合。

## 需要提醒后续协作者的事项

- 后续回答和修改应优先参考本文件，其次是 `README.md` 和 `doc/problem2_baseline.md`。
- 文献阅读结论的细化整理见 `doc/literature_review.md`。
- `outputs/problem2_baseline/best_model.pt` 对应的是最近一次 `20 epoch` 运行中验证集最佳的 `epoch 15`。
- 后续 GPU 实验默认应使用 `..\LCC_GPU\python.exe`，并写入新的输出目录，避免覆盖现有 `problem2_baseline_gpu` 与 `problem2_pretrained_ls_cosine` 结果。
- 当前最新有效结果并不代表最终可提交方案，只能视为 baseline。
- 当前资源布局为“项目代码在 `B题` 目录内，数据集和虚拟环境在项目外部同级目录”。
- 如果继续训练或替换模型，必须同步更新 `AI_CONTEXT.md` 与 `TODO.md`。
