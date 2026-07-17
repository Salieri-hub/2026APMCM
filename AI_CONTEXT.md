# AI_CONTEXT

## 当前项目目标

项目目标是完成 APMCM 2026 B题“肺癌疾病诊断图像识别与分类问题”的阶段性求解，当前重点包括：

1. 梳理问题一的数据统计分析要求并形成可写入论文的结论。
2. 搭建问题二的可运行 baseline 分类模型。
3. 基于 baseline 结果定位性能瓶颈，为后续优化提供依据。

## 当前整体进度

当前已完成题目理解、数据结构核对、baseline 搭建与一次 `20 epoch` 实跑验证，项目已进入“结果分析与模型优化准备”阶段。

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

## 已修改模块

- `src/main.py`
- `doc/problem2_baseline.md`
- `outputs/problem2_baseline/*`
- `README.md`
- `AI_CONTEXT.md`
- `TODO.md`

## 当前模型 / 算法状态

当前 baseline 配置：

- 框架：PyTorch + timm
- 主干网络：`efficientnet_b0`
- 损失函数：`CrossEntropyLoss`
- 优化器：`AdamW`
- 默认轮数：`25`
- 最近一次验证运行：`20 epoch`
- 设备：CPU
- 预训练：未启用

最近一次运行的核心结果：

- 最佳验证集轮次：`epoch 15`
- 最佳验证集准确率：`62.50%`
- 测试集准确率：`39.68%`
- 测试集 `macro F1`：`0.4427`
- 测试集 `weighted F1`：`0.3663`

当前模型误差特征：

- `normal` 类识别相对稳定。
- `large.cell.carcinoma` 召回率过高，存在明显过预测。
- `adenocarcinoma` 与 `squamous.cell.carcinoma` 大量被误判为 `large.cell.carcinoma`。
- 验证集和测试集之间存在明显泛化落差。

## 当前存在的问题

1. 当前环境为 CPU 版 PyTorch，完整训练耗时较长。
2. baseline 未使用预训练权重，小样本场景下特征学习能力偏弱。
3. 模型对类别分布与类间边界学习不稳定，测试集偏置明显。
4. 现阶段尚未加入学习率调度、类别加权、针对性增强等优化手段。

## 下一步计划

1. 使用 `--pretrained` 重新训练 baseline。
2. 增加学习率调度器，降低后期震荡。
3. 尝试类别加权或焦点损失，重点压低腺癌和鳞癌误诊率。
4. 基于混淆矩阵做误差分析，检查高混淆样本。
5. 视运行资源情况决定是否切换 GPU 环境或缩短迭代试验周期。

## 需要提醒后续协作者的事项

- 后续回答和修改应优先参考本文件，其次是 `README.md` 和 `doc/problem2_baseline.md`。
- `outputs/problem2_baseline/best_model.pt` 对应的是最近一次 `20 epoch` 运行中验证集最佳的 `epoch 15`。
- 当前最新有效结果并不代表最终可提交方案，只能视为 baseline。
- 当前资源布局为“项目代码在 `B题` 目录内，数据集和虚拟环境在项目外部同级目录”。
- 如果继续训练或替换模型，必须同步更新 `AI_CONTEXT.md` 与 `TODO.md`。
