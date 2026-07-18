# TODO

## 已完成任务

- [x] 阅读赛题 PDF 并拆解问题一与问题二要求
- [x] 统计并核对数据集目录结构与样本规模
- [x] 在项目外部的 `..\LCC` 环境中搭建问题二 baseline 依赖
- [x] 实现 `src/main.py` 训练 / 验证 / 测试主流程
- [x] 生成问题二 baseline 说明文档
- [x] 完成一次 `20 epoch` baseline 训练与结果检查
- [x] 补齐项目级文档 `README.md`、`AI_CONTEXT.md`、`TODO.md`
- [x] 初步阅读 `..\相关论文` 中的参考文献并整理可迁移优化思路
- [x] 核对本机 GPU 与当前 `..\LCC` 环境状态
- [x] 将训练脚本改为 GPU 优先版本，支持 `CUDA/CPU` 自动切换、AMP 和 GPU 输出目录
- [x] 基于 `..\LCC310` 建立 `..\LCC_GPU` CUDA 环境
- [x] 完成 `..\LCC_GPU` 的 CUDA 导入验证与最小 GPU smoke test
- [x] 完成一次 `GPU + pretrained` 的正式 baseline 训练
- [x] 增加 `label smoothing` 与学习率调度器参数
- [x] 完成一次 `GPU + pretrained + label smoothing + cosine scheduler` 对比实验
- [x] 增加类别加权 `CrossEntropyLoss` 参数
- [x] 完成一次 `balanced class-weighted CrossEntropy` 对比实验
- [x] 增加 `Focal Loss` 与 `--focal-gamma` 参数
- [x] 完成一次 `GPU + pretrained + focal loss(gamma=2) + label smoothing + cosine scheduler` 对比实验

## 未完成任务

- [ ] 将问题一整理为正式论文式答案
- [ ] 优化问题二 baseline 的泛化性能
- [ ] 重点降低腺癌和鳞状细胞癌的误诊率
- [ ] 补充更完整的实验对比与可视化结果

## 新增待办事项

- [ ] 视资源情况补试 EfficientNet-B1
- [ ] 在当前 `Focal Loss` 基线上继续尝试采样策略
- [ ] 如有必要再尝试手动类别权重，而不是直接使用 `balanced` 权重
- [ ] 尝试 `Focal Loss + 手动权重`，平衡腺癌 / 鳞癌 / 大细胞癌召回率
- [ ] 试验轻量注意力模块 `SE` / `CBAM`
- [ ] 输出更细化的误差分析结果，并补充 `Grad-CAM` 或高混淆样本检查
- [ ] 在实验记录中明确比较 `macro F1`、各类召回率与混淆矩阵
- [ ] 根据最新实验结果更新 `doc/problem2_baseline.md` 与 `doc/literature_review.md`

## 当前优先级

- 高优先级：提升问题二 baseline 的测试集性能与稳定性
- 高优先级：修复当前 `Focal Loss` 下鳞癌召回率回落的问题
- 高优先级：优先验证采样策略或 `Focal Loss + 手动权重` 是否能在保住腺癌 / 大细胞癌收益的同时拉回鳞癌召回率
- 中优先级：整理问题一正式答案和问题二实验总结
- 中优先级：补充图表、混淆矩阵解读和论文可用文字
- 低优先级：整理目录和长期维护文档
