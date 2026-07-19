# TODO

## 已完成任务

- [x] 阅读赛题 PDF 并拆解问题一与问题二要求
- [x] 统计并核对数据集目录结构与样本规模
- [x] 实现 `src/main.py` 训练 / 验证 / 测试主流程
- [x] 将训练脚本改为 GPU 优先版本，支持 `CUDA/CPU` 自动切换与 AMP
- [x] 建立并验证 `..\LCC_GPU` CUDA 环境
- [x] 在 `main.py` 中加入 `label smoothing`
- [x] 在 `main.py` 中加入学习率调度器
- [x] 在 `main.py` 中加入 `Focal Loss`
- [x] 在 `main.py` 中加入类别加权 `CrossEntropy`
- [x] 在 `main.py` 中加入 `MixUp / CutMix`
- [x] 在 `main.py` 中加入 `SE / CBAM`
- [x] 在 `main.py` 中加入 `single / expert / cascade` 三种运行模式
- [x] 完成 B0 `12` 组单模型正式实验
- [x] 完成 B0 `10` 组三肿瘤专家级联实验
- [x] 完成 B0 `30` 组两两肿瘤专家级联实验
- [x] 完成 `40` 个专家模型训练
- [x] 核查专家模型训练日志，确认主要问题为过拟合而非欠拟合
- [x] 生成三肿瘤级联批处理脚本 `scripts/run_all_cascade_tumor3.ps1`
- [x] 生成两两肿瘤级联批处理脚本 `scripts/run_all_cascade_tumor_pairs.ps1`
- [x] 将默认 backbone 从 `EfficientNet-B0` 切换到 `EfficientNet-B1`
- [x] 增加 B1 默认输入尺寸 `240`
- [x] 增加 B1 正式实验命名规则 `_b1`
- [x] 生成 B1 的 `50` 组正式实验批处理脚本：
- [x] `scripts/run_all_efficientnet_b1_50.ps1`
- [x] `scripts/run_all_efficientnet_b1_50.cmd`
- [x] 更新 `README.md`、`AI_CONTEXT.md`、`TODO.md`、`doc/problem2_baseline.md`

## 当前进行中的任务

- [ ] 运行 `EfficientNet-B1` 的 `50` 组正式实验
- [ ] 将问题一整理为正式论文式答案
- [ ] 将历史 `52` 组 B0 结果与新一轮 B1 结果整合为论文正文可直接使用的实验分析段落
- [ ] 补充最佳单模型与最佳级联模型的误差分析

## 后续可选任务

- [ ] 对 B1 最佳主模型微调 `expert_margin_threshold`
- [ ] 对 B1 最佳主模型比较 `top-k=2` 与其他触发规则
- [ ] 对最佳级联结果补充更细的样本级修正统计
- [ ] 补充 `Grad-CAM` 或高混淆样本可视化
- [ ] 如有必要，再比较 `EfficientNet-B1` 与其他 backbone
- [ ] 如有必要，再尝试“主模型判定非 normal 即强制进入专家模型”的触发变体

## 当前优先级

- 高优先级：先把 B1 的 `50` 组正式实验跑完
- 高优先级：比较 B1 与历史 B0 在单模型和级联两条主线上的差异
- 中优先级：围绕 B1 最佳结果补充误差分析和混淆矩阵解读
- 中优先级：整理问题一正式答案
- 低优先级：继续扩充更重 backbone

## 当前结论备忘

- 当前已验证最佳单模型：`v3.4_pretrained_focal_ls_cosine_cbam`
- 当前已验证最佳单模型 backbone：`EfficientNet-B0`
- 当前已验证最佳单模型测试集准确率：`86.35%`
- 当前已验证最佳单模型测试集 `macro F1`：`0.8646`
- 当前已验证最佳全局结果：`cascade_v3.4_pretrained_focal_ls_cosine_cbam`
- 当前已验证最佳全局测试集准确率：`87.62%`
- 当前已验证最佳全局测试集 `macro F1`：`0.8773`
- 新一轮待执行：B1 `10` 组单模型 + `10` 组三肿瘤级联 + `30` 组两两级联
