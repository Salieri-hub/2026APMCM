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
- [x] 完成 `12` 组单模型正式消融实验
- [x] 完成 `10` 组三肿瘤专家级联实验
- [x] 完成 `30` 组两两肿瘤专家级联实验
- [x] 完成 `40` 个专家模型训练
- [x] 核查专家模型训练日志，确认主要问题为过拟合而非欠拟合
- [x] 生成三肿瘤级联批处理脚本 `scripts/run_all_cascade_tumor3.ps1`
- [x] 生成两两肿瘤级联批处理脚本 `scripts/run_all_cascade_tumor_pairs.ps1`
- [x] 输出 `doc/ablation_results.md`
- [x] 输出 `doc/实验版本对比说明_20260718.docx`
- [x] 输出 `doc/outputs_实验版本说明与消融对比_20260719.docx`
- [x] 输出 `doc/all_52_experiments_comparison_20260719.docx`
- [x] 输出 `doc/ablation_results_中文翻译_20260719.docx`
- [x] 更新 `README.md`、`AI_CONTEXT.md`、`TODO.md`、`doc/problem2_baseline.md`

## 当前进行中的任务

- [ ] 将问题一整理为正式论文式答案
- [ ] 将 `52` 组实验结果进一步压缩为论文正文可直接使用的实验分析段落
- [ ] 补充最佳单模型与最佳级联模型的误差分析
- [ ] 评估当前专家触发阈值是否还需要微调

## 后续可选任务

- [ ] 在 `v3.4_pretrained_focal_ls_cosine_cbam` 主模型上微调 `expert_margin_threshold`
- [ ] 在 `v3.4_pretrained_focal_ls_cosine_cbam` 主模型上比较 `top-k=2` 与其他触发规则
- [ ] 对最佳级联结果补充更细的样本级修正统计
- [ ] 补充 `Grad-CAM` 或高混淆样本可视化
- [ ] 如有必要，再尝试 `EfficientNet-B1` 或其他 backbone
- [ ] 如有必要，再尝试“主模型判定非 normal 即强制进入专家模型”的触发变体

## 当前优先级

- 高优先级：整理 `52` 组实验结论，形成论文可用版本
- 高优先级：围绕最佳级联结果补充误差分析和混淆矩阵解读
- 中优先级：微调专家触发逻辑，判断能否稳定超过当前最佳结果
- 中优先级：整理问题一正式答案
- 低优先级：继续扩充 backbone 或更重结构

## 当前结论备忘

- 最佳单模型：`v3.4_pretrained_focal_ls_cosine_cbam`
- 最佳单模型测试集准确率：`86.35%`
- 最佳单模型测试集 `macro F1`：`0.8646`
- 最佳全局结果：`cascade_v3.4_pretrained_focal_ls_cosine_cbam`
- 最佳全局测试集准确率：`87.62%`
- 最佳全局测试集 `macro F1`：`0.8773`
- 三肿瘤级联优于对应单模型：`8 / 10`
- 两两肿瘤级联优于对应单模型：`21 / 30`
