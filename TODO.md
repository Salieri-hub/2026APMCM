# TODO

## 已完成

- [x] 将默认主干从 `EfficientNet-B3` 切换为 `EfficientNet-B4`
- [x] 将默认输入尺寸从 `288` 更新为 `320`
- [x] 为 `B4` 增加本地预训练权重加载支持
- [x] 保留历史 `B0/B1/B2/B3` checkpoint 的兼容性
- [x] 继续沿用共享输出结构：
- [x] `outputs/weights/<experiment_name>/`
- [x] `outputs/results/<experiment_name>/`
- [x] 新增 `B4` 的 `50` 组正式实验批量脚本
- [x] 完成 `B0` 到 `B4` 的单模型、专家模型和级联实验归档
- [x] 完成 `B4` 的 `50` 组正式实验批量运行
- [x] 已在 `outputs/results` 中确认当前最佳单模型：`v3.2_pretrained_focal_ls_cosine_cutmix_b4`（测试集准确率 `94.29%`，Macro F1 `0.9415`）
- [x] 已在 `outputs/results` 中确认当前最佳级联结果：`cascade_pair_ad_sq_v3.2_pretrained_focal_ls_cosine_cutmix_b4`（测试集准确率 `94.60%`，Macro F1 `0.9439`）
- [x] `src/main.py` 已完成拆分，当前仅保留入口，核心逻辑迁移到 `src/lcc/`
- [x] 将全部项目 Markdown 文档改为中文并同步到当前 `B4` 基线与实验进度

## 进行中

- [ ] 将当前最佳 `B4` 结果继续同步到最终 Word 报告与论文正文
- [ ] 整理 `B0` 到 `B4` 的横向对比摘要与图表

## 可选后续

- [ ] 继续调节 `expert_margin_threshold` 与 `top-k` 触发策略
- [ ] 围绕 `v3.2_pretrained_focal_ls_cosine_cutmix_b4` 继续做局部增量实验
- [ ] 为专家模型与级联策略单独整理最终结果摘要
