# TODO

## 已完成

- [x] 将默认主干从 `EfficientNet-B2` 切换为 `EfficientNet-B3`
- [x] 将默认输入尺寸从 `256` 更新为 `288`
- [x] 为 `B3` 增加本地预训练权重加载支持
- [x] 保留历史 `B0/B1/B2` checkpoint 的兼容性
- [x] 继续沿用共享输出结构：
- [x] `outputs/weights/<experiment_name>/`
- [x] `outputs/results/<experiment_name>/`
- [x] 新增 `B3` 的 `50` 组正式实验批量脚本
- [x] 将全部项目 Markdown 文档改为中文并同步到 `B3` 基线

## 进行中

- [ ] 运行 `50` 组正式 `B3` 实验
- [ ] 汇总 `B3` 与历史 `B0` 的对比结果
- [ ] 在拿到 `B3` 结果后更新 Word 报告

## 可选后续

- [ ] 在 `B3` 跑完后继续调节级联触发阈值
- [ ] 增补 `B0`、`B2`、`B3` 的横向对比文档
- [ ] 为 `40` 组专家模型训练单独整理结果摘要
