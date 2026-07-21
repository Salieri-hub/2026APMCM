# 进度报告

最近更新：`2026-07-21`

## 1. 概要

仓库默认基线已经从 `EfficientNet-B3` 继续迁移到 `EfficientNet-B4`。

本次迁移保留了既有的单模型、专家模型和级联框架，同时更新了默认主干、默认输入尺寸和对应的批量脚本。

## 2. 已完成修改

### 2.1 主干迁移

- 默认 `--model-name` 更新为 `efficientnet_b4`
- 默认 `--image-size` 更新为 `320`
- 新增 `B4` 本地预训练权重加载配置
- 保留历史 `B0/B1/B2/B3` checkpoint 兼容性

### 2.2 输出结构

所有新实验继续写入 `outputs/` 下的共享目录：

- `outputs/weights/<experiment_name>/best_model.pt`
- `outputs/results/<experiment_name>/metrics_summary.json`
- `outputs/results/<experiment_name>/...csv 文件...`

该结构延续此前的共享目录设计，不再回退到旧的按实验嵌套目录形式。

### 2.3 批量脚本

本次新增：

- `scripts/run_all_efficientnet_b4_50.ps1`
- `scripts/run_all_efficientnet_b4_50.cmd`

脚本覆盖范围如下：

- `10` 组单模型实验
- `10` 组三分类肿瘤专家级联实验
- `30` 组两两专家级联实验
- 总计：`50`

## 3. 当前参考结果

当前已经完整验证的历史最佳单模型仍为 `B0`：

- 实验名：`v3.4_pretrained_focal_ls_cosine_cbam`
- 测试集准确率：`86.35%`
- Macro F1：`0.8646`

当前已经完整验证的历史最佳级联仍为 `B0`：

- 实验名：`cascade_v3.4_pretrained_focal_ls_cosine_cbam`
- 测试集准确率：`87.62%`
- Macro F1：`0.8773`

## 4. 当前状态

`B4` 代码路径、文档和批量脚本已经就绪。

尚未完成的部分包括：

- 正式 `50` 组 `B4` 实验
- `B4` 与历史 `B0` 的结果对比
- 基于 `B4` 结果的后续 Word 文档整理

## 5. 运行命令

如果当前目录位于 `2026APMCM`：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_all_efficientnet_b4_50.ps1" -PythonExe "..\LCC_GPU\python.exe"
```

或者：

```powershell
.\scripts\run_all_efficientnet_b4_50.cmd
```

## 6. 下一步

运行 `B4` 的 `50` 组正式实验，随后整理跨主干版本的对比文档。
