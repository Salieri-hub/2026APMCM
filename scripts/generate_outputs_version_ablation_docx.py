from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
DOC_DIR = ROOT / "doc"
DOC_PATH = DOC_DIR / "outputs_实验版本说明与消融对比_20260719.docx"

CLASS_ORDER = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma",
]

CLASS_DISPLAY = {
    "adenocarcinoma": "腺癌",
    "large.cell.carcinoma": "大细胞癌",
    "normal": "正常",
    "squamous.cell.carcinoma": "鳞癌",
}

VERSION_META = [
    {
        "name": "v1.0_scratch_ce_cpu",
        "based_on": "无",
        "compare_to": ["v1.1_scratch_ce_cuda"],
        "primary_compare": None,
        "change_name": "初始 scratch + CE CPU baseline",
        "change_explanation": "随机初始化的 EfficientNet-B0，损失函数使用 CrossEntropyLoss，在 CPU 上训练，用于建立最原始的可运行基线。",
    },
    {
        "name": "v1.1_scratch_ce_cuda",
        "based_on": "v1.0_scratch_ce_cpu",
        "compare_to": ["v1.0_scratch_ce_cpu", "v2.0_pretrained_ce"],
        "primary_compare": "v1.0_scratch_ce_cpu",
        "change_name": "仅切换到 CUDA",
        "change_explanation": "只把训练硬件从 CPU 切到 CUDA，模型结构和损失函数保持不变。这个版本主要用于验证 GPU 加速本身会带来多大性能变化。",
    },
    {
        "name": "v2.0_pretrained_ce",
        "based_on": "v1.1_scratch_ce_cuda",
        "compare_to": ["v1.1_scratch_ce_cuda", "v2.1_pretrained_ce_cosine", "v2.2_pretrained_ce_ls"],
        "primary_compare": "v1.1_scratch_ce_cuda",
        "change_name": "启用预训练权重",
        "change_explanation": "在 scratch CE CUDA 版本上启用 pretrained。预训练本质上是迁移学习，先使用大规模数据学到的通用视觉特征，再在肺癌 CT 数据上微调。",
    },
    {
        "name": "v2.1_pretrained_ce_cosine",
        "based_on": "v2.0_pretrained_ce",
        "compare_to": ["v2.0_pretrained_ce", "v2.2_pretrained_ce_ls"],
        "primary_compare": "v2.0_pretrained_ce",
        "change_name": "加入 cosine scheduler",
        "change_explanation": "在预训练 CE 基线之上加入 cosine scheduler。cosine scheduler 是一种学习率调度器，会让学习率按余弦曲线逐步下降，使训练后期更新更平滑。",
    },
    {
        "name": "v2.2_pretrained_ce_ls",
        "based_on": "v2.0_pretrained_ce",
        "compare_to": ["v2.0_pretrained_ce", "v2.1_pretrained_ce_cosine", "v2.3_pretrained_ce_ls_cosine"],
        "primary_compare": "v2.0_pretrained_ce",
        "change_name": "加入 label smoothing",
        "change_explanation": "在预训练 CE 基线之上加入 label smoothing。label smoothing 是一种标签平滑策略，会把原本绝对的 one-hot 标签稍微软化，以降低模型过度自信、提升泛化能力。",
    },
    {
        "name": "v2.3_pretrained_ce_ls_cosine",
        "based_on": "v2.2_pretrained_ce_ls",
        "compare_to": ["v2.2_pretrained_ce_ls", "v2.4_pretrained_ce_ls_cosine_weightedce", "v3.0_pretrained_focal_ls_cosine"],
        "primary_compare": "v2.2_pretrained_ce_ls",
        "change_name": "在 label smoothing 基础上再加 cosine scheduler",
        "change_explanation": "这个版本把 label smoothing 和 cosine scheduler 叠加起来，目的是测试两种训练策略组合后是否能进一步提升泛化能力。",
    },
    {
        "name": "v2.4_pretrained_ce_ls_cosine_weightedce",
        "based_on": "v2.3_pretrained_ce_ls_cosine",
        "compare_to": ["v2.3_pretrained_ce_ls_cosine"],
        "primary_compare": "v2.3_pretrained_ce_ls_cosine",
        "change_name": "加入 balanced class-weighted CE",
        "change_explanation": "在 CE + label smoothing + cosine 的基础上，再把 CrossEntropyLoss 改成类别加权版本。balanced class-weighted CE 会根据训练集类别频次给少数类更高损失权重，用来缓解类别不平衡。",
    },
    {
        "name": "v3.0_pretrained_focal_ls_cosine",
        "based_on": "v2.3_pretrained_ce_ls_cosine",
        "compare_to": [
            "v2.3_pretrained_ce_ls_cosine",
            "v3.1_pretrained_focal_ls_cosine_mixup",
            "v3.2_pretrained_focal_ls_cosine_cutmix",
            "v3.3_pretrained_focal_ls_cosine_se",
            "v3.4_pretrained_focal_ls_cosine_cbam",
        ],
        "primary_compare": "v2.3_pretrained_ce_ls_cosine",
        "change_name": "把 CE 换成 focal loss",
        "change_explanation": "这个版本保留 pretrained、label smoothing 和 cosine scheduler，但把 CE 换成 focal loss。focal loss 会降低易分类样本的损失权重，把训练重点放到难样本和误分类样本上。",
    },
    {
        "name": "v3.1_pretrained_focal_ls_cosine_mixup",
        "based_on": "v3.0_pretrained_focal_ls_cosine",
        "compare_to": ["v3.0_pretrained_focal_ls_cosine"],
        "primary_compare": "v3.0_pretrained_focal_ls_cosine",
        "change_name": "加入 MixUp",
        "change_explanation": "在 focal loss 基线之上加入 MixUp。MixUp 是一种样本混合增强，会对两张图像和对应标签做线性插值，以增强模型的平滑性和鲁棒性。",
    },
    {
        "name": "v3.2_pretrained_focal_ls_cosine_cutmix",
        "based_on": "v3.0_pretrained_focal_ls_cosine",
        "compare_to": ["v3.0_pretrained_focal_ls_cosine"],
        "primary_compare": "v3.0_pretrained_focal_ls_cosine",
        "change_name": "加入 CutMix",
        "change_explanation": "在 focal loss 基线之上加入 CutMix。CutMix 会把一张图像的局部区域剪切并粘贴到另一张图像上，同时按面积比例混合标签，以增强模型对局部扰动的鲁棒性。",
    },
    {
        "name": "v3.3_pretrained_focal_ls_cosine_se",
        "based_on": "v3.0_pretrained_focal_ls_cosine",
        "compare_to": ["v3.0_pretrained_focal_ls_cosine", "v3.4_pretrained_focal_ls_cosine_cbam"],
        "primary_compare": "v3.0_pretrained_focal_ls_cosine",
        "change_name": "加入额外 SE 注意力模块",
        "change_explanation": "在 focal loss 基线之上加入额外的 SE 模块。SE 是一种通道注意力机制，会学习每个通道的重要性并对特征进行重标定。",
    },
    {
        "name": "v3.4_pretrained_focal_ls_cosine_cbam",
        "based_on": "v3.0_pretrained_focal_ls_cosine",
        "compare_to": ["v3.0_pretrained_focal_ls_cosine", "v3.3_pretrained_focal_ls_cosine_se"],
        "primary_compare": "v3.0_pretrained_focal_ls_cosine",
        "change_name": "加入 CBAM 注意力模块",
        "change_explanation": "在 focal loss 基线之上加入 CBAM。CBAM 是一种同时包含通道注意力和空间注意力的轻量注意力模块，既能强调关键通道，也能强调关键空间区域。",
    },
]

COMPARISON_GROUPS = [
    {
        "title": "1. 硬件切换对照：v1.0_scratch_ce_cpu vs v1.1_scratch_ce_cuda",
        "pairs": [("v1.0_scratch_ce_cpu", "v1.1_scratch_ce_cuda")],
        "conclusion_template": "只把训练从 CPU 切到 CUDA，测试准确率变化 {acc_delta}，Macro F1 变化 {f1_delta}。这说明更快的硬件主要提升训练效率，本身并不能显著改变模型判别能力。",
    },
    {
        "title": "2. 迁移学习对照：v1.1_scratch_ce_cuda vs v2.0_pretrained_ce",
        "pairs": [("v1.1_scratch_ce_cuda", "v2.0_pretrained_ce")],
        "conclusion_template": "启用预训练后，测试准确率提升 {acc_delta}，Macro F1 提升 {f1_delta}。这是当前所有改动中单次增益最大的一步，说明迁移学习是本任务最关键的提升来源。",
    },
    {
        "title": "3. CE 家族中的轻量训练策略对照：v2.0 / v2.1 / v2.2 / v2.3 / v2.4",
        "pairs": [
            ("v2.0_pretrained_ce", "v2.1_pretrained_ce_cosine"),
            ("v2.0_pretrained_ce", "v2.2_pretrained_ce_ls"),
            ("v2.2_pretrained_ce_ls", "v2.3_pretrained_ce_ls_cosine"),
            ("v2.3_pretrained_ce_ls_cosine", "v2.4_pretrained_ce_ls_cosine_weightedce"),
        ],
        "conclusion_template": (
            "在 CE 家族中，单独加入 cosine 相比 v2.0 的准确率变化 {p1_acc}、Macro F1 变化 {p1_f1}；"
            "单独加入 label smoothing 相比 v2.0 的准确率变化 {p2_acc}、Macro F1 变化 {p2_f1}；"
            "在 label smoothing 基础上继续叠加 cosine 后，相比 v2.2 的准确率变化 {p3_acc}、Macro F1 变化 {p3_f1}；"
            "继续加入 balanced weighted CE 后，相比 v2.3 的准确率变化 {p4_acc}、Macro F1 变化 {p4_f1}。"
            "结论是：label smoothing 是 CE 家族里最有效的低成本正则，cosine 单独有小幅帮助，但与 label smoothing 叠加后并未继续提升，而 balanced weighted CE 在这组数据上出现了过度校正。"
        ),
    },
    {
        "title": "4. 损失函数替换对照：v2.3_pretrained_ce_ls_cosine vs v3.0_pretrained_focal_ls_cosine",
        "pairs": [("v2.3_pretrained_ce_ls_cosine", "v3.0_pretrained_focal_ls_cosine")],
        "conclusion_template": "在 backbone、预训练和调度策略都不变的前提下，把 CE 换成 focal loss 后，测试准确率变化 {acc_delta}，Macro F1 变化 {f1_delta}。这说明 focal loss 对难样本学习有正向帮助。",
    },
    {
        "title": "5. 混合增强对照：v3.0 vs v3.1 vs v3.2",
        "pairs": [
            ("v3.0_pretrained_focal_ls_cosine", "v3.1_pretrained_focal_ls_cosine_mixup"),
            ("v3.0_pretrained_focal_ls_cosine", "v3.2_pretrained_focal_ls_cosine_cutmix"),
        ],
        "conclusion_template": "在 focal loss 基线上加入 MixUp 后，测试准确率变化 {p1_acc}、Macro F1 变化 {p1_f1}；加入 CutMix 后，测试准确率变化 {p2_acc}、Macro F1 变化 {p2_f1}。两种混合增强都降低了性能，说明在当前小样本医学图像任务中，过强的样本混合可能破坏了细微病灶线索。",
    },
    {
        "title": "6. 注意力结构对照：v3.0 vs v3.3 vs v3.4",
        "pairs": [
            ("v3.0_pretrained_focal_ls_cosine", "v3.3_pretrained_focal_ls_cosine_se"),
            ("v3.0_pretrained_focal_ls_cosine", "v3.4_pretrained_focal_ls_cosine_cbam"),
            ("v3.3_pretrained_focal_ls_cosine_se", "v3.4_pretrained_focal_ls_cosine_cbam"),
        ],
        "conclusion_template": "相对 focal baseline，额外加入 SE 后测试准确率变化 {p1_acc}、Macro F1 变化 {p1_f1}；加入 CBAM 后测试准确率变化 {p2_acc}、Macro F1 变化 {p2_f1}；CBAM 再相对 SE 的准确率变化 {p3_acc}、Macro F1 变化 {p3_f1}。说明仅追加通道重标定并不充分，而同时建模通道与空间注意力的 CBAM 更适合当前任务。",
    },
]

REPRESENTATIVE_VERSIONS = [
    "v1.0_scratch_ce_cpu",
    "v1.1_scratch_ce_cuda",
    "v2.0_pretrained_ce",
    "v2.2_pretrained_ce_ls",
    "v3.0_pretrained_focal_ls_cosine",
    "v3.4_pretrained_focal_ls_cosine_cbam",
]


def load_metrics(version_name: str) -> dict:
    path = OUTPUTS_DIR / version_name / "metrics_summary.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_confusion(version_name: str) -> list[list[str]]:
    path = OUTPUTS_DIR / version_name / "test_confusion_matrix.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def float4(value: float) -> str:
    return f"{value:.4f}"


def delta_pct(target: float, base: float) -> str:
    diff = (target - base) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.2f} 个百分点"


def delta_float(target: float, base: float) -> str:
    diff = target - base
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.4f}"


def get_record(meta: dict) -> dict:
    metrics = load_metrics(meta["name"])
    report = metrics["test"]["report"]
    recalls = {cls: report[cls]["recall"] for cls in CLASS_ORDER}
    return {
        **meta,
        "metrics": metrics,
        "test_acc": metrics["test"]["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "best_epoch": metrics["best_epoch"],
        "recalls": recalls,
    }


def paragraph(text: str, *, bold: bool = False, size: int = 22, center: bool = False) -> str:
    ppr = "<w:pPr><w:spacing w:after=\"120\"/>"
    if center:
        ppr += "<w:jc w:val=\"center\"/>"
    ppr += "</w:pPr>"
    rpr = [
        "<w:rFonts w:ascii=\"Calibri\" w:hAnsi=\"Calibri\" w:eastAsia=\"宋体\"/>",
        f"<w:sz w:val=\"{size}\"/>",
        f"<w:szCs w:val=\"{size}\"/>",
    ]
    if bold:
        rpr.append("<w:b/>")
    return (
        "<w:p>"
        f"{ppr}"
        "<w:r>"
        f"<w:rPr>{''.join(rpr)}</w:rPr>"
        f"<w:t xml:space=\"preserve\">{escape(text)}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def table(rows: list[list[str]], col_widths: list[int]) -> str:
    grid_cols = "".join(f"<w:gridCol w:w=\"{width}\"/>" for width in col_widths)
    tbl_rows = []
    for row_idx, row in enumerate(rows):
        cells = []
        for cell in row:
            content = paragraph(cell, bold=row_idx == 0, size=18)
            cells.append(
                "<w:tc>"
                "<w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/></w:tcPr>"
                f"{content}"
                "</w:tc>"
            )
        tbl_rows.append(f"<w:tr>{''.join(cells)}</w:tr>")

    borders = (
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "</w:tblBorders>"
    )

    return (
        "<w:tbl>"
        f"<w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>{borders}</w:tblPr>"
        f"<w:tblGrid>{grid_cols}</w:tblGrid>"
        f"{''.join(tbl_rows)}"
        "</w:tbl>"
    )


def build_version_paragraph(record: dict, records_by_name: dict[str, dict]) -> str:
    compare_list = "、".join(record["compare_to"])
    text = (
        f"{record['name']} 由 {record['based_on']} 修改而来，可与 {compare_list} 构成对照消融实验。"
        f"本版本的关键改动是：{record['change_name']}。{record['change_explanation']}"
        f"该版本测试准确率为 {pct(record['test_acc'])}，Macro F1 为 {float4(record['macro_f1'])}。"
    )
    if record["primary_compare"] and record["primary_compare"] in records_by_name:
        base = records_by_name[record["primary_compare"]]
        text += (
            f" 相对主要对照版本 {record['primary_compare']}，"
            f"测试准确率变化 {delta_pct(record['test_acc'], base['test_acc'])}，"
            f"Macro F1 变化 {delta_float(record['macro_f1'], base['macro_f1'])}。"
        )
    return paragraph(text, size=21)


def build_summary_table(records: list[dict], records_by_name: dict[str, dict]) -> str:
    rows = [[
        "版本目录",
        "由谁修改而来",
        "主要对照版本",
        "关键改动",
        "Test Acc",
        "Macro F1",
        "相对主要对照的变化",
    ]]
    for record in records:
        compare = record["primary_compare"] or "-"
        if record["primary_compare"] and record["primary_compare"] in records_by_name:
            base = records_by_name[record["primary_compare"]]
            delta_text = (
                f"Acc {delta_pct(record['test_acc'], base['test_acc'])}; "
                f"F1 {delta_float(record['macro_f1'], base['macro_f1'])}"
            )
        else:
            delta_text = "-"
        rows.append([
            record["name"],
            record["based_on"],
            compare,
            record["change_name"],
            pct(record["test_acc"]),
            float4(record["macro_f1"]),
            delta_text,
        ])
    return table(rows, [2600, 2300, 2300, 2500, 1200, 1200, 2600])


def build_conclusion_paragraphs(records_by_name: dict[str, dict]) -> list[str]:
    parts: list[str] = []
    for group in COMPARISON_GROUPS:
        parts.append(paragraph(group["title"], bold=True, size=22))
        values: dict[str, str] = {}
        for idx, (base_name, target_name) in enumerate(group["pairs"], start=1):
            base = records_by_name[base_name]
            target = records_by_name[target_name]
            acc_key = "acc_delta" if len(group["pairs"]) == 1 else f"p{idx}_acc"
            f1_key = "f1_delta" if len(group["pairs"]) == 1 else f"p{idx}_f1"
            values[acc_key] = delta_pct(target["test_acc"], base["test_acc"])
            values[f1_key] = delta_float(target["macro_f1"], base["macro_f1"])
        parts.append(paragraph(group["conclusion_template"].format(**values), size=21))
    return parts


def build_ranking_table(records: list[dict]) -> str:
    ranked = sorted(records, key=lambda x: (x["test_acc"], x["macro_f1"]), reverse=True)
    rows = [["排名", "版本", "所属家族", "Test Acc", "Macro F1", "简要判断"]]
    for idx, record in enumerate(ranked, start=1):
        if record["name"].startswith("v1."):
            family = "v1.x scratch + CE"
        elif record["name"].startswith("v2."):
            family = "v2.x pretrained + CE"
        else:
            family = "v3.x pretrained + focal + ls + cosine"

        if record["name"] == "v3.4_pretrained_focal_ls_cosine_cbam":
            note = "当前综合最优"
        elif record["name"] == "v2.2_pretrained_ce_ls":
            note = "CE 家族最佳"
        elif record["name"] == "v2.0_pretrained_ce":
            note = "预训练带来第一次大幅跃升"
        else:
            note = "-"

        rows.append([
            str(idx),
            record["name"],
            family,
            pct(record["test_acc"]),
            float4(record["macro_f1"]),
            note,
        ])
    return table(rows, [900, 3200, 2800, 1200, 1200, 2000])


def build_recall_table(records_by_name: dict[str, dict]) -> str:
    rows = [["代表版本", "腺癌召回", "大细胞癌召回", "正常召回", "鳞癌召回"]]
    for name in REPRESENTATIVE_VERSIONS:
        record = records_by_name[name]
        rows.append([
            name,
            pct(record["recalls"]["adenocarcinoma"]),
            pct(record["recalls"]["large.cell.carcinoma"]),
            pct(record["recalls"]["normal"]),
            pct(record["recalls"]["squamous.cell.carcinoma"]),
        ])
    return table(rows, [3200, 1600, 1600, 1400, 1600])


def build_confusion_table(version_name: str) -> str:
    rows = load_confusion(version_name)
    rows[0] = ["真实/预测", "腺癌", "大细胞癌", "正常", "鳞癌"]
    for row in rows[1:]:
        row[0] = CLASS_DISPLAY.get(row[0], row[0])
    return table(rows, [1800, 1600, 1600, 1400, 1600])


def build_chart_perspective_paragraphs(records_by_name: dict[str, dict]) -> list[str]:
    v20 = records_by_name["v2.0_pretrained_ce"]
    v22 = records_by_name["v2.2_pretrained_ce_ls"]
    v34 = records_by_name["v3.4_pretrained_focal_ls_cosine_cbam"]

    parts = [
        paragraph("四、补充的图表分析视角", bold=True, size=24),
        paragraph("这一部分不替代前面的逐版本说明和消融分析，而是补充 experiment_summary_charts_2026-07-19.docx 中那种“整体趋势 + 代表性图表 + 阶段结论”的分析视角。", size=21),
        paragraph("4.1 命名规则与正式实验范围", bold=True, size=22),
        paragraph("正式实验目录统一遵循 vX.Y_change 命名规则，其中 v1.x 表示 scratch + CE 家族，v2.x 表示 pretrained + CE 家族，v3.x 表示 pretrained + focal + label smoothing + cosine 家族。当前共完成 12 组正式实验，smoke test 不纳入正式对比。", size=21),
        paragraph("4.2 总体趋势总览", bold=True, size=22),
        paragraph("从全局趋势看，最明显的三个有效提升方向分别是：预训练、label smoothing、CBAM。硬件切换本身只带来极小变化，说明速度提升不等于识别能力提升；预训练使模型第一次从“不可用”跃升到“有竞争力”；CBAM 则把当前最强路线进一步推到全体最优。", size=21),
        paragraph("从家族视角看，v1.x 主要用于建立最初基线；v2.x 负责验证迁移学习和 CE 家族的训练策略；v3.x 则代表当前最强主线，即 pretrained + focal + label smoothing + cosine。在 2026 年 7 月 19 日这个时间点，项目已经从“跑通 baseline”阶段进入到“围绕最优路线做定向优化”的阶段。", size=21),
        build_ranking_table(list(records_by_name.values())),
        paragraph("4.3 代表性版本的分类别召回视角", bold=True, size=22),
        paragraph("如果把代表性版本的四类召回率放在一起看，模型性能提升并不是所有类别一起均匀上升，而是体现为不同改动对不同类别边界的影响。下表可视为“召回热力图”的文字化版本。", size=21),
        build_recall_table(records_by_name),
        paragraph(
            f"从召回结构看，v1.0_scratch_ce_cpu 对腺癌几乎失效，腺癌召回只有 6.67%，同时对大细胞癌存在明显过预测；"
            f"v2.0_pretrained_ce 启用预训练后，腺癌召回直接提升到 {pct(v20['recalls']['adenocarcinoma'])}，鳞癌召回提升到 {pct(v20['recalls']['squamous.cell.carcinoma'])}；"
            f"v2.2_pretrained_ce_ls 进一步把鳞癌召回推到 {pct(v22['recalls']['squamous.cell.carcinoma'])}，说明 label smoothing 对缓解过度自信和类间边界僵硬非常有效；"
            f"v3.4_pretrained_focal_ls_cosine_cbam 则把腺癌召回继续提升到 {pct(v34['recalls']['adenocarcinoma'])}，同时保持大细胞癌和正常类的高召回，但鳞癌召回回落到 {pct(v34['recalls']['squamous.cell.carcinoma'])}。"
            f"这说明当前最优模型虽然整体性能最好，但鳞癌仍然是后续优化重点。",
            size=21,
        ),
        paragraph("4.4 最佳模型混淆矩阵视角", bold=True, size=22),
        paragraph("图表文档里还单独强调了最佳模型的测试混淆矩阵。把这部分转成文字后，可以更清楚地看到当前剩余误差不是平均分布的，而是集中在少数固定混淆通道上。", size=21),
        build_confusion_table("v3.4_pretrained_focal_ls_cosine_cbam"),
        paragraph("对最佳模型 v3.4_pretrained_focal_ls_cosine_cbam 而言，正常类几乎已经稳定，54 张测试样本中有 53 张预测正确；大细胞癌也较强，51 张中有 46 张预测正确；腺癌 120 张中有 109 张预测正确，已经明显优于前期版本。当前最大的残余问题是鳞癌：90 张鳞癌中仍有 26 张被误分，其中 13 张被判为腺癌，13 张被判为大细胞癌。这说明最佳模型的主要改进空间已经收缩到鳞癌相关误差分析，而不是所有类别都需要平均用力。", size=21),
        paragraph("4.5 阶段性口头总结视角", bold=True, size=22),
        paragraph("如果用更口头化、汇报式的方式概括当前实验状态，可以得到四个判断：第一，12 组正式实验已经完成，项目已进入优化阶段；第二，当前最强路线是 v3.x，即 pretrained + focal + label smoothing + cosine；第三，当前综合最优实验是 v3.4_pretrained_focal_ls_cosine_cbam，测试准确率 86.35%，Macro F1 0.8646；第四，下一步重点仍然是鳞癌召回率提升和高频误判样本分析。", size=21),
    ]
    return parts


def build_document_xml(records: list[dict]) -> str:
    records_by_name = {record["name"]: record for record in records}
    best = max(records, key=lambda x: (x["test_acc"], x["macro_f1"]))

    body_parts = [
        paragraph("基于 outputs 的实验版本说明与消融对比", bold=True, size=28, center=True),
        paragraph("来源：2026APMCM/outputs", size=20, center=True),
        paragraph("生成时间：2026-07-19", size=20, center=True),
        paragraph(
            f"当前所有正式版本中，综合表现最好的实验是 {best['name']}，测试准确率 {pct(best['test_acc'])}，Macro F1 为 {float4(best['macro_f1'])}。",
            size=21,
        ),
        paragraph("一、逐版本说明", bold=True, size=24),
    ]

    for record in records:
        body_parts.append(paragraph(record["name"], bold=True, size=22))
        body_parts.append(build_version_paragraph(record, records_by_name))

    body_parts.extend([
        paragraph("二、全量模型对照表", bold=True, size=24),
        build_summary_table(records, records_by_name),
        paragraph("三、详细实验对比结论", bold=True, size=24),
    ])
    body_parts.extend(build_conclusion_paragraphs(records_by_name))
    body_parts.extend(build_chart_perspective_paragraphs(records_by_name))

    body = "".join(body_parts) + (
        "<w:sectPr>"
        "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"1080\" w:right=\"1080\" w:bottom=\"1080\" w:left=\"1080\" "
        "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 wp14\">"
        f"<w:body>{body}</w:body>"
        "</w:document>"
    )


def build_core_xml() -> str:
    now = datetime(2026, 7, 19, 14, 10, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
        "xmlns:dc=\"http://purl.org/dc/elements/1.1/\" "
        "xmlns:dcterms=\"http://purl.org/dc/terms/\" "
        "xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" "
        "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">"
        "<dc:title>实验版本说明与消融对比</dc:title>"
        "<dc:creator>Codex</dc:creator>"
        "<cp:lastModifiedBy>Codex</cp:lastModifiedBy>"
        f"<dcterms:created xsi:type=\"dcterms:W3CDTF\">{now}</dcterms:created>"
        f"<dcterms:modified xsi:type=\"dcterms:W3CDTF\">{now}</dcterms:modified>"
        "</cp:coreProperties>"
    )


def build_docx(document_xml: str) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""
    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
"""
    app_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
</Properties>
"""
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
  </w:style>
</w:styles>
"""

    with ZipFile(DOC_PATH, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("docProps/core.xml", build_core_xml())
        zf.writestr("docProps/app.xml", app_xml)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/styles.xml", styles_xml)
        zf.writestr("word/_rels/document.xml.rels", document_rels)


def main() -> None:
    records = [get_record(meta) for meta in VERSION_META]
    document_xml = build_document_xml(records)
    build_docx(document_xml)
    print(DOC_PATH)


if __name__ == "__main__":
    main()
