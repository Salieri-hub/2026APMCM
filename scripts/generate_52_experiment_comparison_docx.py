from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
DOC_DIR = ROOT / "doc"
DOC_PATH = DOC_DIR / "all_52_experiments_comparison_20260719.docx"

BASE_RUNS = [
    "v1.0_scratch_ce_cpu",
    "v1.1_scratch_ce_cuda",
    "v2.0_pretrained_ce",
    "v2.1_pretrained_ce_cosine",
    "v2.2_pretrained_ce_ls",
    "v2.3_pretrained_ce_ls_cosine",
    "v2.4_pretrained_ce_ls_cosine_weightedce",
    "v3.0_pretrained_focal_ls_cosine",
    "v3.1_pretrained_focal_ls_cosine_mixup",
    "v3.2_pretrained_focal_ls_cosine_cutmix",
    "v3.3_pretrained_focal_ls_cosine_se",
    "v3.4_pretrained_focal_ls_cosine_cbam",
]

MAIN_RUNS = [name for name in BASE_RUNS if name.startswith("v2.") or name.startswith("v3.")]
PAIR_TAGS = ["ad_lc", "ad_sq", "lc_sq"]
PAIR_LABELS = {
    "ad_lc": "腺癌-大细胞癌专家",
    "ad_sq": "腺癌-鳞癌专家",
    "lc_sq": "大细胞癌-鳞癌专家",
}


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
    for row_index, row in enumerate(rows):
        cells = []
        for cell in row:
            content = paragraph(cell, bold=row_index == 0, size=18)
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


def load_metrics(run_name: str) -> dict:
    path = OUTPUTS_DIR / run_name / "metrics_summary.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_record(run_name: str) -> dict:
    metrics = load_metrics(run_name)
    report = metrics["test"]["report"]
    record = {
        "name": run_name,
        "metrics": metrics,
        "acc": metrics["test"]["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    if run_name in BASE_RUNS:
        record["group"] = "原始12次"
        record["main_run"] = run_name
        record["expert_type"] = "-"
        record["delta_acc"] = 0.0
        record["delta_f1"] = 0.0
        record["expert_invocations"] = None
        record["expert_corrected"] = None
        record["expert_hurt"] = None
    elif run_name.startswith("cascade_pair_"):
        remainder = run_name[len("cascade_pair_") :]
        pair_tag = next((tag for tag in PAIR_TAGS if remainder.startswith(tag + "_")), None)
        if pair_tag is None:
            raise ValueError(f"Cannot parse pairwise cascade run name: {run_name}")
        main_run = remainder[len(pair_tag) + 1 :]
        base = load_metrics(main_run)
        base_report = base["test"]["report"]
        record["group"] = "二分支专家30次"
        record["main_run"] = main_run
        record["expert_type"] = PAIR_LABELS[pair_tag]
        record["delta_acc"] = metrics["test"]["accuracy"] - base["test"]["accuracy"]
        record["delta_f1"] = report["macro avg"]["f1-score"] - base_report["macro avg"]["f1-score"]
        record["expert_invocations"] = metrics["test"]["cascade_stats"]["expert_invocations"]
        record["expert_corrected"] = metrics["test"]["cascade_stats"]["expert_corrected_predictions"]
        record["expert_hurt"] = metrics["test"]["cascade_stats"]["expert_hurt_predictions"]
    elif run_name.startswith("cascade_"):
        main_run = run_name[len("cascade_") :]
        base = load_metrics(main_run)
        base_report = base["test"]["report"]
        record["group"] = "三分支专家10次"
        record["main_run"] = main_run
        record["expert_type"] = "三肿瘤专家"
        record["delta_acc"] = metrics["test"]["accuracy"] - base["test"]["accuracy"]
        record["delta_f1"] = report["macro avg"]["f1-score"] - base_report["macro avg"]["f1-score"]
        record["expert_invocations"] = metrics["test"]["cascade_stats"]["expert_invocations"]
        record["expert_corrected"] = metrics["test"]["cascade_stats"]["expert_corrected_predictions"]
        record["expert_hurt"] = metrics["test"]["cascade_stats"]["expert_hurt_predictions"]
    else:
        raise ValueError(f"Unsupported run name: {run_name}")
    return record


def collect_records() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    base_records = [build_record(name) for name in BASE_RUNS]
    tumor3_records = [build_record(f"cascade_{name}") for name in MAIN_RUNS]
    pair_records = [build_record(f"cascade_pair_{tag}_{name}") for tag in PAIR_TAGS for name in MAIN_RUNS]
    all_records = base_records + tumor3_records + pair_records
    if len(base_records) != 12 or len(tumor3_records) != 10 or len(pair_records) != 30 or len(all_records) != 52:
        raise ValueError(
            f"Unexpected record counts: base={len(base_records)}, tumor3={len(tumor3_records)}, "
            f"pair={len(pair_records)}, total={len(all_records)}"
        )
    return base_records, tumor3_records, pair_records, all_records


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_group_summary_table(base_records: list[dict], tumor3_records: list[dict], pair_records: list[dict]) -> str:
    groups = [
        ("原始12次", base_records),
        ("三分支专家10次", tumor3_records),
        ("二分支专家30次", pair_records),
    ]
    rows = [["实验组", "数量", "最佳实验", "最佳Acc", "最佳Macro F1", "平均Acc", "平均Macro F1"]]
    for label, records in groups:
        best = max(records, key=lambda r: (r["acc"], r["macro_f1"]))
        rows.append(
            [
                label,
                str(len(records)),
                best["name"],
                pct(best["acc"]),
                float4(best["macro_f1"]),
                pct(mean([r["acc"] for r in records])),
                float4(mean([r["macro_f1"] for r in records])),
            ]
        )
    return table(rows, [2200, 900, 4200, 1300, 1500, 1300, 1600])


def build_base_table(records: list[dict]) -> str:
    rows = [["实验", "Test Acc", "Macro F1", "说明"]]
    notes = {
        "v1.0_scratch_ce_cpu": "最初 CPU baseline",
        "v1.1_scratch_ce_cuda": "仅切换到 CUDA",
        "v2.0_pretrained_ce": "启用预训练",
        "v2.1_pretrained_ce_cosine": "预训练 + cosine",
        "v2.2_pretrained_ce_ls": "预训练 + label smoothing",
        "v2.3_pretrained_ce_ls_cosine": "label smoothing + cosine",
        "v2.4_pretrained_ce_ls_cosine_weightedce": "再加 balanced weighted CE",
        "v3.0_pretrained_focal_ls_cosine": "换成 focal loss",
        "v3.1_pretrained_focal_ls_cosine_mixup": "再加 MixUp",
        "v3.2_pretrained_focal_ls_cosine_cutmix": "再加 CutMix",
        "v3.3_pretrained_focal_ls_cosine_se": "再加 SE",
        "v3.4_pretrained_focal_ls_cosine_cbam": "再加 CBAM",
    }
    for record in records:
        rows.append([record["name"], pct(record["acc"]), float4(record["macro_f1"]), notes[record["name"]]])
    return table(rows, [3600, 1300, 1500, 3600])


def build_tumor3_table(records: list[dict]) -> str:
    rows = [["实验", "对应主模型", "Test Acc", "Macro F1", "相对主模型Acc变化", "相对主模型F1变化", "专家触发次数"]]
    for record in records:
        rows.append(
            [
                record["name"],
                record["main_run"],
                pct(record["acc"]),
                float4(record["macro_f1"]),
                delta_pct(record["acc"], record["acc"] - record["delta_acc"]),
                delta_float(record["macro_f1"], record["macro_f1"] - record["delta_f1"]),
                str(record["expert_invocations"]),
            ]
        )
    return table(rows, [3800, 3000, 1200, 1400, 2100, 1800, 1600])


def build_pair_table(records: list[dict]) -> str:
    rows = [["实验", "对应主模型", "专家类型", "Test Acc", "Macro F1", "相对主模型Acc变化", "专家触发次数"]]
    for record in records:
        rows.append(
            [
                record["name"],
                record["main_run"],
                record["expert_type"],
                pct(record["acc"]),
                float4(record["macro_f1"]),
                delta_pct(record["acc"], record["acc"] - record["delta_acc"]),
                str(record["expert_invocations"]),
            ]
        )
    return table(rows, [3600, 2800, 2200, 1200, 1400, 2000, 1500])


def build_top_ranking_table(all_records: list[dict], top_k: int = 15) -> str:
    ranked = sorted(all_records, key=lambda r: (r["acc"], r["macro_f1"]), reverse=True)[:top_k]
    rows = [["排名", "实验", "实验组", "专家类型", "Test Acc", "Macro F1", "相对主模型Acc变化"]]
    for index, record in enumerate(ranked, start=1):
        rows.append(
            [
                str(index),
                record["name"],
                record["group"],
                record["expert_type"],
                pct(record["acc"]),
                float4(record["macro_f1"]),
                delta_pct(record["acc"], record["acc"] - record["delta_acc"]) if record["group"] != "原始12次" else "-",
            ]
        )
    return table(rows, [900, 4200, 1800, 2200, 1200, 1400, 1900])


def build_best_cascade_table(base_records: list[dict], tumor3_records: list[dict], pair_records: list[dict]) -> str:
    base_by_name = {record["name"]: record for record in base_records}
    rows = [["主模型", "主模型Acc / F1", "最佳级联实验", "级联类型", "级联Acc / F1", "Acc变化", "F1变化"]]
    for main_run in MAIN_RUNS:
        candidates = [record for record in tumor3_records + pair_records if record["main_run"] == main_run]
        best = max(candidates, key=lambda r: (r["acc"], r["macro_f1"]))
        base = base_by_name[main_run]
        rows.append(
            [
                main_run,
                f"{pct(base['acc'])} / {float4(base['macro_f1'])}",
                best["name"],
                best["expert_type"],
                f"{pct(best['acc'])} / {float4(best['macro_f1'])}",
                delta_pct(best["acc"], base["acc"]),
                delta_float(best["macro_f1"], base["macro_f1"]),
            ]
        )
    return table(rows, [2800, 2000, 4200, 2200, 2100, 1600, 1500])


def build_document_xml(base_records: list[dict], tumor3_records: list[dict], pair_records: list[dict], all_records: list[dict]) -> str:
    best_base = max(base_records, key=lambda r: (r["acc"], r["macro_f1"]))
    best_tumor3 = max(tumor3_records, key=lambda r: (r["acc"], r["macro_f1"]))
    best_pair = max(pair_records, key=lambda r: (r["acc"], r["macro_f1"]))
    best_overall = max(all_records, key=lambda r: (r["acc"], r["macro_f1"]))

    tumor3_better = sum(1 for r in tumor3_records if (r["delta_acc"], r["delta_f1"]) > (0.0, 0.0))
    pair_better = sum(1 for r in pair_records if (r["delta_acc"], r["delta_f1"]) > (0.0, 0.0))

    pair_avg = {
        tag: (
            mean([r["acc"] for r in pair_records if f"cascade_pair_{tag}_" in r["name"]]),
            mean([r["macro_f1"] for r in pair_records if f"cascade_pair_{tag}_" in r["name"]]),
            sum(1 for r in pair_records if f"cascade_pair_{tag}_" in r["name"] and (r["delta_acc"], r["delta_f1"]) <= (0.0, 0.0)),
        )
        for tag in PAIR_TAGS
    }

    top5 = sorted(all_records, key=lambda r: (r["acc"], r["macro_f1"]), reverse=True)[:5]
    best_overall_stats = best_overall["metrics"]["test"]["cascade_stats"]

    body_parts = [
        paragraph("52 组实验结果总对比分析", bold=True, size=28, center=True),
        paragraph("基于 outputs_实验版本说明与消融对比_20260719.docx 的扩展分析", size=20, center=True),
        paragraph("生成时间：2026-07-19", size=20, center=True),
        paragraph(
            "本报告在原有 12 次正式消融实验分析基础上，进一步纳入 30 次二分支专家级联实验和 10 次三分支专家级联实验，共比较 52 组实验结果。",
            size=21,
        ),
        paragraph("一、实验范围与分组", bold=True, size=24),
        paragraph(
            "52 组实验由三部分构成：原始 12 次正式实验、10 次三肿瘤专家级联实验、30 次两两肿瘤专家级联实验。"
            "这里的三肿瘤专家是指 adenocarcinoma / large.cell.carcinoma / squamous.cell.carcinoma 三分类专家分支；"
            "两两肿瘤专家则分为腺癌-大细胞癌、腺癌-鳞癌、大细胞癌-鳞癌三种二分类专家分支。",
            size=21,
        ),
        build_group_summary_table(base_records, tumor3_records, pair_records),
        paragraph("二、原始 12 次实验结果", bold=True, size=24),
        paragraph(
            f"原始 12 次实验中，最佳结果仍然来自 {best_base['name']}，测试准确率 {pct(best_base['acc'])}，Macro F1 为 {float4(best_base['macro_f1'])}。"
            "这说明在不启用级联的前提下，CBAM 仍然是当前主线中最有效的结构增强方案。",
            size=21,
        ),
        build_base_table(base_records),
        paragraph("三、10 次三分支专家级联结果", bold=True, size=24),
        paragraph(
            f"10 次三分支专家级联中，最佳结果来自 {best_tumor3['name']}，测试准确率 {pct(best_tumor3['acc'])}，Macro F1 为 {float4(best_tumor3['macro_f1'])}。"
            f"它也是全部 52 组实验中的全局最优。三分支级联中共有 {tumor3_better}/10 次相对各自主模型取得提升，"
            "说明三肿瘤专家在大多数情况下都能对主模型提供补充，但这种收益并不是无条件存在的。",
            size=21,
        ),
        build_tumor3_table(tumor3_records),
        paragraph("四、30 次两两肿瘤专家级联结果", bold=True, size=24),
        paragraph(
            f"30 次两两肿瘤专家级联中，最佳结果来自 {best_pair['name']}，测试准确率 {pct(best_pair['acc'])}，Macro F1 为 {float4(best_pair['macro_f1'])}。"
            f"二分支级联共有 {pair_better}/30 次相对各自主模型取得提升，整体也明显优于原始 12 次实验平均水平。",
            size=21,
        ),
        build_pair_table(pair_records),
        paragraph("五、总排名与关键发现", bold=True, size=24),
        paragraph(
            f"全部 52 组实验的第一名是 {best_overall['name']}，测试准确率 {pct(best_overall['acc'])}，Macro F1 为 {float4(best_overall['macro_f1'])}。"
            f"这个最佳实验一共触发专家模型 {best_overall_stats['expert_invocations']} 次，其中修正了 {best_overall_stats['expert_corrected_predictions']} 次主模型错误，"
            f"仅带来 {best_overall_stats['expert_hurt_predictions']} 次负面改动，说明在当前触发规则下，级联调用次数并不高，但只要命中关键样本就能带来净收益。",
            size=21,
        ),
        build_top_ranking_table(all_records, top_k=15),
        paragraph(
            "从前 15 名可以看出两个明显规律。第一，前 5 名全部来自 v3.4_pretrained_focal_ls_cosine_cbam 主模型路线，说明最强主模型仍然决定了最终天花板。"
            "第二，前 3 名全部是级联版本，说明在最强主模型基础上继续加专家分支，仍有进一步提升空间。",
            size=21,
        ),
        paragraph("六、每个主模型的最佳级联版本", bold=True, size=24),
        paragraph(
            "下面这张表以主模型为单位，列出每个主模型在四种级联策略（三肿瘤专家 + 三种两两肿瘤专家）中表现最好的版本，"
            "方便直接判断“是否值得级联”和“哪一种专家分支更适合该主模型”。",
            size=21,
        ),
        build_best_cascade_table(base_records, tumor3_records, pair_records),
        paragraph("七、综合结论", bold=True, size=24),
        paragraph(
            f"1. 从平均水平看，原始 12 次实验的平均测试准确率为 {pct(mean([r['acc'] for r in base_records]))}，"
            f"平均 Macro F1 为 {float4(mean([r['macro_f1'] for r in base_records]))}；"
            f"三分支专家 10 次实验的平均测试准确率为 {pct(mean([r['acc'] for r in tumor3_records]))}，平均 Macro F1 为 {float4(mean([r['macro_f1'] for r in tumor3_records]))}；"
            f"二分支专家 30 次实验的平均测试准确率为 {pct(mean([r['acc'] for r in pair_records]))}，平均 Macro F1 为 {float4(mean([r['macro_f1'] for r in pair_records]))}。"
            "因此，无论看最优值还是看平均值，加入专家分支后的级联方案整体都优于单模型方案。",
            size=21,
        ),
        paragraph(
            "2. 三分支专家更稳定。它在 10 个主模型上有 8 次取得提升，仅在 v2.1_pretrained_ce_cosine 和 v2.3_pretrained_ce_ls_cosine 上略有退步。"
            "相比之下，两两肿瘤专家虽然也能带来不少提升，但总体波动更大，说明更细粒度的二分类专家更依赖于主模型本身的边界质量和触发时机。",
            size=21,
        ),
        paragraph(
            f"3. 在三种二分支专家中，平均表现最好的是“大细胞癌-鳞癌专家”，平均测试准确率 {pct(pair_avg['lc_sq'][0])}、平均 Macro F1 {float4(pair_avg['lc_sq'][1])}，"
            f"并且只有 {pair_avg['lc_sq'][2]} 次不优于主模型；最不稳定的是“腺癌-鳞癌专家”，其不优于主模型的次数达到 {pair_avg['ad_sq'][2]} 次。"
            "这说明不同专家子任务的收益并不一致，当前最有效的二分类专家更偏向修正大细胞癌与鳞癌之间的边界。",
            size=21,
        ),
        paragraph(
            "4. Cascade 的收益并不是均匀分布的。对本来就较强的主模型，例如 v2.2_pretrained_ce_ls 和 v3.4_pretrained_focal_ls_cosine_cbam，级联带来的提升虽然不算巨大，但可以把结果继续推到新的最优。"
            "对原本性能较弱的 v3.1_mixup 和 v3.2_cutmix，级联也能明显回收一部分损失，说明专家分支特别适合修补主模型在肿瘤子类边界上的系统性弱点。",
            size=21,
        ),
        paragraph(
            "5. 当前 52 组实验的最终建议是：如果只选一个最优方案，优先使用三分支专家级联的 cascade_v3.4_pretrained_focal_ls_cosine_cbam；"
            "如果希望在结构更简单的前提下保留较强效果，则可以考虑主模型 v3.4_pretrained_focal_ls_cosine_cbam 本体，或二分支中的 cascade_pair_lc_sq_v3.4_pretrained_focal_ls_cosine_cbam。"
            "如果后续还要继续扩展实验，下一步最值得尝试的是围绕 v3.4 主线继续优化专家触发规则，而不是重新回到较弱的主模型家族。",
            size=21,
        ),
        paragraph(
            "附加观察：前 5 名实验依次为 "
            + "、".join(item["name"] for item in top5)
            + "。这进一步说明当前实验的有效增量主要集中在“强主模型 + 有针对性的专家级联”这一方向上。",
            size=21,
        ),
    ]

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
    now = datetime(2026, 7, 19, 20, 55, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
        "xmlns:dc=\"http://purl.org/dc/elements/1.1/\" "
        "xmlns:dcterms=\"http://purl.org/dc/terms/\" "
        "xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" "
        "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">"
        "<dc:title>52组实验结果总对比分析</dc:title>"
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
    base_records, tumor3_records, pair_records, all_records = collect_records()
    document_xml = build_document_xml(base_records, tumor3_records, pair_records, all_records)
    build_docx(document_xml)
    print(DOC_PATH)


if __name__ == "__main__":
    main()
