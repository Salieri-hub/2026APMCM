from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile


DOC_DATE = "20260721"
DOC_NAME = f"b3_50_experiments_comparison_{DOC_DATE}.docx"
RUN_SUFFIX = "_b3"


@dataclass(frozen=True)
class RunResult:
    name: str
    category: str
    pair_label: str | None
    val_acc: float
    test_acc: float
    macro_f1: float
    weighted_f1: float
    best_epoch: int | None
    expert_invocations: int | None
    expert_changed: int | None
    expert_corrected: int | None
    expert_hurt: int | None


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def f1(value: float) -> str:
    return f"{value:.4f}"


def family_order() -> list[str]:
    return [
        "v2.0_pretrained_ce_b3",
        "v2.1_pretrained_ce_cosine_b3",
        "v2.2_pretrained_ce_ls_b3",
        "v2.3_pretrained_ce_ls_cosine_b3",
        "v2.4_pretrained_ce_ls_cosine_weightedce_b3",
        "v3.0_pretrained_focal_ls_cosine_b3",
        "v3.1_pretrained_focal_ls_cosine_mixup_b3",
        "v3.2_pretrained_focal_ls_cosine_cutmix_b3",
        "v3.3_pretrained_focal_ls_cosine_se_b3",
        "v3.4_pretrained_focal_ls_cosine_cbam_b3",
    ]


def expected_formal_run_names() -> list[str]:
    names: list[str] = []
    for base in family_order():
        names.append(base)
        names.append(f"cascade_{base}")
        names.append(f"cascade_pair_ad_lc_{base}")
        names.append(f"cascade_pair_ad_sq_{base}")
        names.append(f"cascade_pair_lc_sq_{base}")
    return names


def base_name_from_variant(run_name: str) -> str:
    prefixes = [
        "cascade_pair_ad_lc_",
        "cascade_pair_ad_sq_",
        "cascade_pair_lc_sq_",
        "cascade_",
    ]
    for prefix in prefixes:
        if run_name.startswith(prefix):
            return run_name[len(prefix) :]
    return run_name


def detect_category(run_name: str) -> tuple[str, str | None]:
    if run_name.startswith("cascade_pair_ad_lc_"):
        return "两两专家级联", "腺癌 vs 大细胞癌"
    if run_name.startswith("cascade_pair_ad_sq_"):
        return "两两专家级联", "腺癌 vs 鳞癌"
    if run_name.startswith("cascade_pair_lc_sq_"):
        return "两两专家级联", "大细胞癌 vs 鳞癌"
    if run_name.startswith("cascade_"):
        return "三分类肿瘤级联", None
    return "单模型", None


def load_formal_results(results_root: Path) -> list[RunResult]:
    rows: list[RunResult] = []
    for metrics_path in sorted(results_root.glob(f"*{RUN_SUFFIX}/metrics_summary.json")):
        run_name = metrics_path.parent.name
        if run_name.startswith("expert_"):
            continue

        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        category, pair_label = detect_category(run_name)
        if data["mode"] == "single":
            val_acc = data["best_validation"]["accuracy"]
            best_epoch = data["best_epoch"]
            cascade_stats = None
        else:
            val_acc = data["validation"]["accuracy"]
            best_epoch = data.get("main_model", {}).get("best_epoch")
            cascade_stats = data["test"].get("cascade_stats")

        rows.append(
            RunResult(
                name=run_name,
                category=category,
                pair_label=pair_label,
                val_acc=val_acc,
                test_acc=data["test"]["accuracy"],
                macro_f1=data["test"]["report"]["macro avg"]["f1-score"],
                weighted_f1=data["test"]["report"]["weighted avg"]["f1-score"],
                best_epoch=best_epoch,
                expert_invocations=None if cascade_stats is None else cascade_stats.get("expert_invocations"),
                expert_changed=None if cascade_stats is None else cascade_stats.get("expert_changed_predictions"),
                expert_corrected=None if cascade_stats is None else cascade_stats.get("expert_corrected_predictions"),
                expert_hurt=None if cascade_stats is None else cascade_stats.get("expert_hurt_predictions"),
            )
        )

    expected = set(expected_formal_run_names())
    actual = {row.name for row in rows}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        message = []
        if missing:
            message.append(f"缺少 {len(missing)} 个正式实验结果：{missing}")
        if extra:
            message.append(f"发现 {len(extra)} 个未识别结果：{extra}")
        raise SystemExit("；".join(message))

    if len(rows) != 50:
        raise SystemExit(f"预期读取 50 个正式实验结果，实际读取到 {len(rows)} 个。")
    return rows


def make_paragraph(text: str, *, bold: bool = False, size: int = 22, align: str | None = None) -> str:
    ppr: list[str] = []
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    ppr.append('<w:spacing w:after="120"/>')
    rpr: list[str] = []
    if bold:
        rpr.append("<w:b/>")
    rpr.append(f'<w:sz w:val="{size}"/>')
    rpr.append(f'<w:szCs w:val="{size}"/>')
    return (
        "<w:p>"
        + ("<w:pPr>" + "".join(ppr) + "</w:pPr>" if ppr else "")
        + "<w:r><w:rPr>"
        + "".join(rpr)
        + "</w:rPr>"
        + f'<w:t xml:space="preserve">{escape(text)}</w:t>'
        + "</w:r></w:p>"
    )


def make_page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def make_table(rows: list[list[str]], widths: list[int]) -> str:
    table_parts = [
        "<w:tbl>",
        (
            "<w:tblPr>"
            '<w:tblW w:w="0" w:type="auto"/>'
            "<w:tblBorders>"
            '<w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            '<w:left w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            '<w:right w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="808080"/>'
            '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="808080"/>'
            "</w:tblBorders>"
            "</w:tblPr>"
        ),
        "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>",
    ]
    for row_index, row in enumerate(rows):
        table_parts.append("<w:tr>")
        for col_index, cell in enumerate(row):
            shade = '<w:shd w:fill="D9EAF7"/>' if row_index == 0 else ""
            table_parts.append(
                "<w:tc>"
                + f'<w:tcPr><w:tcW w:w="{widths[col_index]}" w:type="dxa"/>{shade}</w:tcPr>'
                + make_paragraph(cell, bold=(row_index == 0), size=20 if row_index == 0 else 18)
                + "</w:tc>"
            )
        table_parts.append("</w:tr>")
    table_parts.append("</w:tbl>")
    return "".join(table_parts)


def build_document_xml(results: list[RunResult]) -> str:
    by_name = {row.name: row for row in results}
    families = family_order()

    singles = [by_name[name] for name in families]
    tumor3 = [by_name[f"cascade_{name}"] for name in families]
    pair_ad_lc = [by_name[f"cascade_pair_ad_lc_{name}"] for name in families]
    pair_ad_sq = [by_name[f"cascade_pair_ad_sq_{name}"] for name in families]
    pair_lc_sq = [by_name[f"cascade_pair_lc_sq_{name}"] for name in families]
    pairs = pair_ad_lc + pair_ad_sq + pair_lc_sq

    overall_best = max(results, key=lambda item: (item.test_acc, item.macro_f1, item.val_acc))
    best_single = max(singles, key=lambda item: (item.test_acc, item.macro_f1, item.val_acc))
    best_tumor3 = max(tumor3, key=lambda item: (item.test_acc, item.macro_f1, item.val_acc))
    best_pair = max(pairs, key=lambda item: (item.test_acc, item.macro_f1, item.val_acc))

    avg_single_acc = mean(row.test_acc for row in singles)
    avg_single_macro = mean(row.macro_f1 for row in singles)
    avg_tumor3_acc = mean(row.test_acc for row in tumor3)
    avg_tumor3_macro = mean(row.macro_f1 for row in tumor3)
    avg_pair_acc = mean(row.test_acc for row in pairs)
    avg_pair_macro = mean(row.macro_f1 for row in pairs)

    avg_ad_lc = mean(row.test_acc for row in pair_ad_lc)
    avg_ad_sq = mean(row.test_acc for row in pair_ad_sq)
    avg_lc_sq = mean(row.test_acc for row in pair_lc_sq)
    pair_tag_best_label, pair_tag_best_value = max(
        [
            ("腺癌 vs 大细胞癌", avg_ad_lc),
            ("腺癌 vs 鳞癌", avg_ad_sq),
            ("大细胞癌 vs 鳞癌", avg_lc_sq),
        ],
        key=lambda item: item[1],
    )

    tumor3_better = sum(1 for single, casc in zip(singles, tumor3) if casc.test_acc > single.test_acc)
    pair_better = 0
    family_rows = [["主线版本", "单模型", "三分类级联", "最佳两两级联", "家族最佳结论"]]
    for base in families:
        single = by_name[base]
        casc3 = by_name[f"cascade_{base}"]
        pair_candidates = [
            by_name[f"cascade_pair_ad_lc_{base}"],
            by_name[f"cascade_pair_ad_sq_{base}"],
            by_name[f"cascade_pair_lc_sq_{base}"],
        ]
        best_family_pair = max(pair_candidates, key=lambda item: (item.test_acc, item.macro_f1, item.val_acc))
        family_best = max([single, casc3, *pair_candidates], key=lambda item: (item.test_acc, item.macro_f1, item.val_acc))
        if best_family_pair.test_acc > single.test_acc:
            pair_better += 1
        family_rows.append(
            [
                base,
                pct(single.test_acc),
                pct(casc3.test_acc),
                f"{best_family_pair.pair_label} / {pct(best_family_pair.test_acc)}",
                f"{family_best.name}（{family_best.category}）",
            ]
        )

    top10_rows = [["排名", "实验名", "类别", "验证准确率", "测试准确率", "Macro F1"]]
    top10_results = sorted(results, key=lambda item: (-item.test_acc, -item.macro_f1, -item.val_acc, item.name))[:10]
    for index, row in enumerate(top10_results, start=1):
        label = row.pair_label if row.pair_label else row.category
        top10_rows.append([str(index), row.name, label, pct(row.val_acc), pct(row.test_acc), f1(row.macro_f1)])

    single_rows = [["实验名", "验证准确率", "测试准确率", "Macro F1", "Weighted F1", "Best Epoch"]]
    for row in singles:
        single_rows.append(
            [
                row.name,
                pct(row.val_acc),
                pct(row.test_acc),
                f1(row.macro_f1),
                f1(row.weighted_f1),
                str(row.best_epoch or ""),
            ]
        )

    tumor3_rows = [["实验名", "验证准确率", "测试准确率", "Macro F1", "专家调用", "纠正/伤害"]]
    for row in tumor3:
        tumor3_rows.append(
            [
                row.name,
                pct(row.val_acc),
                pct(row.test_acc),
                f1(row.macro_f1),
                str(row.expert_invocations or 0),
                f"{row.expert_corrected or 0}/{row.expert_hurt or 0}",
            ]
        )

    pair_rows = [["实验名", "专家对", "验证准确率", "测试准确率", "Macro F1", "纠正/伤害"]]
    for row in sorted(pairs, key=lambda item: (base_name_from_variant(item.name), item.pair_label or "")):
        pair_rows.append(
            [
                row.name,
                row.pair_label or "",
                pct(row.val_acc),
                pct(row.test_acc),
                f1(row.macro_f1),
                f"{row.expert_corrected or 0}/{row.expert_hurt or 0}",
            ]
        )

    pair_group_rows = [["专家对", "平均测试准确率", "最高测试准确率", "最低测试准确率"]]
    for label, group in [
        ("腺癌 vs 大细胞癌", pair_ad_lc),
        ("腺癌 vs 鳞癌", pair_ad_sq),
        ("大细胞癌 vs 鳞癌", pair_lc_sq),
    ]:
        pair_group_rows.append(
            [
                label,
                pct(mean(row.test_acc for row in group)),
                pct(max(row.test_acc for row in group)),
                pct(min(row.test_acc for row in group)),
            ]
        )

    body: list[str] = []
    body.append(make_paragraph("EfficientNet-B3 50组正式实验结果对比", bold=True, size=32, align="center"))
    body.append(
        make_paragraph(
            "统计范围：10组单模型 + 10组三分类肿瘤级联 + 30组两两专家级联，共50组正式四分类结果；40组 expert_* 训练结果不纳入本文主表。",
            size=20,
        )
    )

    body.append(make_paragraph("一、核心结论", bold=True, size=26))
    body.append(
        make_paragraph(
            f"1. 全局最佳实验为 {overall_best.name}，测试准确率 {pct(overall_best.test_acc)}，Macro F1 = {f1(overall_best.macro_f1)}。",
            size=20,
        )
    )
    body.append(
        make_paragraph(
            f"2. 最佳单模型为 {best_single.name}；最佳三分类级联为 {best_tumor3.name}；最佳两两级联为 {best_pair.name}。",
            size=20,
        )
    )
    body.append(
        make_paragraph(
            f"3. 分组平均表现为：单模型 {pct(avg_single_acc)} / {f1(avg_single_macro)}，三分类级联 {pct(avg_tumor3_acc)} / {f1(avg_tumor3_macro)}，两两级联 {pct(avg_pair_acc)} / {f1(avg_pair_macro)}。",
            size=20,
        )
    )
    body.append(
        make_paragraph(
            f"4. 在10个主线家族中，三分类级联有 {tumor3_better}/10 个家族优于对应单模型；每个家族中的最佳两两级联有 {pair_better}/10 个家族优于对应单模型。",
            size=20,
        )
    )
    body.append(
        make_paragraph(
            f"5. 三类两两专家对中，平均测试准确率最高的是“{pair_tag_best_label}”，平均值为 {pct(pair_tag_best_value)}。",
            size=20,
        )
    )

    body.append(make_paragraph("二、Top 10 总排名", bold=True, size=26))
    body.append(make_table(top10_rows, [700, 4300, 1800, 1300, 1300, 1100]))

    body.append(make_page_break())
    body.append(make_paragraph("三、10组单模型结果", bold=True, size=26))
    body.append(make_table(single_rows, [4200, 1300, 1300, 1100, 1200, 1000]))

    body.append(make_paragraph("四、10组三分类肿瘤级联结果", bold=True, size=26))
    body.append(make_table(tumor3_rows, [4100, 1200, 1200, 1100, 1100, 1200]))

    body.append(make_page_break())
    body.append(make_paragraph("五、30组两两专家级联结果", bold=True, size=26))
    body.append(make_table(pair_rows, [4100, 1800, 1200, 1200, 1100, 1200]))

    body.append(make_page_break())
    body.append(make_paragraph("六、按主线版本的家族对照", bold=True, size=26))
    body.append(make_table(family_rows, [3200, 1200, 1400, 2200, 3600]))

    body.append(make_paragraph("七、按专家对的平均表现", bold=True, size=26))
    body.append(make_table(pair_group_rows, [2400, 1800, 1800, 1800]))

    body.append(make_paragraph("八、实验解释", bold=True, size=26))
    body.append(
        make_paragraph(
            "1. 同一家族内的5组结果可以直接对照，因为它们的主干、损失函数、调度器和增强策略一致，只在是否启用专家分支及专家划分方式上不同。",
            size=20,
        )
    )
    body.append(
        make_paragraph(
            "2. 单模型、三分类级联和两两级联三类结果同时保留，有助于区分“训练策略本身有效”还是“级联结构带来额外收益”。",
            size=20,
        )
    )
    body.append(
        make_paragraph(
            "3. 如果后续只保留少量候选方案继续深入，优先检查全局最优家族、最佳单模型家族以及平均表现最好的两两专家对。",
            size=20,
        )
    )

    body.append(
        '<w:sectPr>'
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
    )

    body_xml = "".join(body)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14">'
        f"<w:body>{body_xml}</w:body></w:document>"
    )


def write_docx(output_path: Path, document_xml: str) -> None:
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zip_file:
        zip_file.writestr("[Content_Types].xml", content_types)
        zip_file.writestr("_rels/.rels", rels)
        zip_file.writestr("word/document.xml", document_xml)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "outputs" / "results"
    doc_root = repo_root / "doc"
    output_path = doc_root / DOC_NAME

    results = load_formal_results(results_root)
    document_xml = build_document_xml(results)
    write_docx(output_path, document_xml)

    overall_best = max(results, key=lambda item: (item.test_acc, item.macro_f1, item.val_acc))
    print(f"Generated: {output_path}")
    print(
        "Best overall: "
        f"{overall_best.name} | test_acc={overall_best.test_acc:.6f} | macro_f1={overall_best.macro_f1:.6f}"
    )


if __name__ == "__main__":
    main()
