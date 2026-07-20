from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile


DOC_DATE = "20260720"
DOC_NAME = f"b2_50_experiments_comparison_{DOC_DATE}.docx"


@dataclass
class RunResult:
    name: str
    category: str
    pair_tag: str | None
    val_acc: float
    test_acc: float
    macro_f1: float
    weighted_f1: float
    best_epoch: int | None
    note: str


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def f1(value: float) -> str:
    return f"{value:.4f}"


def detect_category(name: str) -> tuple[str, str | None]:
    if name.startswith("cascade_pair_"):
        if "_ad_lc_" in name:
            return "两两专家级联", "腺癌 vs 大细胞癌"
        if "_ad_sq_" in name:
            return "两两专家级联", "腺癌 vs 鳞癌"
        if "_lc_sq_" in name:
            return "两两专家级联", "大细胞癌 vs 鳞癌"
        return "两两专家级联", None
    if name.startswith("cascade_"):
        return "三肿瘤专家级联", None
    return "单模型", None


def load_formal_results(results_root: Path) -> list[RunResult]:
    rows: list[RunResult] = []
    for metrics_path in sorted(results_root.glob("*_b2/metrics_summary.json")):
        run_name = metrics_path.parent.name
        if run_name.startswith("expert_"):
            continue

        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        category, pair_tag = detect_category(run_name)

        if data["mode"] == "single":
            val_acc = data["best_validation"]["accuracy"]
            best_epoch = data["best_epoch"]
        else:
            val_acc = data["validation"]["accuracy"]
            best_epoch = data.get("main_model", {}).get("best_epoch")

        note = pair_tag if pair_tag else category
        rows.append(
            RunResult(
                name=run_name,
                category=category,
                pair_tag=pair_tag,
                val_acc=val_acc,
                test_acc=data["test"]["accuracy"],
                macro_f1=data["test"]["report"]["macro avg"]["f1-score"],
                weighted_f1=data["test"]["report"]["weighted avg"]["f1-score"],
                best_epoch=best_epoch,
                note=note,
            )
        )
    return rows


def build_family_order() -> list[str]:
    return [
        "v2.0_pretrained_ce_b2",
        "v2.1_pretrained_ce_cosine_b2",
        "v2.2_pretrained_ce_ls_b2",
        "v2.3_pretrained_ce_ls_cosine_b2",
        "v2.4_pretrained_ce_ls_cosine_weightedce_b2",
        "v3.0_pretrained_focal_ls_cosine_b2",
        "v3.1_pretrained_focal_ls_cosine_mixup_b2",
        "v3.2_pretrained_focal_ls_cosine_cutmix_b2",
        "v3.3_pretrained_focal_ls_cosine_se_b2",
        "v3.4_pretrained_focal_ls_cosine_cbam_b2",
    ]


def base_name_from_variant(run_name: str) -> str:
    if run_name.startswith("cascade_pair_ad_lc_"):
        return run_name[len("cascade_pair_ad_lc_") :]
    if run_name.startswith("cascade_pair_ad_sq_"):
        return run_name[len("cascade_pair_ad_sq_") :]
    if run_name.startswith("cascade_pair_lc_sq_"):
        return run_name[len("cascade_pair_lc_sq_") :]
    if run_name.startswith("cascade_"):
        return run_name[len("cascade_") :]
    return run_name


def make_paragraph(text: str, *, bold: bool = False, size: int = 22, align: str | None = None) -> str:
    ppr = []
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    ppr.append('<w:spacing w:after="120"/>')
    rpr = []
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
    tbl = [
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
    for row_idx, row in enumerate(rows):
        tbl.append("<w:tr>")
        for col_idx, cell in enumerate(row):
            shade = '<w:shd w:fill="D9EAF7"/>' if row_idx == 0 else ""
            tbl.append(
                "<w:tc>"
                + f'<w:tcPr><w:tcW w:w="{widths[col_idx]}" w:type="dxa"/>{shade}</w:tcPr>'
                + make_paragraph(cell, bold=(row_idx == 0), size=20 if row_idx == 0 else 18)
                + "</w:tc>"
            )
        tbl.append("</w:tr>")
    tbl.append("</w:tbl>")
    return "".join(tbl)


def build_document_xml(results: list[RunResult]) -> str:
    by_name = {row.name: row for row in results}
    family_order = build_family_order()

    singles = [by_name[name] for name in family_order]
    tumor3 = [by_name["cascade_" + name] for name in family_order]
    pair_ad_lc = [by_name["cascade_pair_ad_lc_" + name] for name in family_order]
    pair_ad_sq = [by_name["cascade_pair_ad_sq_" + name] for name in family_order]
    pair_lc_sq = [by_name["cascade_pair_lc_sq_" + name] for name in family_order]
    pairs = pair_ad_lc + pair_ad_sq + pair_lc_sq

    overall_best = max(results, key=lambda x: (x.test_acc, x.macro_f1, x.val_acc))
    best_single = max(singles, key=lambda x: (x.test_acc, x.macro_f1, x.val_acc))
    best_tumor3 = max(tumor3, key=lambda x: (x.test_acc, x.macro_f1, x.val_acc))
    best_pair = max(pairs, key=lambda x: (x.test_acc, x.macro_f1, x.val_acc))

    tumor3_better = 0
    best_pair_better = 0
    family_rows = [["主线版本", "单模型", "三肿瘤级联", "最佳两两级联", "家族结论"]]
    for base in family_order:
        s = by_name[base]
        t = by_name["cascade_" + base]
        p_candidates = [
            by_name["cascade_pair_ad_lc_" + base],
            by_name["cascade_pair_ad_sq_" + base],
            by_name["cascade_pair_lc_sq_" + base],
        ]
        p_best = max(p_candidates, key=lambda x: (x.test_acc, x.macro_f1, x.val_acc))
        if t.test_acc > s.test_acc:
            tumor3_better += 1
        if p_best.test_acc > s.test_acc:
            best_pair_better += 1
        family_rows.append(
            [
                base,
                pct(s.test_acc),
                pct(t.test_acc),
                f"{p_best.pair_tag} / {pct(p_best.test_acc)}",
                f"最佳方案：{base_name_from_variant(p_best.name)} 系列中的 {p_best.category}",
            ]
        )

    top10_rows = [["排名", "实验名", "类别", "验证准确率", "测试准确率", "Macro F1"]]
    for idx, row in enumerate(sorted(results, key=lambda x: (-x.test_acc, -x.macro_f1, -x.val_acc, x.name))[:10], start=1):
        top10_rows.append([str(idx), row.name, row.category if not row.pair_tag else row.pair_tag, pct(row.val_acc), pct(row.test_acc), f1(row.macro_f1)])

    single_rows = [["实验名", "主要对照对象", "验证准确率", "测试准确率", "Macro F1", "Best Epoch"]]
    for row in singles:
        single_rows.append(
            [
                row.name,
                f"cascade_{row.name} + 3 组 pairwise",
                pct(row.val_acc),
                pct(row.test_acc),
                f1(row.macro_f1),
                str(row.best_epoch or ""),
            ]
        )

    tumor3_rows = [["实验名", "主要对照对象", "验证准确率", "测试准确率", "Macro F1", "Best Epoch"]]
    for row in tumor3:
        tumor3_rows.append(
            [
                row.name,
                f"{base_name_from_variant(row.name)} + 3 组 pairwise",
                pct(row.val_acc),
                pct(row.test_acc),
                f1(row.macro_f1),
                str(row.best_epoch or ""),
            ]
        )

    pair_rows = [["实验名", "专家对", "主要对照对象", "验证准确率", "测试准确率", "Macro F1"]]
    for row in sorted(pairs, key=lambda x: (base_name_from_variant(x.name), x.pair_tag or "")):
        pair_rows.append(
            [
                row.name,
                row.pair_tag or "",
                f"{base_name_from_variant(row.name)} + tumor3 + 其余 2 组 pairwise",
                pct(row.val_acc),
                pct(row.test_acc),
                f1(row.macro_f1),
            ]
        )

    avg_single_acc = mean(row.test_acc for row in singles)
    avg_single_macro = mean(row.macro_f1 for row in singles)
    avg_tumor3_acc = mean(row.test_acc for row in tumor3)
    avg_tumor3_macro = mean(row.macro_f1 for row in tumor3)
    avg_pair_acc = mean(row.test_acc for row in pairs)
    avg_pair_macro = mean(row.macro_f1 for row in pairs)
    avg_ad_lc = mean(row.test_acc for row in pair_ad_lc)
    avg_ad_sq = mean(row.test_acc for row in pair_ad_sq)
    avg_lc_sq = mean(row.test_acc for row in pair_lc_sq)

    body: list[str] = []
    body.append(make_paragraph("EfficientNet-B2 50组正式实验结果对比", bold=True, size=32, align="center"))
    body.append(make_paragraph("统计范围：10组单模型 + 10组三肿瘤专家级联 + 30组两两专家级联，共50组正式四分类结果。未计入40组 expert_* 子任务训练结果。", size=20))

    body.append(make_paragraph("一、可直接相互对照的实验关系", bold=True, size=26))
    body.append(make_paragraph("1. 同一主线版本内，可以直接对照 5 组结果：1组单模型、1组三肿瘤专家级联、3组两两专家级联。此时主干、损失、调度、增强和注意力配置相同，只有是否引入专家分支以及专家类别划分不同。", size=20))
    body.append(make_paragraph("2. 同一实验形态 across 主线版本也可以直接对照，例如 10 组单模型之间、10 组三肿瘤级联之间、或同一 pair tag 的 10 组级联之间。此时主要比较训练策略本身，例如 cosine、label smoothing、focal、MixUp、CutMix、SE、CBAM。", size=20))
    body.append(make_paragraph("3. 两两专家级联内部还可以按专家对再分成三条线：腺癌 vs 大细胞癌、腺癌 vs 鳞癌、大细胞癌 vs 鳞癌。它们可以横向比较哪一类二级细分最有效。", size=20))

    body.append(make_paragraph("二、总体最优结果", bold=True, size=26))
    body.append(make_paragraph(f"全局最优：{overall_best.name}，测试准确率 {pct(overall_best.test_acc)}，Macro F1 = {f1(overall_best.macro_f1)}。", size=20))
    body.append(make_paragraph(f"最佳单模型：{best_single.name}，测试准确率 {pct(best_single.test_acc)}，Macro F1 = {f1(best_single.macro_f1)}。", size=20))
    body.append(make_paragraph(f"最佳三肿瘤级联：{best_tumor3.name}，测试准确率 {pct(best_tumor3.test_acc)}，Macro F1 = {f1(best_tumor3.macro_f1)}。", size=20))
    body.append(make_paragraph(f"最佳两两专家级联：{best_pair.name}，测试准确率 {pct(best_pair.test_acc)}，Macro F1 = {f1(best_pair.macro_f1)}。", size=20))
    body.append(make_paragraph(f"分组平均表现：单模型 {pct(avg_single_acc)} / {f1(avg_single_macro)}，三肿瘤级联 {pct(avg_tumor3_acc)} / {f1(avg_tumor3_macro)}，两两专家级联 {pct(avg_pair_acc)} / {f1(avg_pair_macro)}。", size=20))

    body.append(make_paragraph("三、Top 10 总排名", bold=True, size=26))
    body.append(make_table(top10_rows, [700, 4300, 1800, 1300, 1300, 1100]))

    body.append(make_page_break())
    body.append(make_paragraph("四、10组单模型结果", bold=True, size=26))
    body.append(make_table(single_rows, [4200, 2600, 1200, 1200, 1100, 1000]))

    body.append(make_paragraph("五、10组三肿瘤专家级联结果", bold=True, size=26))
    body.append(make_table(tumor3_rows, [4200, 2600, 1200, 1200, 1100, 1000]))

    body.append(make_page_break())
    body.append(make_paragraph("六、30组两两专家级联结果", bold=True, size=26))
    body.append(make_table(pair_rows, [4200, 1700, 2500, 1200, 1200, 1100]))

    body.append(make_page_break())
    body.append(make_paragraph("七、按主线版本的家族对照", bold=True, size=26))
    body.append(make_table(family_rows, [3200, 1200, 1400, 2100, 3800]))

    body.append(make_paragraph("八、实验对比结论", bold=True, size=26))
    body.append(make_paragraph(f"1. 三肿瘤专家级联在 10 组家族中有 {tumor3_better}/10 组优于对应单模型，说明它整体有增益，但不是每一组都稳定获益。", size=20))
    body.append(make_paragraph(f"2. 每个主线家族的最佳两两专家级联都优于对应单模型，即 {best_pair_better}/10 组成立，说明 pairwise expert 的上限更高。", size=20))
    body.append(make_paragraph(f"3. 三类 pairwise expert 中，平均测试准确率最高的是“腺癌 vs 鳞癌”这条线，平均为 {pct(avg_ad_sq)}；其次是“大细胞癌 vs 鳞癌” {pct(avg_lc_sq)}；“腺癌 vs 大细胞癌”最低，为 {pct(avg_ad_lc)}。", size=20))
    body.append(make_paragraph("4. 本轮 B2 的最强主线集中在 v3.2 CutMix 和 v3.0 Focal+LS+Cosine 附近，其中 v3.2 CutMix 同时拿下最佳单模型、最佳三肿瘤级联和最佳总结果，说明在 B2 条件下 CutMix 对泛化最有效。", size=20))
    body.append(make_paragraph("5. v2.4 balanced weighted CE 与 v3.4 CBAM 在本轮 B2 中整体偏弱，说明标准 balanced weighted CE 以及当前这版 CBAM 组合并没有成为稳定增益项。", size=20))
    body.append(make_paragraph("6. 如果下一步只保留少量候选方案继续深挖，优先级应为：v3.2 主线、v3.0 主线，以及 pairwise ad_sq 级联路线。", size=20))

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
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "outputs" / "results"
    doc_root = repo_root / "doc"
    output_path = doc_root / DOC_NAME

    results = load_formal_results(results_root)
    formal_count = len(results)
    if formal_count != 50:
        raise SystemExit(f"Expected 50 formal B2 runs, found {formal_count}.")

    document_xml = build_document_xml(results)
    write_docx(output_path, document_xml)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
