from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = ROOT / "doc" / "ablation_results.md"
OUTPUT_DOCX = ROOT / "doc" / "ablation_results_中文翻译_20260719.docx"


HEADING_MAP = {
    "Ablation Results": "消融实验结果",
    "Setup": "实验设置",
    "Summary": "结果汇总",
    "Attention Analysis": "注意力机制分析",
    "Conclusions": "结论",
}

SETUP_TEXT_MAP = {
    "All runs use the same dataset split, `EfficientNet-B0`, `batch_size=16`, `image_size=224`, `lr=3e-4`, `weight_decay=1e-4`, and `seed=42`.": (
        "所有实验均使用相同的数据集划分、`EfficientNet-B0` 主干网络，以及一致的训练超参数："
        "`batch_size=16`、`image_size=224`、`lr=3e-4`、`weight_decay=1e-4`、`seed=42`。"
    ),
}

TABLE_TEXT_MAP = {
    "Run": "实验",
    "Key change": "关键改动",
    "Acc": "准确率",
    "Macro F1": "Macro F1",
    "Note": "说明",
    "scratch CE": "从零训练 + CE",
    "+ pretrained": "+ 预训练",
    "+ label smoothing + cosine": "+ 标签平滑 + 余弦调度",
    "+ balanced weights": "+ 平衡类别权重",
    "+ focal loss": "+ Focal Loss",
    "scratch CE on CUDA": "CUDA 上从零训练 + CE",
    "+ label smoothing only": "仅加标签平滑",
    "+ cosine only": "仅加余弦调度",
    "+ MixUp": "+ MixUp",
    "+ CutMix": "+ CutMix",
    "+ extra SE": "+ 额外 SE 模块",
    "+ CBAM": "+ CBAM",
    "legacy CPU baseline": "旧版 CPU baseline",
    "biggest single gain": "最大的单项提升",
    "historical best CE run": "历史最佳 CE 实验",
    "hurts overall": "整体表现下降",
    "historical best before ablations": "消融扩展前的历史最佳",
    "same idea as legacy baseline": "与旧版 baseline 同一路线",
    "best simple regularizer": "最佳轻量正则策略",
    "mild gain": "小幅提升",
    "too aggressive here": "在当前任务中过于激进",
    "also degrades": "同样带来退化",
    "weak improvement": "提升有限",
    "best overall": "综合最优",
}

ATTENTION_BULLET_MAP = {
    "`ablation_pretrained_focal_ls_cosine_se` reaches `77.14%` accuracy and `0.7884` macro F1. It does not beat the focal baseline, so the extra SE block looks mostly redundant on top of `EfficientNet-B0`.": (
        "`ablation_pretrained_focal_ls_cosine_se` 的准确率为 `77.14%`，`macro F1` 为 `0.7884`。"
        "它没有超过 focal baseline，因此在 `EfficientNet-B0` 之上再加一个 SE 模块大体上是冗余的。"
    ),
    "`ablation_pretrained_focal_ls_cosine_cbam` reaches `86.35%` accuracy and `0.8646` macro F1. Compared with SE, it improves accuracy by `9.21` points and macro F1 by `0.0762`.": (
        "`ablation_pretrained_focal_ls_cosine_cbam` 的准确率为 `86.35%`，`macro F1` 为 `0.8646`。"
        "与 SE 版本相比，准确率提升了 `9.21` 个百分点，`macro F1` 提升了 `0.0762`。"
    ),
    "Per-class tradeoff: CBAM greatly lifts `adenocarcinoma` recall, keeps `large.cell.carcinoma` and `normal` strong, but lowers `squamous.cell.carcinoma` recall.": (
        "分类别来看，CBAM 显著提升了 `adenocarcinoma` 的召回率，同时保持了 `large.cell.carcinoma`"
        " 和 `normal` 的较强表现，但会降低 `squamous.cell.carcinoma` 的召回率。"
    ),
}

CONCLUSION_BULLET_MAP = {
    "Pretraining is the largest improvement.": "预训练带来了最大的性能提升。",
    "Label smoothing is the strongest low-cost regularizer here.": "在当前实验中，标签平滑是效果最强的低成本正则化手段。",
    "`balanced` class weighting, MixUp, and CutMix are not helpful in this small-sample setting.": (
        "在当前小样本设定下，`balanced` 类别加权、MixUp 和 CutMix 都没有带来帮助。"
    ),
    "Extra SE is weaker than CBAM and likely redundant on top of `EfficientNet-B0`.": (
        "额外加入的 SE 模块弱于 CBAM，并且在 `EfficientNet-B0` 之上大概率是冗余的。"
    ),
    "Best run: `ablation_pretrained_focal_ls_cosine_cbam`.": (
        "当前最佳实验为：`ablation_pretrained_focal_ls_cosine_cbam`。"
    ),
}


def parse_markdown() -> dict:
    lines = SOURCE_MD.read_text(encoding="utf-8").splitlines()
    title = ""
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            current_section = line[3:].strip()
            sections[current_section] = []
            continue
        if current_section is not None:
            sections[current_section].append(line.rstrip())

    return {"title": title, "sections": sections}


def parse_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(cell.startswith("---") or cell.endswith("---:") or cell == "---:" for cell in cells):
            continue
        rows.append(cells)
    return rows


def translate_text(text: str, mapping: dict[str, str]) -> str:
    return mapping.get(text, text)


def paragraph(text: str, *, bold: bool = False, size: int = 22, center: bool = False) -> str:
    ppr = "<w:pPr><w:spacing w:after=\"120\"/>"
    if center:
        ppr += "<w:jc w:val=\"center\"/>"
    ppr += "</w:pPr>"
    rpr_parts = [
        "<w:rFonts w:ascii=\"Calibri\" w:hAnsi=\"Calibri\" w:eastAsia=\"宋体\"/>",
        f"<w:sz w:val=\"{size}\"/>",
        f"<w:szCs w:val=\"{size}\"/>",
    ]
    if bold:
        rpr_parts.append("<w:b/>")
    rpr = f"<w:rPr>{''.join(rpr_parts)}</w:rPr>"
    return (
        "<w:p>"
        f"{ppr}"
        f"<w:r>{rpr}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>"
        "</w:p>"
    )


def table(rows: list[list[str]]) -> str:
    grid_cols = "".join("<w:gridCol w:w=\"2200\"/>" for _ in rows[0])
    tbl_rows = []
    for row_index, row in enumerate(rows):
        cells = []
        for cell in row:
            cell_text = paragraph(cell, bold=row_index == 0, size=20)
            cells.append(
                "<w:tc>"
                "<w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/></w:tcPr>"
                f"{cell_text}"
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


def build_document_xml(parsed: dict) -> str:
    sections = parsed["sections"]
    summary_rows_raw = parse_table(sections["Summary"])
    summary_rows = [[translate_text(cell, TABLE_TEXT_MAP) for cell in row] for row in summary_rows_raw]

    setup_lines = [line for line in sections["Setup"] if line.strip()]
    attention_lines = [line[2:] for line in sections["Attention Analysis"] if line.startswith("- ")]
    conclusion_lines = [line[2:] for line in sections["Conclusions"] if line.startswith("- ")]

    parts = [
        paragraph(translate_text(parsed["title"], HEADING_MAP), bold=True, size=28, center=True),
        paragraph("来源：2026APMCM/doc/ablation_results.md", size=20, center=True),
        paragraph("生成时间：2026-07-19", size=20, center=True),
        paragraph("一、实验设置", bold=True, size=24),
    ]

    for line in setup_lines:
        parts.append(paragraph(translate_text(line, SETUP_TEXT_MAP), size=21))

    parts.extend(
        [
            paragraph("二、结果汇总", bold=True, size=24),
            table(summary_rows),
            paragraph("三、注意力机制分析", bold=True, size=24),
        ]
    )

    for line in attention_lines:
        parts.append(paragraph(f"• {translate_text(line, ATTENTION_BULLET_MAP)}", size=21))

    parts.append(paragraph("四、结论", bold=True, size=24))
    for line in conclusion_lines:
        parts.append(paragraph(f"• {translate_text(line, CONCLUSION_BULLET_MAP)}", size=21))

    body = "".join(parts) + (
        "<w:sectPr>"
        "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
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
    now = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
        "xmlns:dc=\"http://purl.org/dc/elements/1.1/\" "
        "xmlns:dcterms=\"http://purl.org/dc/terms/\" "
        "xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" "
        "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">"
        "<dc:title>消融实验结果中文翻译</dc:title>"
        "<dc:creator>Codex</dc:creator>"
        "<cp:lastModifiedBy>Codex</cp:lastModifiedBy>"
        f"<dcterms:created xsi:type=\"dcterms:W3CDTF\">{now}</dcterms:created>"
        f"<dcterms:modified xsi:type=\"dcterms:W3CDTF\">{now}</dcterms:modified>"
        "</cp:coreProperties>"
    )


def build_docx() -> None:
    parsed = parse_markdown()
    document_xml = build_document_xml(parsed)

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

    OUTPUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT_DOCX, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("docProps/core.xml", build_core_xml())
        zf.writestr("docProps/app.xml", app_xml)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/styles.xml", styles_xml)
        zf.writestr("word/_rels/document.xml.rels", document_rels)


def main() -> None:
    build_docx()
    print(OUTPUT_DOCX)


if __name__ == "__main__":
    main()
