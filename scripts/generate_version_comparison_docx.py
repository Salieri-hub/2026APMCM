from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DOC_DIR = ROOT / "doc"
DOC_PATH = DOC_DIR / "实验版本对比说明_20260718.docx"

CLASS_DISPLAY = {
    "adenocarcinoma": "腺癌",
    "large.cell.carcinoma": "大细胞癌",
    "normal": "正常",
    "squamous.cell.carcinoma": "鳞癌",
}

VERSIONS = [
    {
        "id": "V1",
        "name": "CPU baseline（随机初始化）",
        "dir": "problem2_baseline原来的",
        "principle": [
            "使用 EfficientNet-B0 作为主干网络，但不加载预训练权重。",
            "在 CPU 上训练，目标是先验证训练、验证、测试与结果导出链路是否完整可跑通。",
            "该版本不包含 label smoothing、学习率调度器或类别加权，因此更接近最朴素的基础线。",
        ],
        "positioning": "用于建立最原始可运行基线，验证工程链路，不追求最终效果。",
    },
    {
        "id": "V2",
        "name": "GPU + pretrained",
        "dir": "problem2_baseline_gpu",
        "principle": [
            "在 V1 基础上启用 timm 预训练权重，相当于引入迁移学习。",
            "迁移学习先利用大规模自然图像数据学到的底层纹理与形状特征，再在肺部 CT 数据上微调，通常能显著改善小样本医学图像任务的起点。",
            "同时切换到 GPU + AMP 混合精度，缩短训练时间并提高实验迭代速度。",
        ],
        "positioning": "这是第一版真正有效的提升版本，核心收益来自预训练与 GPU 训练。",
    },
    {
        "id": "V3",
        "name": "GPU + pretrained + label smoothing + cosine",
        "dir": "problem2_pretrained_ls_cosine",
        "principle": [
            "在 V2 的基础上加入 label smoothing=0.1，避免模型把训练标签学得过于绝对，降低过度自信预测。",
            "加入 cosine 学习率调度器，让学习率在训练后期更平滑地下降，减轻验证集大幅波动。",
            "这类组合通常用于缓解轻度过拟合，并提高模型在测试集上的泛化稳定性。",
        ],
        "positioning": "这是当前测试集表现最好的版本，属于现阶段最推荐的主线配置。",
    },
    {
        "id": "V4",
        "name": "V3 + balanced class-weighted CrossEntropy",
        "dir": "problem2_pretrained_ls_cosine_weightedce",
        "principle": [
            "在 V3 基础上，将 CrossEntropyLoss 改为按训练集类别频次自动计算的 balanced class weights。",
            "权重思路是让样本相对少、难学的类别在损失中占更大比重，从而抑制模型对高频或易学类别的偏置。",
            "这是文献中常见的类别不平衡处理方式，但它的效果高度依赖数据分布和具体混淆模式。",
        ],
        "positioning": "该版本完成了验证，但结果不如 V3，说明标准 balanced 权重并不是当前最优方向。",
    },
]


def load_metrics(version_dir: str) -> dict:
    with (OUTPUT_DIR / version_dir / "metrics_summary.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def fmt_float(value: float) -> str:
    return f"{value:.4f}"


def build_version_records() -> list[dict]:
    records: list[dict] = []
    for version in VERSIONS:
        metrics = load_metrics(version["dir"])
        test_report = metrics["test"]["report"]
        record = {
            **version,
            "metrics": metrics,
            "best_epoch": metrics["best_epoch"],
            "val_acc": metrics["best_validation"]["accuracy"],
            "test_acc": metrics["test"]["accuracy"],
            "macro_f1": test_report["macro avg"]["f1-score"],
            "weighted_f1": test_report["weighted avg"]["f1-score"],
            "elapsed_seconds": metrics["elapsed_seconds"],
            "recalls": {
                CLASS_DISPLAY[class_name]: test_report[class_name]["recall"]
                for class_name in metrics["dataset"]["class_names"]
            },
        }
        records.append(record)
    return records


def choose_best(records: list[dict]) -> dict:
    return max(records, key=lambda item: (item["test_acc"], item["macro_f1"]))


def paragraph(text: str, *, bold: bool = False, size: int = 22, center: bool = False) -> str:
    ppr = ""
    if center:
        ppr = "<w:pPr><w:jc w:val=\"center\"/></w:pPr>"
    rpr_parts = [f"<w:sz w:val=\"{size}\"/>", f"<w:szCs w:val=\"{size}\"/>"]
    if bold:
        rpr_parts.append("<w:b/>")
    rpr = f"<w:rPr>{''.join(rpr_parts)}</w:rPr>"
    return (
        "<w:p>"
        f"{ppr}"
        f"<w:r>{rpr}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>"
        "</w:p>"
    )


def blank_paragraph() -> str:
    return "<w:p/>"


def table(rows: list[list[str]]) -> str:
    grid_cols = "".join("<w:gridCol w:w=\"1800\"/>" for _ in rows[0])
    tbl_rows = []
    for row_index, row in enumerate(rows):
        cells = []
        for cell in row:
            cell_text = paragraph(cell, bold=row_index == 0, size=20)
            cells.append(f"<w:tc><w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/></w:tcPr>{cell_text}</w:tc>")
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


def build_document_xml(records: list[dict]) -> str:
    best = choose_best(records)
    body_parts = [
        paragraph("APMCM 2026 B题 实验版本对比说明", bold=True, size=28, center=True),
        paragraph("生成时间：2026-07-18", size=20, center=True),
        blank_paragraph(),
        paragraph("一、截至目前共做了哪几个版本", bold=True, size=24),
        paragraph(
            "截至 2026-07-18，已经完成 4 个可对比的正式版本：V1 CPU baseline、V2 GPU + pretrained、V3 GPU + pretrained + label smoothing + cosine、V4 V3 + balanced class-weighted CrossEntropy。",
            size=21,
        ),
        paragraph(
            f"其中当前测试集综合表现最好的版本是 {best['id']}（{best['name']}），测试准确率 {fmt_pct(best['test_acc'])}，测试 Macro F1 为 {fmt_float(best['macro_f1'])}。",
            size=21,
        ),
        blank_paragraph(),
        paragraph("二、各版本原理说明", bold=True, size=24),
    ]

    for record in records:
        body_parts.append(paragraph(f"{record['id']}：{record['name']}", bold=True, size=22))
        body_parts.append(paragraph(f"定位：{record['positioning']}", size=21))
        for item in record["principle"]:
            body_parts.append(paragraph(f"• {item}", size=21))
        body_parts.append(
            paragraph(
                "核心配置："
                f"pretrained={record['metrics']['config'].get('pretrained', False)}, "
                f"device={record['metrics']['config'].get('device', 'cpu')}, "
                f"label_smoothing={record['metrics']['config'].get('label_smoothing', 0.0)}, "
                f"scheduler={record['metrics']['config'].get('scheduler', 'none')}, "
                f"class_weighting={record['metrics']['config'].get('class_weighting', 'none')}",
                size=21,
            )
        )
        body_parts.append(blank_paragraph())

    body_parts.extend(
        [
            paragraph("三、结果对比表", bold=True, size=24),
            table(
                [
                    ["版本", "最佳验证准确率", "测试准确率", "测试Macro F1", "测试Weighted F1", "训练耗时"],
                    *[
                        [
                            f"{record['id']} {record['name']}",
                            fmt_pct(record["val_acc"]),
                            fmt_pct(record["test_acc"]),
                            fmt_float(record["macro_f1"]),
                            fmt_float(record["weighted_f1"]),
                            f"{record['elapsed_seconds']:.1f}s",
                        ]
                        for record in records
                    ],
                ]
            ),
            blank_paragraph(),
            paragraph("四、各类别召回率对比", bold=True, size=24),
            table(
                [
                    ["版本", "腺癌召回率", "大细胞癌召回率", "正常召回率", "鳞癌召回率"],
                    *[
                        [
                            record["id"],
                            fmt_pct(record["recalls"]["腺癌"]),
                            fmt_pct(record["recalls"]["大细胞癌"]),
                            fmt_pct(record["recalls"]["正常"]),
                            fmt_pct(record["recalls"]["鳞癌"]),
                        ]
                        for record in records
                    ],
                ]
            ),
            blank_paragraph(),
            paragraph("五、对比结论", bold=True, size=24),
            paragraph("1. V1 仅用于证明工程链路可运行，测试集准确率只有 39.68%，主要问题是模型大量把样本预测成大细胞癌。", size=21),
            paragraph("2. V2 通过迁移学习把测试集准确率显著提升到 76.19%，说明预训练是当前项目中最关键的一步提升。", size=21),
            paragraph("3. V3 在 V2 基础上加入 label smoothing 和 cosine 调度后，测试集准确率提升到 77.46%，Macro F1 提升到 0.7894，是当前最优版本。", size=21),
            paragraph("4. V4 进一步加入标准 balanced class-weighted CrossEntropy 后，测试集准确率反而下降到 73.33%，说明标准类别权重在当前数据上不是最优方向。", size=21),
            paragraph("5. 当前最值得继续推进的路线不是继续使用 balanced 权重，而是以 V3 为主线，优先尝试 Focal Loss、采样策略或手动类别权重。", size=21),
            blank_paragraph(),
            paragraph("六、建议的下一步", bold=True, size=24),
            paragraph("• 保留 V3 作为当前主线配置。", size=21),
            paragraph("• 下一轮优先尝试 Focal Loss 或采样策略，目标是在不损失鳞癌召回率的前提下，拉回腺癌召回率。", size=21),
            paragraph("• 如果继续尝试类别加权，建议改为手动权重，而不是直接使用标准 balanced 权重。", size=21),
        ]
    )

    body = "".join(body_parts) + (
        "<w:sectPr>"
        "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
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
    now = datetime(2026, 7, 18, 17, 50, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
        "xmlns:dc=\"http://purl.org/dc/elements/1.1/\" "
        "xmlns:dcterms=\"http://purl.org/dc/terms/\" "
        "xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" "
        "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">"
        "<dc:title>实验版本对比说明</dc:title>"
        "<dc:creator>Codex</dc:creator>"
        "<cp:lastModifiedBy>Codex</cp:lastModifiedBy>"
        f"<dcterms:created xsi:type=\"dcterms:W3CDTF\">{now}</dcterms:created>"
        f"<dcterms:modified xsi:type=\"dcterms:W3CDTF\">{now}</dcterms:modified>"
        "</cp:coreProperties>"
    )


def build_docx(records: list[dict]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    document_xml = build_document_xml(records)
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
    records = build_version_records()
    build_docx(records)
    print(DOC_PATH)


if __name__ == "__main__":
    main()
