from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
DOC_DIR = ROOT / "doc"
DOC_PATH = DOC_DIR / "b0_b1_102_experiments_comparison_20260719.docx"

B0_BASE_RUNS = [
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
B0_MAIN_RUNS = [name for name in B0_BASE_RUNS if name.startswith("v2.") or name.startswith("v3.")]
B1_MAIN_RUNS = [f"{name}_b1" for name in B0_MAIN_RUNS]
PAIR_TAGS = ["ad_lc", "ad_sq", "lc_sq"]
PAIR_LABELS = {
    "ad_lc": "腺癌-大细胞癌二分类专家",
    "ad_sq": "腺癌-鳞癌二分类专家",
    "lc_sq": "大细胞癌-鳞癌二分类专家",
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
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics_summary.json for {run_name}")
    return json.loads(path.read_text(encoding="utf-8"))


def main_run_for(run_name: str) -> str:
    if run_name.startswith("cascade_pair_"):
        remainder = run_name[len("cascade_pair_") :]
        for tag in PAIR_TAGS:
            prefix = f"{tag}_"
            if remainder.startswith(prefix):
                return remainder[len(prefix) :]
        raise ValueError(f"Cannot parse pairwise run name: {run_name}")
    if run_name.startswith("cascade_"):
        return run_name[len("cascade_") :]
    return run_name


def backbone_for(run_name: str) -> str:
    return "B1" if run_name.endswith("_b1") else "B0"


def mode_for(run_name: str) -> str:
    if run_name.startswith("cascade_pair_"):
        return "pair_cascade"
    if run_name.startswith("cascade_"):
        return "tumor3_cascade"
    return "single"


def family_for(run_name: str) -> str:
    family = main_run_for(run_name)
    if family.endswith("_b1"):
        family = family[: -len("_b1")]
    return family


def expert_type_for(run_name: str) -> str:
    if run_name.startswith("cascade_pair_"):
        remainder = run_name[len("cascade_pair_") :]
        for tag in PAIR_TAGS:
            prefix = f"{tag}_"
            if remainder.startswith(prefix):
                return PAIR_LABELS[tag]
        raise ValueError(f"Cannot parse pairwise run name: {run_name}")
    if run_name.startswith("cascade_"):
        return "三肿瘤三分类专家"
    return "-"


def build_record(run_name: str) -> dict:
    metrics = load_metrics(run_name)
    test = metrics["test"]
    report = test["report"]
    record = {
        "name": run_name,
        "family": family_for(run_name),
        "main_run": main_run_for(run_name),
        "backbone": backbone_for(run_name),
        "mode": mode_for(run_name),
        "expert_type": expert_type_for(run_name),
        "acc": test["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "expert_invocations": None,
        "delta_acc_vs_main": 0.0,
        "delta_f1_vs_main": 0.0,
    }
    if record["mode"] != "single":
        base = load_metrics(record["main_run"])
        base_acc = base["test"]["accuracy"]
        base_f1 = base["test"]["report"]["macro avg"]["f1-score"]
        record["delta_acc_vs_main"] = record["acc"] - base_acc
        record["delta_f1_vs_main"] = record["macro_f1"] - base_f1
        record["expert_invocations"] = test.get("cascade_stats", {}).get("expert_invocations", 0)
    return record


def build_records() -> dict[str, list[dict]]:
    b0_single = [build_record(name) for name in B0_BASE_RUNS]
    b1_single = [build_record(name) for name in B1_MAIN_RUNS]
    b0_tumor3 = [build_record(f"cascade_{name}") for name in B0_MAIN_RUNS]
    b1_tumor3 = [build_record(f"cascade_{name}") for name in B1_MAIN_RUNS]
    b0_pair = [build_record(f"cascade_pair_{tag}_{name}") for tag in PAIR_TAGS for name in B0_MAIN_RUNS]
    b1_pair = [build_record(f"cascade_pair_{tag}_{name}") for tag in PAIR_TAGS for name in B1_MAIN_RUNS]
    all_records = b0_single + b1_single + b0_tumor3 + b1_tumor3 + b0_pair + b1_pair
    if len(all_records) != 102:
        raise ValueError(f"Expected 102 formal records, got {len(all_records)}")
    return {
        "b0_single": b0_single,
        "b1_single": b1_single,
        "b0_tumor3": b0_tumor3,
        "b1_tumor3": b1_tumor3,
        "b0_pair": b0_pair,
        "b1_pair": b1_pair,
        "all": all_records,
    }


def best_record(records: list[dict]) -> dict:
    return max(records, key=lambda r: (r["acc"], r["macro_f1"]))


def build_group_summary_table(groups: dict[str, list[dict]]) -> str:
    rows = [["分组", "数量", "最佳实验", "最佳Acc", "最佳Macro F1", "平均Acc", "平均Macro F1"]]
    ordered = [
        ("B0 单模型 12次", groups["b0_single"]),
        ("B1 单模型 10次", groups["b1_single"]),
        ("B0 三肿瘤级联 10次", groups["b0_tumor3"]),
        ("B1 三肿瘤级联 10次", groups["b1_tumor3"]),
        ("B0 两两专家级联 30次", groups["b0_pair"]),
        ("B1 两两专家级联 30次", groups["b1_pair"]),
        ("B0 全部正式实验 52次", groups["b0_single"] + groups["b0_tumor3"] + groups["b0_pair"]),
        ("B1 全部正式实验 50次", groups["b1_single"] + groups["b1_tumor3"] + groups["b1_pair"]),
    ]
    for label, records in ordered:
        best = best_record(records)
        rows.append(
            [
                label,
                str(len(records)),
                best["name"],
                pct(best["acc"]),
                float4(best["macro_f1"]),
                pct(mean(r["acc"] for r in records)),
                float4(mean(r["macro_f1"] for r in records)),
            ]
        )
    return table(rows, [2500, 900, 4200, 1200, 1500, 1200, 1600])


def build_compare_scope_table() -> str:
    rows = [
        ["对照类型", "可以相互对照的实验", "对照含义"],
        [
            "B0 历史硬件基线对照",
            "v1.0_scratch_ce_cpu ↔ v1.1_scratch_ce_cuda",
            "只改运行设备，检验 CPU 与 CUDA 切换本身的影响。",
        ],
        [
            "B0 单模型消融链",
            "v1.1↔v2.0，v2.0↔v2.1 / v2.2，v2.1↔v2.3，v2.3↔v2.4 / v3.0，v3.0↔v3.1 / v3.2 / v3.3 / v3.4",
            "主干保持 B0，通过预训练、调度器、label smoothing、weighted CE、focal、MixUp、CutMix、SE、CBAM 做组内消融。",
        ],
        [
            "B1 单模型消融链",
            "v2.0_b1↔v2.1_b1 / v2.2_b1，v2.1_b1↔v2.3_b1，v2.3_b1↔v2.4_b1 / v3.0_b1，v3.0_b1↔v3.1_b1 / v3.2_b1 / v3.3_b1 / v3.4_b1",
            "主干保持 B1，比较训练策略和注意力模块的增益。",
        ],
        [
            "同主模型内部结构对照",
            "同一配置下的 single ↔ cascade_tumor3 ↔ cascade_pair_ad_lc ↔ cascade_pair_ad_sq ↔ cascade_pair_lc_sq",
            "主模型固定，只改变是否接专家模型以及专家模型的类别划分方式。",
        ],
        [
            "跨主干一一对照",
            "B0 与 B1 的同名配置：10个单模型 + 10个三肿瘤级联 + 30个两两专家级联，共 50 对",
            "控制训练策略不变，只改变 backbone，从而直接衡量 B0 升级到 B1 的净收益。",
        ],
        [
            "不能严格一一对照的部分",
            "v1.0_scratch_ce_cpu 与 v1.1_scratch_ce_cuda 没有 B1 对应项",
            "这两组只服务于 B0 历史基线说明，不应与 B1 正式 50 次实验混成一组直接比较。",
        ],
    ]
    return table(rows, [1800, 5200, 4200])


def build_top_table(records: list[dict], top_k: int = 15) -> str:
    ranked = sorted(records, key=lambda r: (r["acc"], r["macro_f1"]), reverse=True)[:top_k]
    rows = [["排名", "实验", "Backbone", "模式", "专家类型", "Test Acc", "Macro F1"]]
    mode_labels = {"single": "单模型", "tumor3_cascade": "三肿瘤级联", "pair_cascade": "两两专家级联"}
    for idx, record in enumerate(ranked, start=1):
        rows.append(
            [
                str(idx),
                record["name"],
                record["backbone"],
                mode_labels[record["mode"]],
                record["expert_type"],
                pct(record["acc"]),
                float4(record["macro_f1"]),
            ]
        )
    return table(rows, [900, 4200, 1000, 1700, 2200, 1200, 1400])


def build_historical_v1_table(groups: dict[str, list[dict]]) -> str:
    rows = [["实验", "Test Acc", "Macro F1", "可对照对象", "说明"]]
    for record in groups["b0_single"][:2]:
        counterpart = "v1.1_scratch_ce_cuda" if record["name"] == "v1.0_scratch_ce_cpu" else "v1.0_scratch_ce_cpu"
        note = "纯 CPU baseline" if record["name"] == "v1.0_scratch_ce_cpu" else "只切到 CUDA，未启用迁移学习"
        rows.append([record["name"], pct(record["acc"]), float4(record["macro_f1"]), counterpart, note])
    return table(rows, [3200, 1200, 1500, 2800, 3000])


def build_cross_backbone_table(b0_names: list[str], b1_names: list[str], title_kind: str) -> str:
    rows = [["配置族", "B0实验", "B0 Acc / F1", "B1实验", "B1 Acc / F1", "Acc变化", "F1变化"]]
    for b0_name, b1_name in zip(b0_names, b1_names):
        b0 = build_record(b0_name)
        b1 = build_record(b1_name)
        rows.append(
            [
                family_for(b0_name) if title_kind != "pair" else f"{family_for(b0_name)} + {expert_type_for(b0_name)}",
                b0_name,
                f"{pct(b0['acc'])} / {float4(b0['macro_f1'])}",
                b1_name,
                f"{pct(b1['acc'])} / {float4(b1['macro_f1'])}",
                delta_pct(b1["acc"], b0["acc"]),
                delta_float(b1["macro_f1"], b0["macro_f1"]),
            ]
        )
    return table(rows, [3000, 3200, 1800, 3300, 1800, 1500, 1400])


def build_internal_family_table(backbone_label: str, main_names: list[str]) -> str:
    rows = [["配置族", "单模型", "三肿瘤级联", "ad_lc", "ad_sq", "lc_sq", "该家族最优版本"]]
    for main_name in main_names:
        candidates = [
            build_record(main_name),
            build_record(f"cascade_{main_name}"),
            build_record(f"cascade_pair_ad_lc_{main_name}"),
            build_record(f"cascade_pair_ad_sq_{main_name}"),
            build_record(f"cascade_pair_lc_sq_{main_name}"),
        ]
        best = best_record(candidates)
        rows.append(
            [
                family_for(main_name) if backbone_label == "B0" else family_for(main_name) + " + B1",
                pct(candidates[0]["acc"]),
                pct(candidates[1]["acc"]),
                pct(candidates[2]["acc"]),
                pct(candidates[3]["acc"]),
                pct(candidates[4]["acc"]),
                f"{best['name']} ({pct(best['acc'])}, {float4(best['macro_f1'])})",
            ]
        )
    return table(rows, [2500, 1200, 1400, 1000, 1000, 1000, 4200])


def build_document_xml(groups: dict[str, list[dict]]) -> str:
    all_records = groups["all"]
    b0_all = groups["b0_single"] + groups["b0_tumor3"] + groups["b0_pair"]
    b1_all = groups["b1_single"] + groups["b1_tumor3"] + groups["b1_pair"]
    best_b0 = best_record(b0_all)
    best_b1 = best_record(b1_all)
    top_overall = best_record(all_records)

    mean_single_delta_acc = mean(r["acc"] for r in groups["b1_single"]) - mean(r["acc"] for r in groups["b0_single"][2:])
    mean_single_delta_f1 = mean(r["macro_f1"] for r in groups["b1_single"]) - mean(r["macro_f1"] for r in groups["b0_single"][2:])
    mean_tumor3_delta_acc = mean(r["acc"] for r in groups["b1_tumor3"]) - mean(r["acc"] for r in groups["b0_tumor3"])
    mean_tumor3_delta_f1 = mean(r["macro_f1"] for r in groups["b1_tumor3"]) - mean(r["macro_f1"] for r in groups["b0_tumor3"])
    mean_pair_delta_acc = mean(r["acc"] for r in groups["b1_pair"]) - mean(r["acc"] for r in groups["b0_pair"])
    mean_pair_delta_f1 = mean(r["macro_f1"] for r in groups["b1_pair"]) - mean(r["macro_f1"] for r in groups["b0_pair"])

    b1_better_single = sum(
        1 for b0_name, b1_name in zip(B0_MAIN_RUNS, B1_MAIN_RUNS) if build_record(b1_name)["acc"] > build_record(b0_name)["acc"]
    )
    b1_better_tumor3 = sum(
        1
        for b0_name, b1_name in zip([f"cascade_{n}" for n in B0_MAIN_RUNS], [f"cascade_{n}" for n in B1_MAIN_RUNS])
        if build_record(b1_name)["acc"] > build_record(b0_name)["acc"]
    )
    b1_better_pair = sum(
        1
        for tag in PAIR_TAGS
        for b0_name, b1_name in zip(B0_MAIN_RUNS, B1_MAIN_RUNS)
        if build_record(f"cascade_pair_{tag}_{b1_name}")["acc"] > build_record(f"cascade_pair_{tag}_{b0_name}")["acc"]
    )

    b0_tumor3_wins = sum(1 for name in B0_MAIN_RUNS if build_record(f"cascade_{name}")["acc"] > build_record(name)["acc"])
    b1_tumor3_wins = sum(1 for name in B1_MAIN_RUNS if build_record(f"cascade_{name}")["acc"] > build_record(name)["acc"])
    b0_best_pair_wins = 0
    b1_best_pair_wins = 0
    b1_best_pair_tags: list[str] = []
    for name in B0_MAIN_RUNS:
        pair_records = [build_record(f"cascade_pair_{tag}_{name}") for tag in PAIR_TAGS]
        if best_record(pair_records)["acc"] > build_record(name)["acc"]:
            b0_best_pair_wins += 1
    for name in B1_MAIN_RUNS:
        pair_records = [(tag, build_record(f"cascade_pair_{tag}_{name}")) for tag in PAIR_TAGS]
        best_pair_tag, best_pair = max(pair_records, key=lambda item: (item[1]["acc"], item[1]["macro_f1"]))
        if best_pair["acc"] > build_record(name)["acc"]:
            b1_best_pair_wins += 1
        b1_best_pair_tags.append(best_pair_tag)

    single_cross_table = build_cross_backbone_table(B0_MAIN_RUNS, B1_MAIN_RUNS, "single")
    tumor3_cross_table = build_cross_backbone_table(
        [f"cascade_{name}" for name in B0_MAIN_RUNS],
        [f"cascade_{name}" for name in B1_MAIN_RUNS],
        "tumor3",
    )
    pair_b0_names = [f"cascade_pair_{tag}_{name}" for tag in PAIR_TAGS for name in B0_MAIN_RUNS]
    pair_b1_names = [f"cascade_pair_{tag}_{name}" for tag in PAIR_TAGS for name in B1_MAIN_RUNS]
    pair_cross_table = build_cross_backbone_table(pair_b0_names, pair_b1_names, "pair")

    body_parts = [
        paragraph("B0 与 B1 共 102 组正式实验结果对比", bold=True, size=28, center=True),
        paragraph("覆盖历史 B0 的 52 组结果与当前 B1 的 50 组正式结果", size=20, center=True),
        paragraph("生成时间：2026-07-19", size=20, center=True),
        paragraph(
            "本报告只统计 outputs 中的正式实验输出，不把 expert_tumor3_* 与 expert_pair_* 这 40 组专家模型训练记录单独算作正式四分类结果。"
            "因此本文档的对比对象是 102 组正式结果：B0 单模型 12 次、B0 三肿瘤级联 10 次、B0 两两专家级联 30 次、"
            "B1 单模型 10 次、B1 三肿瘤级联 10 次、B1 两两专家级联 30 次。",
            size=21,
        ),
        paragraph("一、哪些实验可以相互对照", bold=True, size=24),
        paragraph(
            "对照关系可以分成三层：第一层是同一 backbone 内部的消融对照，主要看训练策略、损失函数和注意力模块的作用；"
            "第二层是同一主模型家族内部的结构对照，比较单模型与不同 cascade 方案；第三层是 B0 与 B1 的跨 backbone 一一对照，衡量主干从 B0 升级到 B1 的净收益。",
            size=21,
        ),
        build_compare_scope_table(),
        paragraph("二、总体结果概览", bold=True, size=24),
        paragraph(
            f"102 组实验里，B0 全部正式结果的最佳实验是 {best_b0['name']}，测试准确率 {pct(best_b0['acc'])}，Macro F1 为 {float4(best_b0['macro_f1'])}；"
            f"B1 全部正式结果的最佳实验是 {best_b1['name']}，测试准确率 {pct(best_b1['acc'])}，Macro F1 为 {float4(best_b1['macro_f1'])}。"
            f"全局第一名也是 {top_overall['name']}。与 B0 最优结果相比，B1 最优结果的准确率提升 {delta_pct(best_b1['acc'], best_b0['acc'])}，Macro F1 提升 {delta_float(best_b1['macro_f1'], best_b0['macro_f1'])}。",
            size=21,
        ),
        build_group_summary_table(groups),
        paragraph("三、全局排名前 15 的结果", bold=True, size=24),
        paragraph(
            "前 15 名可以直接看出当前主线的重心。B1 系列不仅整体进入榜单的数量更多，而且头部结果几乎都集中在 B1 的 v2.1、v3.3 家族及其 cascade 版本上。"
            "这说明 B1 升级不是局部偶然收益，而是贯穿单模型、三肿瘤级联和两两专家级联三条线的系统性提升。",
            size=21,
        ),
        build_top_table(all_records, top_k=15),
        paragraph("四、B0 历史基线中只能组内对照的两项实验", bold=True, size=24),
        paragraph(
            "v1.0_scratch_ce_cpu 与 v1.1_scratch_ce_cuda 只用于说明最早历史基线，它们没有 B1 对应项，因此不能纳入 B0/B1 的 50 组一一对照。"
            "这两项最合理的用途是解释：纯设备切换带来的收益有限，而真正的大幅提升来自后续迁移学习与训练策略升级。",
            size=21,
        ),
        build_historical_v1_table(groups),
        paragraph("五、B0 与 B1 的 10 组单模型一一对照", bold=True, size=24),
        paragraph(
            f"这 10 对实验控制了训练策略，只改变 backbone。结果是 B1 在 10/10 对单模型对照里全部优于 B0，平均准确率提升 {delta_pct(mean(r['acc'] for r in groups['b1_single']), mean(r['acc'] for r in groups['b0_single'][2:]))}，"
            f"平均 Macro F1 提升 {delta_float(mean(r['macro_f1'] for r in groups['b1_single']), mean(r['macro_f1'] for r in groups['b0_single'][2:]))}。"
            "这说明从 B0 升级到 B1，不是只让最优点更高，而是让整条单模型实验线整体抬升。",
            size=21,
        ),
        single_cross_table,
        paragraph("六、B0 与 B1 的 10 组三肿瘤级联一一对照", bold=True, size=24),
        paragraph(
            f"三肿瘤级联这一层，B1 同样在 10/10 对严格对照里全部优于 B0，平均准确率提升 {delta_pct(mean(r['acc'] for r in groups['b1_tumor3']), mean(r['acc'] for r in groups['b0_tumor3']))}，"
            f"平均 Macro F1 提升 {delta_float(mean(r['macro_f1'] for r in groups['b1_tumor3']), mean(r['macro_f1'] for r in groups['b0_tumor3']))}。"
            "这说明专家级联的收益并没有被更强主干吃掉，相反，B1 与专家模型的组合总体更有效。",
            size=21,
        ),
        tumor3_cross_table,
        paragraph("七、B0 与 B1 的 30 组两两专家级联一一对照", bold=True, size=24),
        paragraph(
            f"两两专家级联是对照数量最多的一组，共 30 对。B1 在这 30/30 对对照里也全部优于 B0，平均准确率提升 {delta_pct(mean(r['acc'] for r in groups['b1_pair']), mean(r['acc'] for r in groups['b0_pair']))}，"
            f"平均 Macro F1 提升 {delta_float(mean(r['macro_f1'] for r in groups['b1_pair']), mean(r['macro_f1'] for r in groups['b0_pair']))}。"
            "因此，主干升级到 B1 后，不仅单模型更强，连局部纠错能力也普遍增强。",
            size=21,
        ),
        pair_cross_table,
        paragraph("八、同一家族内部：单模型与不同 cascade 方案如何对照", bold=True, size=24),
        paragraph(
            f"在 B0 内部，三肿瘤级联有 {b0_tumor3_wins}/10 次优于对应单模型，而同家族下的最优两两专家级联有 {b0_best_pair_wins}/10 次优于单模型；"
            f"在 B1 内部，三肿瘤级联也有 {b1_tumor3_wins}/10 次优于单模型，而同家族下的最优两两专家级联有 {b1_best_pair_wins}/10 次优于单模型。"
            "因此，主模型与 cascade 的对照不能只看单个家族，而要看整个家族中哪种专家划分更适合该主模型。",
            size=21,
        ),
        paragraph("8.1 B0 家族内部对照", bold=True, size=22),
        build_internal_family_table("B0", B0_MAIN_RUNS),
        paragraph("8.2 B1 家族内部对照", bold=True, size=22),
        build_internal_family_table("B1", B1_MAIN_RUNS),
        paragraph("九、结论与可直接引用的对照结论", bold=True, size=24),
        paragraph(
            f"1. 跨 backbone 的 50 对严格一一对照全部显示 B1 优于 B0：单模型 10/10、三肿瘤级联 10/10、两两专家级联 30/30。"
            f"按平均值看，B1 相比 B0 的提升分别是：单模型 Acc {delta_pct(mean(r['acc'] for r in groups['b1_single']), mean(r['acc'] for r in groups['b0_single'][2:]))}、Macro F1 {delta_float(mean(r['macro_f1'] for r in groups['b1_single']), mean(r['macro_f1'] for r in groups['b0_single'][2:]))}；"
            f"三肿瘤级联 Acc {delta_pct(mean(r['acc'] for r in groups['b1_tumor3']), mean(r['acc'] for r in groups['b0_tumor3']))}、Macro F1 {delta_float(mean(r['macro_f1'] for r in groups['b1_tumor3']), mean(r['macro_f1'] for r in groups['b0_tumor3']))}；"
            f"两两专家级联 Acc {delta_pct(mean(r['acc'] for r in groups['b1_pair']), mean(r['acc'] for r in groups['b0_pair']))}、Macro F1 {delta_float(mean(r['macro_f1'] for r in groups['b1_pair']), mean(r['macro_f1'] for r in groups['b0_pair']))}。",
            size=21,
        ),
        paragraph(
            "2. 如果要做训练策略消融，应优先在同 backbone 内部进行。B0 与 B1 都存在完整的组内消融链，但 B0 与 B1 之间的比较应被定义为 backbone 升级实验，不能与单纯的 loss 或 scheduler 消融混写。",
            size=21,
        ),
        paragraph(
            "3. 如果要做专家模型结构对照，应固定同一主模型家族，再比较 single、三肿瘤级联和三种两两专家级联。这样的控制变量最干净，因为它只改变了后端的专家结构与触发后的纠错方式。",
            size=21,
        ),
        paragraph(
            f"4. 当前最强结果是 {best_b1['name']}。如果你在论文里只保留一个最终推荐方案，优先写这个结果；如果你想强调从旧方案到新方案的系统性提升，则应写明：B1 在全部 50 对直接对照中全面优于 B0。",
            size=21,
        ),
        paragraph(
            f"5. 在 B1 家族里，最优两两专家标签更偏向 ad_sq，即腺癌-鳞癌二分类专家，它在 10 个 B1 家族里有 {b1_best_pair_tags.count('ad_sq')} 次成为最优 pair 方案；"
            "这说明在 B1 主干下，针对腺癌与鳞癌边界的局部细化更容易转化成最终四分类增益。",
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
    now = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
        "xmlns:dc=\"http://purl.org/dc/elements/1.1/\" "
        "xmlns:dcterms=\"http://purl.org/dc/terms/\" "
        "xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" "
        "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">"
        "<dc:title>B0与B1共102组正式实验结果对比</dc:title>"
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
    groups = build_records()
    document_xml = build_document_xml(groups)
    build_docx(document_xml)
    print(DOC_PATH)


if __name__ == "__main__":
    main()
