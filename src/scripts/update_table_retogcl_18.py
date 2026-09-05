#!/usr/bin/env python3
"""把18数据集上的选定 SOTA、ReToGCL 与 CITA-GCL 写入 table.md。"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = PROJECT_ROOT / "table.md"
START_MARKER = "<!-- RETOGCL_18_START -->"
END_MARKER = "<!-- RETOGCL_18_END -->"
DATASETS = (
    "NCI1", "PROTEINS", "COLLAB", "MUTAG", "COLORS-3", "PTC_MR",
    "Mutagenicity", "ClinTox", "BACE", "BBBP", "ABIDE", "ADHD200",
    "SIDER", "HIV", "AIDS", "Tox21", "BZR", "COX2",
)
METRICS = (
    ("accuracy_percent", "Accuracy"),
    ("nmi_percent", "NMI"),
    ("ari_percent", "ARI"),
    ("macro_f1_percent", "Macro-F1"),
)
SOTAS = (
    ("BalanceGCL", "sota", "balancegcl"),
    ("Khan-GCL", "sota", "khangcl"),
    ("CellCLAT", "paper_sota", "cellclat"),
    ("UniImb", "sota", "uniimb"),
    ("DualPrism", "paper_sota", "dualprism"),
    ("DEL", "paper_sota", "del"),
    ("SpectRe", "paper_sota_2025_2026", "spectre"),
    ("TopER", "paper_sota_2025_2026", "toper"),
    ("LEAP", "paper_sota_2025_2026", "leap"),
    ("Hourglass", "paper_sota_2025_2026", "hourglass"),
    ("NodeID", "paper_sota_2025_2026", "nodeid"),
    ("GNN+", "paper_sota_2025_2026", "gnnplus"),
    ("RS-Pool", "paper_sota_2025_2026", "rspool"),
)


def experiment_root(dataset: str) -> Path:
    if dataset in {"NCI1", "PROTEINS", "COLLAB", "MUTAG", "COLORS-3"}:
        return PROJECT_ROOT / "outputs"
    if dataset in {"PTC_MR", "Mutagenicity"}:
        return PROJECT_ROOT / "outputs/three_datasets_20260812"
    if dataset in {"ClinTox", "BACE", "BBBP"}:
        return PROJECT_ROOT / "outputs/medical_benchmark_20260828"
    return PROJECT_ROOT / "outputs/medical_benchmark_extended_20260828"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    for key, _ in METRICS:
        if not math.isfinite(float(value[key]["mean"])):
            raise ValueError(f"发现非有限指标：{path}/{key}")
    return value


def original_path(dataset: str) -> Path:
    filename = f"D7-I10__D7-I01__D10-I04__{dataset}.json"
    if dataset in {
        "NCI1", "PROTEINS", "COLLAB", "MUTAG", "COLORS-3", "PTC_MR",
        "Mutagenicity",
    }:
        return PROJECT_ROOT / "IDEA_TRIPLE_MODULAR_TOP5/jobs" / filename
    return experiment_root(dataset) / "our_model/jobs" / filename


def pl069_path(dataset: str) -> Path:
    original = (
        PROJECT_ROOT / "outputs/retogcl_t077_adaptive_pipeline_100x6/jobs"
        / f"PL069__{dataset}.json"
    )
    if original.exists():
        return original
    return (
        PROJECT_ROOT / "outputs/retogcl_pl069_remaining_12x5fold/jobs"
        / f"PL069__{dataset}.json"
    )


def model_results(dataset: str) -> list[tuple[str, dict]]:
    values = [
        ("ReToGCL（原始）", load_json(original_path(dataset))),
        ("CITA-GCL", load_json(pl069_path(dataset))),
    ]
    root = experiment_root(dataset)
    for name, group, stem in SOTAS:
        values.append(
            (name, load_json(root / group / "jobs" / f"{stem}_{dataset}.json"))
        )
    return values


def metric_text(result: dict, key: str, maximum: float) -> str:
    metric = result[key]
    text = f"{float(metric['mean']):.2f} ± {float(metric['std']):.2f}"
    if math.isclose(float(metric["mean"]), maximum, abs_tol=1e-10):
        return f"**{text}**"
    return text


def comparison(candidate: dict, reference: dict) -> tuple[int, int]:
    wins = 0
    strict = 0
    for sota in reference.values():
        current_wins = sum(
            float(candidate[key]["mean"]) > float(sota[key]["mean"])
            for key, _ in METRICS
        )
        wins += current_wins
        strict += int(current_wins == len(METRICS))
    return strict, wins


def build_section() -> str:
    lines = [
        START_MARKER,
        "## 7. 18个数据集完整对照：13个选定 SOTA、ReToGCL 与 CITA-GCL",
        "",
        "本节统一列出当前论文选定的 13 个 SOTA、原始 ReToGCL 和 CITA-GCL。"
        "所有数值均为五折 `均值 ± 标准差`（%）；粗体为该数据集、该指标的最高均值。",
        "SIDER 按 27 个官方任务分别五折后作任务宏平均，Tox21 按 12 个官方任务采用相同口径。",
        "",
    ]
    summary_rows = []
    for dataset in tqdm(DATASETS, desc="生成18数据集结果表", unit="数据集"):
        results = model_results(dataset)
        maxima = {
            key: max(float(result[key]["mean"]) for _, result in results)
            for key, _ in METRICS
        }
        lines.extend([
            f"### 7.{len(summary_rows) + 1} {dataset}",
            "",
            "| 模型 | Accuracy | NMI | ARI | Macro-F1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for name, result in results:
            cells = [metric_text(result, key, maxima[key]) for key, _ in METRICS]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
        original = results[0][1]
        pl069 = results[1][1]
        sota_results = {name: result for name, result in results[2:]}
        pl_strict, pl_metrics = comparison(pl069, sota_results)
        original_strict, original_metrics = comparison(original, sota_results)
        direct = sum(
            float(pl069[key]["mean"]) > float(original[key]["mean"])
            for key, _ in METRICS
        )
        summary_rows.append(
            (dataset, pl_strict, pl_metrics, original_strict, original_metrics, direct)
        )

    lines.extend([
        "### 7.19 汇总比较",
        "",
        "“全超 SOTA”表示四项指标全部严格超过一个 SOTA；“指标胜项”分母为 "
        "`13 × 4 = 52`。",
        "",
        "| 数据集 | CITA-GCL全超SOTA | CITA-GCL指标胜项 | 原始ReToGCL全超SOTA | 原始指标胜项 | CITA-GCL直接胜原始 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for dataset, pl_strict, pl_metrics, original_strict, original_metrics, direct in sorted(
        summary_rows, key=lambda row: (row[1], row[2]), reverse=True
    ):
        lines.append(
            f"| {dataset} | {pl_strict}/13 | {pl_metrics}/52 | "
            f"{original_strict}/13 | {original_metrics}/52 | {direct}/4 |"
        )
    lines.append(
        "| **合计** | "
        f"**{sum(row[1] for row in summary_rows)}/234** | "
        f"**{sum(row[2] for row in summary_rows)}/936** | "
        f"**{sum(row[3] for row in summary_rows)}/234** | "
        f"**{sum(row[4] for row in summary_rows)}/936** | "
        f"**{sum(row[5] for row in summary_rows)}/72** |"
    )
    lines.extend(["", END_MARKER, ""])
    return "\n".join(lines)


def main() -> None:
    content = TABLE_PATH.read_text(encoding="utf-8")
    content = re.sub(r"更新时间：\d{4}-\d{2}-\d{2}", "更新时间：2026-09-04", content, count=1)
    section = build_section()
    if START_MARKER in content and END_MARKER in content:
        prefix, rest = content.split(START_MARKER, 1)
        _, suffix = rest.split(END_MARKER, 1)
        content = prefix.rstrip() + "\n\n" + section + suffix.lstrip("\n")
    else:
        content = content.rstrip() + "\n\n" + section
    TABLE_PATH.write_text(content, encoding="utf-8")
    print(f"已更新：{TABLE_PATH}")


if __name__ == "__main__":
    main()
