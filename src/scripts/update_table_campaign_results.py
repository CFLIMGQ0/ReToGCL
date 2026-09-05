#!/usr/bin/env python3
"""将新增医学数据集、经典基线和多随机种子结果写入 table.md。"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.run_new_medical5_selected_parallel import DATASETS as NEW_DATASETS, METHODS as NEW_METHODS, ORIGINAL_ID
from src.scripts.run_classic_direct_baselines_parallel import DATASETS as FORMAL_DATASETS, METHODS as CLASSIC_METHODS, Task as ClassicTask, result_path as classic_path
from src.scripts.run_multiseed_validation_parallel import METHODS as MULTISEED_METHODS
from src.scripts.summarize_multiseed_validation import METRICS, path_for as multiseed_path


START = "<!-- CAMPAIGN_20260904_START -->"
END = "<!-- CAMPAIGN_20260904_END -->"
NEW_ROOT = PROJECT_ROOT / "outputs/new_medical5_selected15_20260904"
CLASSIC_ROOT = PROJECT_ROOT / "outputs/classic_direct_baselines_9x6_20260904"
MULTI_ROOT = PROJECT_ROOT / "outputs/multiseed_validation_6x6x10_20260904"
METRIC_TITLES = {
    "accuracy_percent": "Accuracy", "nmi_percent": "NMI",
    "ari_percent": "ARI", "macro_f1_percent": "Macro-F1",
}
DISPLAY = {
    "original": "ReToGCL（原始）", "PL069": "PL069", "retogcl": "ReToGCL",
    "pl069": "PL069", "balancegcl": "BalanceGCL", "khangcl": "Khan-GCL",
    "cellclat": "CellCLAT", "uniimb": "UniImb", "dualprism": "DualPrism",
    "del": "DEL", "spectre": "SpectRe", "toper": "TopER", "leap": "LEAP",
    "hourglass": "Hourglass", "nodeid": "NodeID", "gnnplus": "GNN+",
    "rspool": "RS-Pool", "simplicial_mp": "SimplicialMP",
    "random_gin_linear": "随机初始化 GIN + 线性探测",
    "supervised_gin": "监督式 GIN", "graphcl": "GraphCL", "joaov2": "JOAOv2",
    "rgcl": "RGCL", "simgrace": "SimGRACE",
}


def read_result(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not all(metric in value and math.isfinite(float(value[metric]["mean"])) for metric in METRICS):
        raise ValueError(f"四指标不完整：{path}")
    return value


def new_path(category: str, method: str, dataset: str) -> Path:
    if category == "retogcl":
        return NEW_ROOT / f"retogcl/jobs/{ORIGINAL_ID}__{dataset}.json"
    if category == "pl069":
        return NEW_ROOT / f"pl069/jobs/PL069__{dataset}.json"
    return NEW_ROOT / f"{category}/jobs/{method}_{dataset}.json"


def metric_table(title: str, rows: list[tuple[str, dict]]) -> list[str]:
    lines = [f"### {title}", "", "| 模型 | Accuracy | NMI | ARI | Macro-F1 |", "| --- | ---: | ---: | ---: | ---: |"]
    maxima = {metric: max(float(result[metric]["mean"]) for _, result in rows) for metric in METRICS}
    for name, result in rows:
        cells = []
        for metric in METRICS:
            item = result[metric]
            text = f"{float(item['mean']):.2f} ± {float(item['std']):.2f}"
            if math.isclose(float(item["mean"]), maxima[metric], abs_tol=1e-10):
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def multiseed_cell(item: dict) -> str:
    return (
        f"{float(item['mean']):.2f} ± {float(item['std']):.2f} "
        f"[{float(item['ci95_low']):.2f}, {float(item['ci95_high']):.2f}]"
    )


def directed_pair(summary: dict, dataset: str, reference: str, metric: str) -> dict:
    for item in summary["paired_differences"]:
        if item["dataset"] != dataset or {item["first"], item["second"]} != {"pl069", reference}:
            continue
        result = dict(item[metric])
        if item["first"] != "pl069":
            values = [-float(value) for value in result["values"]]
            count = len(values)
            mean = sum(values) / count
            std = math.sqrt(sum((value - mean) ** 2 for value in values) / (count - 1)) if count > 1 else 0.0
            half_width = (float(result["ci95_high"]) - float(result["ci95_low"])) / 2
            result.update({"values": values, "mean": mean, "std": std, "ci95_low": mean - half_width, "ci95_high": mean + half_width})
        return result
    raise KeyError((dataset, reference, metric))


def build_section() -> str:
    lines = [START, "## 8. 2026-09-04 新增实验", ""]
    lines += [
        "### 8.1 五个新增 TDC 医学分子数据集：13 个选定 SOTA、ReToGCL 与 PL069",
        "",
        "所有数值为统一五折 `均值 ± 标准差`（%）；粗体为同一数据集、同一指标最高均值。",
        "",
    ]
    new_models = [("retogcl", "original"), ("pl069", "PL069")] + [(category, method) for category, method, _ in NEW_METHODS]
    for dataset_index, dataset in enumerate(
        tqdm(NEW_DATASETS, desc="整理新增医学数据", unit="数据集"), start=1
    ):
        rows = [(DISPLAY[method], read_result(new_path(category, method, dataset))) for category, method in new_models]
        lines += metric_table(f"8.1.{dataset_index} {dataset}", rows)

    lines += [
        "### 8.2 六个正式数据集的九个经典直接基线",
        "",
        "随机初始化 GIN 使用冻结的未预训练编码器；监督式 GIN 为端到端五折；其余自监督方法均为预训练后线性探测。",
        "",
    ]
    for dataset in FORMAL_DATASETS:
        rows = [(DISPLAY[method], read_result(classic_path(ClassicTask(method, dataset), CLASSIC_ROOT))) for method in CLASSIC_METHODS]
        lines += metric_table(f"8.2 {dataset}", rows)

    summary = json.loads((MULTI_ROOT / "multiseed_summary.json").read_text(encoding="utf-8"))
    if summary.get("missing"):
        raise RuntimeError(f"多随机种子结果仍缺 {len(summary['missing'])} 项")
    lines += [
        "### 8.3 六模型十随机种子复核",
        "",
        "每格为十个种子级五折均值的 `均值 ± 样本标准差 [95% CI]`（%）。所有模型、所有训练种子共享 `split_seed=42` 的同一套外层五折索引；置信区间采用双侧 Student t（df=9）。",
        "",
        "ReToGCL、PL069、BalanceGCL、Khan-GCL 使用 10 个自监督预训练种子；DEL、SimplicialMP 按项目现有监督式实现使用 10 个训练种子，每个种子分别训练五折，后两者不属于自监督预训练复核。",
        "",
    ]
    for dataset in FORMAL_DATASETS:
        lines += [f"#### {dataset}", "", "| 模型 | Accuracy | NMI | ARI | Macro-F1 |", "| --- | ---: | ---: | ---: | ---: |"]
        records = [(method, summary["aggregate"][f"{method}/{dataset}"]) for method in MULTISEED_METHODS]
        maxima = {metric: max(float(item[metric]["mean"]) for _, item in records) for metric in METRICS}
        for method, item in records:
            cells = []
            for metric in METRICS:
                cell = multiseed_cell(item[metric])
                if math.isclose(float(item[metric]["mean"]), maxima[metric], abs_tol=1e-10):
                    cell = f"**{cell}**"
                cells.append(cell)
            lines.append(f"| {DISPLAY[method]} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "### 8.4 PL069 的种子配对差值",
        "",
        "正值表示 PL069 更高。每格为相同训练种子、相同外层五折索引下 `PL069 − 对照` 的 `均值 ± 标准差 [95% CI]`（百分点）。完整 15 组模型两两配对结果保存在多种子汇总 JSON 中。",
        "",
        "| 数据集 | 对照 | Accuracy | NMI | ARI | Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    references = ("retogcl", "balancegcl", "khangcl", "del", "simplicial_mp")
    for dataset in FORMAL_DATASETS:
        for reference in references:
            cells = [multiseed_cell(directed_pair(summary, dataset, reference, metric)) for metric in METRICS]
            lines.append(f"| {dataset} | {DISPLAY[reference]} | " + " | ".join(cells) + " |")
    lines += ["", END, ""]
    return "\n".join(lines)


def main() -> None:
    table = PROJECT_ROOT / "table.md"
    current = table.read_text(encoding="utf-8")
    section = build_section()
    if START in current:
        before, tail = current.split(START, 1)
        _, after = tail.split(END, 1)
        updated = before.rstrip() + "\n\n" + section + after.lstrip("\n")
    else:
        updated = current.rstrip() + "\n\n" + section
    updated = re.sub(
        r"^更新时间：.*$", "更新时间：" + datetime.now(timezone.utc).date().isoformat(),
        updated, count=1, flags=re.MULTILINE,
    )
    table.write_text(updated, encoding="utf-8")
    print(f"新增实验已写入：{table}")


if __name__ == "__main__":
    main()
