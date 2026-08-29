"""汇总 IDEA_PARA，并对默认配置执行逐折配对与 Holm 校正。"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel


DEFAULT_CONFIG = "idea_loss_weight__0p25"
METRICS = ("accuracy_percent", "nmi_percent", "ari_percent", "macro_f1_percent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("IDEA_PARA"))
    parser.add_argument("--expected", type=int, default=4950)
    return parser.parse_args()


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values); running = 0.0; count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_test(candidate: dict, default: dict) -> tuple[float, float, float]:
    first = np.asarray(candidate["accuracy_percent"]["folds"], dtype=float)
    second = np.asarray(default["accuracy_percent"]["folds"], dtype=float)
    difference = first - second
    mean_difference = float(difference.mean())
    deviation = float(difference.std(ddof=1))
    effect = 0.0 if deviation < 1e-12 else mean_difference / deviation
    if deviation < 1e-12:
        p_value = 0.0 if mean_difference > 0 else 1.0
    else:
        p_value = float(ttest_rel(first, second, alternative="greater").pvalue)
    return mean_difference, effect, p_value


def main() -> None:
    args = parse_args(); root = args.input_dir.resolve()
    experiments = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "jobs").glob("*.json"))]
    if len(experiments) != args.expected:
        raise SystemExit(f"结果尚未齐全：{len(experiments)}/{args.expected}")
    groups = defaultdict(list)
    for item in experiments:
        groups[(item["idea_id"], item["dataset"])].append(item)
    records = []
    for (idea_id, dataset), items in sorted(groups.items()):
        by_config = {item["config_id"]: item for item in items}
        if len(by_config) != 15 or DEFAULT_CONFIG not in by_config:
            raise RuntimeError(f"{idea_id}/{dataset} 配置不完整")
        default = by_config[DEFAULT_CONFIG]
        candidates = [item for item in items if item["config_id"] != DEFAULT_CONFIG]
        raw = [paired_test(item, default) for item in candidates]
        adjusted = holm_adjust([value[2] for value in raw])
        comparisons = []
        for item, (gain, effect, p_value), corrected in zip(candidates, raw, adjusted):
            comparisons.append({
                "config_id": item["config_id"], "parameters": item["parameters"],
                "accuracy_gain_points": round(gain, 4),
                "cohen_dz": round(effect, 4), "p_value_one_sided": p_value,
                "holm_adjusted_p": corrected,
                "exploratory_significant": gain > 0 and corrected < 0.05,
            })
        best = max(items, key=lambda item: item["accuracy_percent"]["mean"])
        best_comparison = next((x for x in comparisons if x["config_id"] == best["config_id"]), None)
        records.append({
            "idea_id": idea_id, "family": best["family"], "dataset": dataset,
            "default_config": DEFAULT_CONFIG, "best_config": best["config_id"],
            "best_parameters": best["parameters"],
            "best_metrics": {key: best[key] for key in METRICS},
            "default_metrics": {key: default[key] for key in METRICS},
            "best_vs_default": best_comparison,
            "comparisons": sorted(comparisons, key=lambda item: item["config_id"]),
        })
    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment_count": len(experiments), "group_count": len(records),
            "test": "one-sided paired t-test on five folds",
            "correction": "Holm within each idea and dataset across 14 non-default candidates",
            "warning": "同一五折用于网格筛选，显著候选必须用独立随机种子复验。",
        },
        "records": records,
    }
    analysis_dir = root / "analysis"; analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = analysis_dir / "parameter_analysis.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    significant = sum(
        bool(item["best_vs_default"] and item["best_vs_default"]["exploratory_significant"])
        for item in records
    )
    lines = [
        "# IDEA 参数消融汇总", "",
        f"- 完整任务：{len(experiments)}", f"- idea×数据集：{len(records)}",
        f"- 最优配置经 Holm 校正后探索性显著的组合：{significant}",
        "- 注意：同一五折参与了参数选择，显著候选仍需独立种子复验。", "",
        "| Idea | 数据集 | 最优配置 | Accuracy | 相对默认提升 | Holm p | 探索性显著 |", "|---|---|---|---:|---:|---:|---|",
    ]
    for item in records:
        comparison = item["best_vs_default"]
        gain = 0.0 if comparison is None else comparison["accuracy_gain_points"]
        p_value = 1.0 if comparison is None else comparison["holm_adjusted_p"]
        significant_text = "是" if comparison and comparison["exploratory_significant"] else "否"
        accuracy = item["best_metrics"]["accuracy_percent"]
        lines.append(
            f"| {item['idea_id']} | {item['dataset']} | `{item['best_config']}` | "
            f"{accuracy['mean']:.2f}±{accuracy['std']:.2f} | {gain:+.2f} | {p_value:.4g} | {significant_text} |"
        )
    markdown_path = analysis_dir / "PARAMETER_ANALYSIS.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"分析完成：{json_path}")
    print(f"摘要完成：{markdown_path}")


if __name__ == "__main__":
    main()
