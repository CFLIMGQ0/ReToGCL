"""从项目归档结果生成18数据集、35个对比方法与ReToGCL的完整LaTeX附录表。"""

from __future__ import annotations

import glob
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "full_sota_results.tex"
METRICS = (
    "accuracy_percent",
    "nmi_percent",
    "ari_percent",
    "macro_f1_percent",
)

DATASET_SOURCES = {
    "NCI1": "outputs",
    "PROTEINS": "outputs",
    "COLLAB": "outputs",
    "MUTAG": "outputs",
    "COLORS-3": "outputs",
    "PTC_MR": "outputs/three_datasets_20260812",
    "Mutagenicity": "outputs/three_datasets_20260812",
    "AIDS": "outputs/medical_benchmark_extended_20260828",
    "BZR": "outputs/medical_benchmark_extended_20260828",
    "COX2": "outputs/medical_benchmark_extended_20260828",
    "ClinTox": "outputs/medical_benchmark_20260828",
    "BACE": "outputs/medical_benchmark_20260828",
    "BBBP": "outputs/medical_benchmark_20260828",
    "SIDER": "outputs/medical_benchmark_extended_20260828",
    "HIV": "outputs/medical_benchmark_extended_20260828",
    "Tox21": "outputs/medical_benchmark_extended_20260828",
    "ABIDE": "outputs/medical_benchmark_extended_20260828",
    "ADHD200": "outputs/medical_benchmark_extended_20260828",
}

METHODS = (
    "balancegcl", "histograph", "uniimb", "difflift", "khangcl",
    "maxcutpool", "simplicial_mp", "del", "cellclat", "plstm",
    "abstaingnn", "edgeprompt", "dualprism", "invgnn", "genvsexp",
    "kagnn", "multinet", "hspgkn", "dgda", "rsnn", "topoformer",
    "g2pm", "git_tasktree", "gnnplus", "hourglass", "hyperenc",
    "leap", "magedgepool", "nodeid", "prune", "rankfeatures",
    "rspool", "spectre", "tif", "toper",
)

DISPLAY_NAMES = {
    "balancegcl": "BalanceGCL",
    "histograph": "HistoGraph",
    "uniimb": "UniImb",
    "difflift": "DiffLift",
    "khangcl": "KhanGCL",
    "maxcutpool": "MaxCutPool",
    "simplicial_mp": "SimplicialMP",
    "del": "DEL",
    "cellclat": "CellCLAT",
    "plstm": "pLSTM",
    "abstaingnn": "AbstainGNN",
    "edgeprompt": "EdgePrompt",
    "dualprism": "DualPrism",
    "invgnn": "InvGNN",
    "genvsexp": "GenVsExp",
    "kagnn": "KAGNN",
    "multinet": "MultiNet",
    "hspgkn": "HSPGKN",
    "dgda": "DGDA",
    "rsnn": "RSNN",
    "topoformer": "TopoFormer",
    "g2pm": "G2PM",
    "git_tasktree": "Git-TaskTree",
    "gnnplus": "GNN+",
    "hourglass": "Hourglass",
    "hyperenc": "HyperEnc",
    "leap": "LEAP",
    "magedgepool": "MagEdgePool",
    "nodeid": "NodeID",
    "prune": "PRUNE",
    "rankfeatures": "RankFeatures",
    "rspool": "RSPool",
    "spectre": "SPECTRE",
    "tif": "TIF",
    "toper": "TopER",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def comparator_records(dataset: str) -> dict[str, dict]:
    root = PROJECT_ROOT / DATASET_SOURCES[dataset]
    records: dict[str, dict] = {}
    for group in ("sota", "paper_sota", "paper_sota_2025_2026"):
        for path in (root / group / "jobs").glob(f"*_{dataset}.json"):
            record = load_json(path)
            if not all(key in record and "mean" in record[key] for key in METRICS):
                continue
            method = str(record.get("method") or path.stem[: -len(f"_{dataset}")])
            if method in records:
                raise RuntimeError(f"{dataset}/{method} 出现重复结果")
            records[method] = record
    missing = sorted(set(METHODS) - set(records))
    extra = sorted(set(records) - set(METHODS))
    if missing or extra:
        raise RuntimeError(f"{dataset} 对比结果不完整：missing={missing}, extra={extra}")
    return records


def proposed_record(dataset: str) -> dict | None:
    pattern = str(
        PROJECT_ROOT / "**" / f"D7-I10__D7-I01__D10-I04__{dataset}.json"
    )
    paths = [Path(path) for path in glob.glob(pattern, recursive=True)]
    paths = [path for path in paths if path.name.endswith(f"__{dataset}.json")]
    if not paths:
        return None
    if len(paths) > 1:
        raise RuntimeError(f"{dataset} 的ReToGCL结果不唯一：{paths}")
    record = load_json(paths[0])
    if not all(key in record and "mean" in record[key] for key in METRICS):
        raise RuntimeError(f"{paths[0]} 缺少四指标摘要")
    return record


def latex_dataset(dataset: str) -> str:
    return {"PTC_MR": r"PTC\_MR", "ADHD200": "ADHD-200"}.get(dataset, dataset)


def is_zero_partition(record: dict) -> bool:
    return all(
        float(record[key][field]) == 0.0
        for key in ("nmi_percent", "ari_percent")
        for field in ("mean", "std")
    )


def result_cell(record: dict, metric: str, best_mean: float) -> str:
    mean = float(record[metric]["mean"])
    std = float(record[metric]["std"])
    value = rf"\res{{{mean:.2f}}}{{{std:.2f}}}"
    if abs(mean - best_mean) < 1e-10:
        return rf"\textbf{{{value}}}"
    return value


def build_table() -> str:
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{0.88}",
        r"\setlength{\tabcolsep}{8pt}",
        r"\begin{longtable}{@{}lrrrr@{}}",
        r"\caption{Complete unified-project results for 35 comparator adaptations and \method{} across 18 datasets (mean $\pm$ standard deviation, \%). Bold marks the largest completed mean within each dataset and metric; it does not indicate significance. Pale-red rows have exactly zero NMI and ARI mean and standard deviation, a constant-partition signature. Purple rows denote \method{}.}\label{tab:all18full}\\",
        r"\toprule",
        r"Method & Accuracy & NMI & ARI & Macro-F1 \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{c}{\tablename\ \thetable\ --- continued}\\",
        r"\toprule",
        r"Method & Accuracy & NMI & ARI & Macro-F1 \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{5}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for dataset in DATASET_SOURCES:
        comparators = comparator_records(dataset)
        proposed = proposed_record(dataset)
        available = [comparators[method] for method in METHODS]
        if proposed is not None:
            available.append(proposed)
        best = {
            metric: max(float(record[metric]["mean"]) for record in available)
            for metric in METRICS
        }
        lines.extend([
            rf"\multicolumn{{5}}{{l}}{{\rule{{0pt}}{{2.2ex}}\textbf{{{latex_dataset(dataset)}}}}}\\",
            r"\midrule",
        ])
        for method in METHODS:
            record = comparators[method]
            prefix = r"\rowcolor{contrastred!25}" if is_zero_partition(record) else ""
            cells = [result_cell(record, metric, best[metric]) for metric in METRICS]
            lines.append(
                prefix + DISPLAY_NAMES[method] + " & " + " & ".join(cells) + r"\\"
            )
        if proposed is None:
            lines.append(
                r"\rowcolor{topologypurple!45}\method{} & -- & -- & -- & --\\"
            )
        else:
            cells = [result_cell(proposed, metric, best[metric]) for metric in METRICS]
            lines.append(
                r"\rowcolor{topologypurple!45}\method{} & " + " & ".join(cells) + r"\\"
            )
        lines.append(r"\midrule")

    lines.extend([
        r"\end{longtable}",
        r"\renewcommand{\arraystretch}{1}",
        r"\endgroup",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    OUTPUT.write_text(build_table(), encoding="utf-8")
    print(f"已生成：{OUTPUT}")


if __name__ == "__main__":
    main()
