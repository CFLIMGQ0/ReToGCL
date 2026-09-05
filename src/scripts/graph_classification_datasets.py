"""统一加载 TU、MoleculeNet、TDC 与脑图分类数据集。"""

from __future__ import annotations

from pathlib import Path
import math

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet, TUDataset

from src.baselines.gcl_baselines import ensure_features


TU_DATASETS = (
    "NCI1", "PROTEINS", "COLLAB", "MUTAG", "COLORS-3",
    "PTC_MR", "Mutagenicity", "AIDS", "BZR", "COX2",
)
MOLECULENET_DATASETS = ("ClinTox", "BACE", "BBBP", "SIDER", "HIV", "Tox21")
BRAIN_DATASETS = ("ABIDE", "ADHD200")
TDC_DATASETS = (
    "hERG_Karim", "CYP2D6_Veith", "CYP3A4_Veith", "Pgp_Broccatelli", "DILI",
)
MULTITASK_DATASETS = ("SIDER", "Tox21")
GRAPH_CLASSIFICATION_DATASETS = TU_DATASETS + MOLECULENET_DATASETS + BRAIN_DATASETS + TDC_DATASETS


def dataset_protocol(name: str) -> dict[str, object]:
    """返回数据标签口径，写入实验产物以保证结果可追溯。"""
    if name == "ClinTox":
        return {
            "source": "PyG MoleculeNet",
            "label_protocol": "joint_FDA_APPROVED_CT_TOX_multiclass",
            "label_columns": ["FDA_APPROVED", "CT_TOX"],
            "joint_classes": {
                "0": [0, 1],
                "1": [1, 0],
                "2": [1, 1],
            },
            "note": "用于统一 Accuracy/NMI/ARI/Macro-F1；不是官方双任务 ROC-AUC 口径。",
        }
    if name in {"BACE", "BBBP"}:
        return {
            "source": "PyG MoleculeNet",
            "label_protocol": "single_task_binary_classification",
        }
    if name in MULTITASK_DATASETS:
        return {
            "source": "PyG MoleculeNet",
            "label_protocol": "official_binary_tasks_independent_five_fold_macro_average",
            "task_count": 27 if name == "SIDER" else 12,
            "missing_labels": "逐任务排除 NaN，不填补",
            "note": "每个官方任务独立分层五折；最终对任务宏平均 Accuracy/NMI/ARI/Macro-F1。",
        }
    if name == "HIV":
        return {
            "source": "PyG MoleculeNet",
            "label_protocol": "single_task_binary_classification",
        }
    if name == "ABIDE":
        return {
            "source": "ABIDE PCP C-PAC filt_noglobal CC200",
            "label_protocol": "ASD_vs_control_binary_classification",
            "sample_selection": "qc_rater_1 == OK",
            "graph_protocol": "CC200 ROI correlation top-10 graph",
        }
    if name == "ADHD200":
        return {
            "source": "ADHD-200 Preprocessed Anatomical Dataset",
            "label_protocol": "all_ADHD_subtypes_vs_control_binary_classification",
            "graph_protocol": "normalized T1 fixed 6x6x6 patch graph",
        }
    if name in TDC_DATASETS:
        endpoints = {
            "hERG_Karim": "hERG_channel_blockade_cardiotoxicity",
            "CYP2D6_Veith": "CYP2D6_inhibition_drug_metabolism",
            "CYP3A4_Veith": "CYP3A4_inhibition_drug_metabolism",
            "Pgp_Broccatelli": "Pgp_inhibition_absorption_and_drug_safety",
            "DILI": "drug_induced_liver_injury",
        }
        return {
            "source": "Therapeutics Data Commons via scikit-fingerprints mirror",
            "label_protocol": "single_task_binary_classification",
            "clinical_endpoint": endpoints[name],
            "split_protocol": "project_stratified_5_fold",
        }
    return {
        "source": "PyG TUDataset",
        "label_protocol": "graph_classification_label",
    }


def _load_moleculenet(root: Path, name: str) -> tuple[list[Data], int, int]:
    dataset = MoleculeNet(root=str(root), name=name)
    joint_mapping: dict[tuple[int, ...], int] = {}
    if name == "ClinTox":
        observed = sorted({tuple(int(value) for value in graph.y.view(-1)) for graph in dataset})
        joint_mapping = {label: index for index, label in enumerate(observed)}

    graphs: list[Data] = []
    for index in range(len(dataset)):
        graph = ensure_features(dataset[index].clone())
        raw_label = graph.y.view(-1)
        if name in MULTITASK_DATASETS:
            graph.task_y = raw_label.float().view(1, -1)
            finite = torch.isfinite(raw_label)
            if not finite.any():
                raise ValueError(f"{name} 第 {index} 个图的所有任务均缺失标签。")
            graph.y = raw_label[finite][0].long().view(1)
        elif not torch.isfinite(raw_label).all():
            raise ValueError(f"{name} 第 {index} 个图含缺失标签，当前分类协议拒绝静默填充。")
        elif name == "ClinTox":
            key = tuple(int(value) for value in raw_label)
            graph.y = torch.tensor([joint_mapping[key]], dtype=torch.long)
        elif name not in MULTITASK_DATASETS:
            graph.y = raw_label.long().view(1)
        graphs.append(graph)

    in_dim = max(1, int(dataset.num_node_features))
    num_classes = len(joint_mapping) if joint_mapping else (2 if name in MULTITASK_DATASETS else int(dataset.num_classes))
    return graphs, in_dim, num_classes


def _load_brain(root: Path, name: str) -> tuple[list[Data], int, int]:
    relative = {
        "ABIDE": Path("ABIDE_PCP/processed/cc200_top10_qc_ok.pt"),
        "ADHD200": Path("ADHD200_Anatomical/processed/normalized128_grid6.pt"),
    }[name]
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"{name} 图缓存不存在：{path}；请先运行 build_brain_graph_datasets.py。")
    value = torch.load(path, map_location="cpu", weights_only=False)
    return value["graphs"], int(value["in_dim"]), int(value["num_classes"])


def _load_tdc(root: Path, name: str) -> tuple[list[Data], int, int]:
    path = root / name / "processed" / "data.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{name} 图缓存不存在：{path}；请先运行 download_tdc_medical_datasets.py。"
        )
    value = torch.load(path, map_location="cpu", weights_only=False)
    graphs = [ensure_features(graph.clone()) for graph in value.get("graphs", [])]
    if not graphs:
        raise ValueError(f"{name} 图缓存为空：{path}")
    return graphs, int(value["in_dim"]), int(value["num_classes"])


def multitask_graphs(graphs: list[Data], task_index: int) -> list[Data]:
    """取一个多任务标签列并排除该列缺失样本，供独立五折使用。"""
    selected: list[Data] = []
    for graph in graphs:
        label = graph.task_y.view(-1)[task_index]
        if torch.isfinite(label):
            item = graph.clone()
            item.y = label.long().view(1)
            selected.append(item)
    return selected


def aggregate_multitask_results(dataset: str, task_results: list[dict]) -> dict:
    """把各官方二分类任务的五折摘要合并为任务宏平均摘要。"""
    metric_keys = ("accuracy_percent", "nmi_percent", "ari_percent", "macro_f1_percent")
    aggregated: dict[str, object] = {
        "dataset": dataset,
        "task": "multitask_graph_classification",
        "training": "official_tasks_independent_five_fold_then_task_macro_average",
        "fold_protocol": "per_task_stratified_5_fold",
        "task_count": len(task_results),
        "per_task_results": task_results,
    }
    for key in metric_keys:
        means = [float(result[key]["mean"]) for result in task_results]
        standard_deviations = [float(result[key]["std"]) for result in task_results]
        grand_mean = sum(means) / max(len(means), 1)
        # 合并任务间差异与任务内五折方差，避免只汇报任务均值而低估波动。
        variance = sum(
            std ** 2 + (mean - grand_mean) ** 2
            for mean, std in zip(means, standard_deviations, strict=True)
        ) / max(len(means), 1)
        aggregated[key] = {"mean": grand_mean, "std": math.sqrt(max(variance, 0.0))}
    return aggregated


def load_graphs(root: Path, name: str) -> tuple[list[Data], int, int]:
    """加载图列表、节点输入维数和类别数。"""
    if name in MOLECULENET_DATASETS:
        return _load_moleculenet(root, name)
    if name in BRAIN_DATASETS:
        return _load_brain(root, name)
    if name in TDC_DATASETS:
        return _load_tdc(root, name)
    if name not in TU_DATASETS:
        raise ValueError(f"不支持的数据集：{name}")

    dataset = TUDataset(
        root=str(root), name=name, cleaned=False,
        use_node_attr=name == "COLORS-3",
    )
    graphs: list[Data] = []
    for index in range(len(dataset)):
        graph = ensure_features(dataset[index].clone())
        graph.y = graph.y.long().view(1)
        graphs.append(graph)
    return graphs, max(1, int(dataset.num_node_features)), int(dataset.num_classes)
