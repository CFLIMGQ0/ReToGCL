"""把 ABIDE 功能连接与 ADHD-200 结构 MRI 转换为可复用的 PyG 图缓存。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--datasets", nargs="+", choices=("ABIDE", "ADHD200"), default=["ABIDE", "ADHD200"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def save_cache(path: Path, graphs: list[Data], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"graphs": graphs, "in_dim": int(graphs[0].x.size(1)), "num_classes": 2, "metadata": metadata}, temporary)
    temporary.replace(path)
    path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def build_abide(root: Path, force: bool) -> Path:
    directory = root / "ABIDE_PCP"
    output = directory / "processed" / "cc200_top10_qc_ok.pt"
    if output.exists() and not force:
        return output
    phenotype = pd.read_csv(directory / "Phenotypic_V1_0b_preprocessed1.csv")
    rows = phenotype[(phenotype["qc_rater_1"] == "OK") & (phenotype["FILE_ID"] != "no_filename")]
    by_file = {str(row.FILE_ID): row for row in rows.itertuples(index=False)}
    graphs: list[Data] = []
    failures: list[str] = []
    source = directory / "cpac_filt_noglobal_rois_cc200"
    for path in tqdm(sorted(source.glob("*_rois_cc200.1D")), desc="ABIDE 构图", unit="人"):
        file_id = path.name.removesuffix("_rois_cc200.1D")
        row = by_file.get(file_id)
        if row is None:
            continue
        try:
            time_series = np.loadtxt(path, dtype=np.float32)
            if time_series.ndim != 2 or time_series.shape[1] != 200 or time_series.shape[0] < 20:
                raise ValueError(f"异常形状 {time_series.shape}")
            finite = np.isfinite(time_series).all(axis=0)
            time_series[:, ~finite] = 0.0
            centered = time_series - time_series.mean(axis=0, keepdims=True)
            scale = centered.std(axis=0, keepdims=True)
            standardized = centered / np.maximum(scale, 1e-6)
            correlation = np.corrcoef(standardized, rowvar=False)
            correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(correlation, 0.0)
            top_k = 10
            neighbours = np.argpartition(np.abs(correlation), -top_k, axis=1)[:, -top_k:]
            adjacency = np.zeros((200, 200), dtype=bool)
            adjacency[np.arange(200)[:, None], neighbours] = True
            adjacency |= adjacency.T
            source_nodes, target_nodes = np.nonzero(adjacency)
            mean = time_series.mean(axis=0)
            std = time_series.std(axis=0)
            skew = (standardized ** 3).mean(axis=0)
            positive_strength = np.maximum(correlation, 0.0).mean(axis=1)
            negative_strength = np.maximum(-correlation, 0.0).mean(axis=1)
            x = np.stack([mean, std, skew, positive_strength, negative_strength], axis=1)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            graph = Data(
                x=torch.from_numpy(x),
                edge_index=torch.tensor(np.stack([source_nodes, target_nodes]), dtype=torch.long),
                edge_attr=torch.tensor(correlation[source_nodes, target_nodes, None], dtype=torch.float32),
                y=torch.tensor([0 if int(row.DX_GROUP) == 2 else 1], dtype=torch.long),
            )
            graph.subject_id = str(row.SUB_ID)
            graph.site_id = str(row.SITE_ID)
            graphs.append(graph)
        except Exception as error:  # 单个损坏样本不能中断整个批处理
            failures.append(f"{path.name}: {error}")
    metadata = {
        "dataset": "ABIDE",
        "source": "ABIDE PCP C-PAC filt_noglobal CC200",
        "sample_selection": "qc_rater_1 == OK 且本地时间序列存在",
        "graph_protocol": "200 ROI 节点；绝对相关系数每节点 top-10 后对称化",
        "node_features": ["时间均值", "时间标准差", "偏度", "正相关强度", "负相关强度"],
        "samples": len(graphs),
        "class_counts": torch.bincount(torch.cat([g.y for g in graphs]), minlength=2).tolist(),
        "failures": failures,
    }
    if not graphs:
        raise RuntimeError("ABIDE 没有产生有效图")
    save_cache(output, graphs, metadata)
    return output


def grid_edges(size: int = 6) -> torch.Tensor:
    edges: list[tuple[int, int]] = []
    index = lambda a, b, c: (a * size + b) * size + c
    for a in range(size):
        for b in range(size):
            for c in range(size):
                for da, db, dc in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    aa, bb, cc = a + da, b + db, c + dc
                    if aa < size and bb < size and cc < size:
                        first, second = index(a, b, c), index(aa, bb, cc)
                        edges.extend(((first, second), (second, first)))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def adhd_labels(directory: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    for path in sorted(directory.glob("*phenotypic.csv")):
        frame = pd.read_csv(path)
        id_column = "ScanDir ID" if "ScanDir ID" in frame else "ID"
        for row in frame[[id_column, "DX"]].dropna().itertuples(index=False, name=None):
            subject = str(int(row[0]))
            diagnosis = int(row[1])
            if diagnosis >= 0:
                labels[subject] = 0 if diagnosis == 0 else 1
    return labels


def patch_features(volume: np.ndarray, grid_size: int = 6) -> np.ndarray:
    boundaries = [np.linspace(0, length, grid_size + 1, dtype=int) for length in volume.shape]
    global_nonzero = volume[np.isfinite(volume) & (volume != 0)]
    center = float(np.median(global_nonzero)) if global_nonzero.size else 0.0
    spread = float(np.std(global_nonzero)) if global_nonzero.size else 1.0
    normalized = np.nan_to_num((volume - center) / max(spread, 1e-6), nan=0.0, posinf=0.0, neginf=0.0)
    features = []
    for a in range(grid_size):
        for b in range(grid_size):
            for c in range(grid_size):
                patch = normalized[
                    boundaries[0][a]:boundaries[0][a + 1],
                    boundaries[1][b]:boundaries[1][b + 1],
                    boundaries[2][c]:boundaries[2][c + 1],
                ]
                features.append([
                    float(patch.mean()), float(patch.std()), float((patch != 0).mean()),
                    (a + 0.5) / grid_size, (b + 0.5) / grid_size, (c + 0.5) / grid_size,
                ])
    return np.asarray(features, dtype=np.float32)


def build_adhd(root: Path, force: bool) -> Path:
    directory = root / "ADHD200_Anatomical" / "adhd200-preprocessed"
    output = root / "ADHD200_Anatomical" / "processed" / "normalized128_grid6.pt"
    if output.exists() and not force:
        return output
    labels = adhd_labels(directory)
    images = sorted(directory.glob("*/sub-*/normalized_resampled_128_*_T1_biascorr_brain.nii"))
    edge_index = grid_edges(6)
    graphs: list[Data] = []
    failures: list[str] = []
    for path in tqdm(images, desc="ADHD-200 构图", unit="人"):
        subject = path.parent.name.removeprefix("sub-")
        if subject not in labels:
            failures.append(f"{path.parent.name}: 无诊断标签")
            continue
        try:
            volume = np.asarray(nib.load(path).dataobj, dtype=np.float32)
            graph = Data(
                x=torch.from_numpy(patch_features(volume)),
                edge_index=edge_index.clone(),
                y=torch.tensor([labels[subject]], dtype=torch.long),
            )
            graph.subject_id = subject
            graph.site_id = path.parents[1].name
            graphs.append(graph)
        except Exception as error:
            failures.append(f"{path.name}: {error}")
    metadata = {
        "dataset": "ADHD200",
        "source": "ADHD-200 Preprocessed Anatomical Dataset",
        "sample_selection": "存在 normalized_resampled_128 T1 且存在 DX 标签",
        "label_protocol": "DX=0 对照，DX>0 合并为 ADHD",
        "graph_protocol": "标准空间 6×6×6 固定体素块图，三维 6 邻接",
        "node_features": ["块均值", "块标准差", "非零比例", "归一化 x", "归一化 y", "归一化 z"],
        "samples": len(graphs),
        "class_counts": torch.bincount(torch.cat([g.y for g in graphs]), minlength=2).tolist(),
        "failures": failures,
    }
    if not graphs:
        raise RuntimeError("ADHD-200 没有产生有效图")
    save_cache(output, graphs, metadata)
    return output


def main() -> None:
    args = parse_args()
    root = args.data_root if args.data_root.is_absolute() else (PROJECT_ROOT / args.data_root).resolve()
    for dataset in args.datasets:
        path = build_abide(root, args.force) if dataset == "ABIDE" else build_adhd(root, args.force)
        print(f"已生成：{path}")


if __name__ == "__main__":
    main()
