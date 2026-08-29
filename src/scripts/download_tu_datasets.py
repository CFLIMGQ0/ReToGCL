"""下载并校验常用 TU Dortmund 图分类数据集。"""

from __future__ import annotations

import argparse
from pathlib import Path

from torch_geometric.datasets import TUDataset
from tqdm import tqdm


SUPPORTED_DATASETS = ("NCI1", "PROTEINS", "COLLAB", "MUTAG", "COLORS-3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载项目支持的 TU Dortmund 图分类数据集。"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/main_data"),
        help="数据集根目录（默认：datasets/main_data）。",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        default=list(SUPPORTED_DATASETS),
        help="需要下载的数据集（默认：全部）。",
    )
    return parser.parse_args()


def load_and_validate(root: Path, name: str) -> dict[str, int | float]:
    dataset = TUDataset(
        root=str(root), name=name, cleaned=False,
        use_node_attr=name == "COLORS-3",
    )

    total_nodes = 0
    total_edges = 0
    for graph in tqdm(dataset, desc=f"校验 {name}", unit="图", leave=False):
        total_nodes += graph.num_nodes
        total_edges += graph.num_edges

    graph_count = len(dataset)
    return {
        "graphs": graph_count,
        "nodes": total_nodes,
        "edges": total_edges,
        "avg_nodes": total_nodes / graph_count,
        "avg_edges": total_edges / graph_count,
        "features": dataset.num_node_features,
        "edge_features": dataset.num_edge_features,
        "classes": dataset.num_classes,
    }


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in tqdm(args.datasets, desc="下载 TU 数据集", unit="个"):
        results[name] = load_and_validate(args.root, name)

    print("\n数据集校验结果：")
    for name, stats in results.items():
        print(
            f"- {name}: 图={stats['graphs']}, 节点={stats['nodes']}, "
            f"边={stats['edges']}, 平均节点={stats['avg_nodes']:.2f}, "
            f"平均边={stats['avg_edges']:.2f}, 节点特征={stats['features']}, "
            f"边特征={stats['edge_features']}, 类别={stats['classes']}"
        )


if __name__ == "__main__":
    main()
