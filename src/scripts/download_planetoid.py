"""下载并校验 PyTorch Geometric Planetoid 数据集。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from torch_geometric.datasets import Planetoid
from tqdm import tqdm
from urllib3.util.retry import Retry


SUPPORTED_DATASETS = ("Cora", "CiteSeer", "PubMed")
RAW_FILE_SUFFIXES = ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")
RAW_BASE_URL = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载 Cora、CiteSeer 和 PubMed 数据集。"
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


def create_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_raw_files(session: requests.Session, root: Path, name: str) -> None:
    raw_dir = root / name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_key = name.lower()

    for suffix in tqdm(
        RAW_FILE_SUFFIXES,
        desc=f"{name} 原始文件",
        unit="个",
        leave=False,
    ):
        filename = f"ind.{dataset_key}.{suffix}"
        target = raw_dir / filename
        if target.is_file() and target.stat().st_size > 0:
            continue

        temporary = target.with_suffix(target.suffix + ".part")
        url = f"{RAW_BASE_URL}/{filename}"
        with session.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0)) or None
            with temporary.open("wb") as output, tqdm(
                total=total,
                desc=filename,
                unit="B",
                unit_scale=True,
                leave=False,
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        progress.update(len(chunk))
        os.replace(temporary, target)


def download_dataset(
    session: requests.Session, root: Path, name: str
) -> dict[str, int]:
    download_raw_files(session, root, name)
    # Planetoid 会在 root 下自动追加数据集名称。
    dataset = Planetoid(root=str(root), name=name, split="public")
    graph = dataset[0]
    return {
        "nodes": graph.num_nodes,
        "edges": graph.num_edges,
        "features": dataset.num_node_features,
        "classes": dataset.num_classes,
        "train": int(graph.train_mask.sum()),
        "val": int(graph.val_mask.sum()),
        "test": int(graph.test_mask.sum()),
    }


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    results = {}
    with create_session() as session:
        for name in tqdm(args.datasets, desc="下载 Planetoid 数据集", unit="个"):
            results[name] = download_dataset(session, args.root, name)

    print("\n数据集校验结果：")
    for name, stats in results.items():
        print(
            f"- {name}: 节点={stats['nodes']}, 边={stats['edges']}, "
            f"特征={stats['features']}, 类别={stats['classes']}, "
            f"划分={stats['train']}/{stats['val']}/{stats['test']}"
        )


if __name__ == "__main__":
    main()
