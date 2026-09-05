#!/usr/bin/env python3
"""下载并验证 5 个 TDC 医学分子图分类数据集。"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from tqdm import tqdm
from torch_geometric.utils.smiles import from_smiles


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "datasets/main_data"

DATASETS = {
    "hERG_Karim": {
        "repository": "TDC_herg_karim",
        "filename": "tdc_herg_karim.csv",
        "clinical_endpoint": "hERG 钾通道阻断（心脏毒性）",
    },
    "CYP2D6_Veith": {
        "repository": "TDC_cyp2d6_veith",
        "filename": "tdc_cyp2d6_veith.csv",
        "clinical_endpoint": "CYP2D6 酶抑制（药物代谢与相互作用）",
    },
    "CYP3A4_Veith": {
        "repository": "TDC_cyp3a4_veith",
        "filename": "tdc_cyp3a4_veith.csv",
        "clinical_endpoint": "CYP3A4 酶抑制（药物代谢与相互作用）",
    },
    "Pgp_Broccatelli": {
        "repository": "TDC_pgp_broccatelli",
        "filename": "tdc_pgp_broccatelli.csv",
        "clinical_endpoint": "P-gp 抑制（吸收、脑渗透与药物安全）",
    },
    "DILI": {
        "repository": "TDC_dili",
        "filename": "tdc_dili.csv",
        "clinical_endpoint": "药物性肝损伤",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "Project-xmlg dataset downloader"})
    with urlopen(request, timeout=120) as response, temporary.open("wb") as target:
        total = int(response.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=destination.parent.parent.name) as progress:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
                progress.update(len(block))
    temporary.replace(destination)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or set(rows[0]) != {"SMILES", "Y"}:
        raise ValueError(f"{path} 列结构错误，期望 SMILES,Y")
    return rows


def build_graph_cache(name: str, raw_path: Path, processed_path: Path) -> dict:
    rows = load_rows(raw_path)
    graphs = []
    invalid: list[dict[str, object]] = []
    label_counts = {0: 0, 1: 0}
    for index, row in enumerate(tqdm(rows, desc=f"{name} 构图", unit="分子")):
        smiles = row["SMILES"].strip()
        try:
            label = int(float(row["Y"]))
            if label not in label_counts:
                raise ValueError(f"非二分类标签 {label}")
            graph = from_smiles(smiles)
            if graph.num_nodes == 0:
                raise ValueError("分子图没有节点")
        except Exception as error:
            invalid.append({"row_index": index, "smiles": smiles, "reason": str(error)})
            continue
        graph.y = torch.tensor([label], dtype=torch.long)
        graph.source_index = torch.tensor([index], dtype=torch.long)
        label_counts[label] += 1
        graphs.append(graph)
    if min(label_counts.values()) < 5:
        raise ValueError(f"{name} 至少一个类别不足 5 个有效样本：{label_counts}")
    payload = {
        "graphs": graphs,
        "in_dim": int(graphs[0].x.size(-1)),
        "num_classes": 2,
        "metadata": {
            "dataset": name,
            "raw_rows": len(rows),
            "valid_graphs": len(graphs),
            "invalid_smiles_count": len(invalid),
            "invalid_smiles": invalid,
            "label_counts": {str(key): value for key, value in label_counts.items()},
            "raw_sha256": sha256(raw_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = processed_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(processed_path)
    return payload["metadata"]


def main() -> None:
    all_metadata = []
    for name, specification in tqdm(DATASETS.items(), desc="TDC 数据集", unit="个"):
        directory = DATA_ROOT / name
        raw_path = directory / "raw" / specification["filename"]
        url = (
            "https://huggingface.co/datasets/scikit-fingerprints/"
            f"{specification['repository']}/resolve/main/{specification['filename']}"
        )
        if not raw_path.exists():
            download(url, raw_path)
        processed_path = directory / "processed" / "data.pt"
        metadata = build_graph_cache(name, raw_path, processed_path)
        metadata.update({
            "clinical_endpoint": specification["clinical_endpoint"],
            "download_url": url,
            "upstream_dataset": f"Therapeutics Data Commons/{name}",
            "mirror": f"scikit-fingerprints/{specification['repository']}",
            "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
            "processed_path": str(processed_path.relative_to(PROJECT_ROOT)),
        })
        provenance = directory / "provenance.json"
        provenance.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        all_metadata.append(metadata)
    summary = DATA_ROOT / "TDC_MEDICAL_DATASETS.json"
    summary.write_text(json.dumps(all_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"5 个数据集下载、构图和审计完成：{summary}")


if __name__ == "__main__":
    main()
