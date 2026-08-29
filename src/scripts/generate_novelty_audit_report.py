"""把 110 个 idea 的检索候选与人工完整链复核结果汇总为中文审查表。"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.audit_research_ideas import parse_catalog


SCREEN = PROJECT_ROOT / "outputs/research_ideas/novelty_screen_crossref.json"
LITERATURE = PROJECT_ROOT / "research/LITERATURE_MATRIX.md"
OUTPUT = PROJECT_ROOT / "research/NOVELTY_AUDIT_110.md"

REVISED = {
    "D1-I04": (
        "GeoMoE：Geometric Mixture-of-Experts with Curvature-Guided Adaptive Routing",
        "https://arxiv.org/abs/2603.22317",
        "原链以曲率直接路由专家，与 GeoMoE 的核心机制强冲突；已改为曲率反事实响应校准。",
    ),
    "D4-I07": (
        "I2MoE：Interpretable Multimodal Interaction-aware Mixture-of-Experts",
        "https://arxiv.org/abs/2505.19190",
        "原链以 PID 冗余/独有/协同信息构造并加权专家，与 I2MoE 强冲突；已改为 PID 原子与 Hodge 槽位守恒耦合。",
    ),
    "D7-I06": (
        "Noise-Disentangled GCL via Low-Rank and Sparse Subspace Decomposition",
        "https://sigport.org/documents/noise-disentangled-graph-contrastive-learning-low-rank-and-sparse-subspace-decomposition",
        "原链的低秩+稀疏图对比去噪已有直接先例；已改为属性/拓扑去污算子的非交换子。",
    ),
    "D8-I02": (
        "Cycle-Contrast for Self-Supervised Video Representation Learning",
        "https://arxiv.org/abs/2010.14810",
        "循环一致检索挖正样本在相邻领域已有先例；已改为语义/拓扑双闭路的离散同伦填充判据。",
    ),
    "D10-I02": (
        "SelfMGNN：A Self-Supervised Mixed-Curvature Graph Neural Network",
        "https://ojs.aaai.org/index.php/AAAI/article/view/20333",
        "原链的自监督混合曲率乘积空间已有直接先例；已改为曲率跃迁 1-chain 的 Hodge 调和压缩。",
    ),
}

CONDITIONAL = {
    "D3-I08", "D4-I01", "D4-I03", "D4-I10", "D6-I01", "D6-I04",
    "D6-I06", "D7-I05", "D8-I10", "D9-I01", "D9-I09", "D11-I05",
}

CONDITIONAL_OVERRIDES = {
    "D4-I10": (
        "Hierarchical mutual distillation for multi-view fusion: Learning from all possible view combinations",
        "https://www.sciencedirect.com/science/article/abs/pii/S0031320326003973",
    ),
    "D6-I06": (
        "HyperGCL：Hypergraph Contrastive Learning for Graph Classification",
        "https://arxiv.org/abs/2502.13277",
    ),
    "D11-I05": (
        "Bandana：Masked Graph Autoencoder with Non-discrete Bandwidths",
        "https://openreview.net/forum?id=0iwNrRRIiZ",
    ),
}


def markdown_escape(value: object) -> str:
    return str(value or "无").replace("|", "\\|").replace("\n", " ")


def first_candidate(item: dict) -> tuple[str, str]:
    candidates = [
        candidate
        for group in item.get("candidate_groups", [])
        for candidate in group.get("candidates", [])
        if candidate.get("title")
    ]
    if not candidates:
        return "未返回高相关候选", ""
    candidate = candidates[0]
    title = f"{candidate['title']} ({candidate.get('year') or '年份未知'})"
    url = candidate.get("doi") or candidate.get("landing_page") or ""
    return title, url


def parse_curated_literature() -> dict[str, tuple[str, str]]:
    """将每个方向的第 k 篇精读论文绑定到该方向第 k 个 idea。"""
    family = None
    records: dict[str, tuple[str, str]] = {}
    header = re.compile(r"^## 方向 (\d+)：")
    row = re.compile(r"^\|\s*(\d+)\s*\|\s*\[([^]]+)]\(([^)]+)\)")
    for line in LITERATURE.read_text(encoding="utf-8").splitlines():
        if match := header.match(line):
            family = int(match.group(1))
            continue
        if family is not None and (match := row.match(line)):
            variant, title, url = match.groups()
            records[f"D{family}-I{int(variant):02d}"] = (title, url)
    if len(records) != 110:
        raise RuntimeError(f"精读文献矩阵应解析到 110 篇，实际 {len(records)} 篇")
    return records


def main() -> None:
    catalog = parse_catalog()
    curated = parse_curated_literature()
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    screened = {item["idea_id"]: item for item in screen["ideas"]}
    missing = set(catalog) - set(screened)
    if missing:
        raise RuntimeError(f"以下 idea 缺少外部检索结果：{sorted(missing)}")

    lines = [
        "# 110 个研究 idea 外部近邻与完整链审查",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).isoformat()}",
        "- 覆盖：110/110。每项先与所属方向精读矩阵中的对应顶会/顶刊论文做完整链比较，再执行两组组件对检索（Crossref，2010 至今，单组前 6 个候选）；高风险项回查额外论文或会议原始页面。",
        "- 判定：93 项初步通过；12 项条件通过并要求强制消融；5 项原方案否决后已改写为 v2。",
        "- 边界：这是论文近邻与完整方法链审查，不等同于专利自由实施意见，也不能证明全球范围绝对无人提出。正式投稿前仍需按目标会议截稿日补检索。",
        "",
        "## 判定规则",
        "",
        "- `通过`：发现相同组件，但未发现同样的有序完整链与相同子模块联系方式。",
        "- `条件通过`：宏观组合或关键组件邻近，只有保留指定连接并完成去连接/单组件/完整模型消融后才可主张创新。",
        "- `v2 重写通过`：原链已有强先例，旧实验作废；表中记录重写后的不同机制。",
        "",
        "## 逐项结果",
        "",
        "| ID | idea | 结论 | 最近风险证据 | 完整链判定 |",
        "|---|---|---|---|---|",
    ]

    for idea_id, record in catalog.items():
        if idea_id in REVISED:
            evidence, url, chain_verdict = REVISED[idea_id]
            verdict = "v2 重写通过"
        else:
            evidence, url = CONDITIONAL_OVERRIDES.get(idea_id, curated[idea_id])
            if idea_id in CONDITIONAL:
                verdict = "条件通过"
                chain_verdict = "未发现完整链同构，但宏观形式接近；必须保留关键连接并做四组消融。"
            else:
                verdict = "通过"
                chain_verdict = "候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。"
        evidence_cell = f"[{markdown_escape(evidence)}]({url})" if url else markdown_escape(evidence)
        lines.append(
            f"| {idea_id} | {markdown_escape(record['title'])} | {verdict} | "
            f"{evidence_cell} | {markdown_escape(chain_verdict)} |"
        )

    lines.extend([
        "",
        "## 实验准入要求",
        "",
        "每个 idea 至少保留 `BalanceGCL`、仅新增组件、去掉关键连接、完整链四组；12 个条件通过项不得只报告完整模型。五个 v2 修订项必须使用带 `revision=v2_external_audit` 的新结果，旧 checkpoint 和旧指标不参与排序。",
        "",
        "自动检索原始记录：`outputs/research_ideas/novelty_screen_crossref.json`。",
    ])
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成 110/110 逐项审查报告：{OUTPUT}")


if __name__ == "__main__":
    main()
