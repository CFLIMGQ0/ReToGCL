"""对 110 个研究 idea 执行可复查的外部文献近邻检索。

OpenAlex 用于高召回筛选；脚本只产生候选近邻，不自动宣称新颖。最终结论还需
回到候选论文的 DOI、出版社或会议原始页面检查完整方法链。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG = PROJECT_ROOT / "research/IDEA_CATALOG.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/research_ideas/novelty_screen.json"

FAMILY_QUERIES = {
    1: "graph representation learning intrinsic similarity metric gating",
    2: "graph neural network cycles cellular hypergraph higher order",
    3: "multiscale hypergraph contrastive learning cycle scale",
    4: "multi-view graph representation subset fusion",
    5: "multi-granularity graph reliability uncertainty fusion",
    6: "graph contrastive learning aggregate multiple positive views",
    7: "graph learning data cleaning view-specific noisy samples",
    8: "graph contrastive learning true synthetic positive bidirectional supervision",
    9: "node edge cycle graph cross-level contrastive learning",
    10: "graph representation intrinsic dimension manifold reduction",
    11: "graph masked autoencoder adaptive masking topology",
}

VARIANT_QUERIES = {
    1: (
        "boundary entropy Wasserstein barycenter semantic WL heat kernel",
        "sheaf holonomy loop consistency conflict gate",
        "Riemannian Mahalanobis semi-relaxed Gromov-Wasserstein conformal p-value",
        "curvature counterfactual rewiring response Jacobian similarity expert calibration",
        "augmentation trajectory Koopman spectrum similarity",
        "tropical Hilbert distance max-plus boundary gate",
        "Bures Wasserstein node covariance semantic similarity",
        "local intrinsic dimension mismatch reject semantic neighbors",
        "CKA resistance kernel energy distance triangle metric regularization",
        "counterfactual graph edit path intrinsic similarity",
    ),
    2: (
        "persistent lifetime weighted chordless cycle cell message passing",
        "cycle node hypergraph edge hypergraph incidence optimal transport",
        "Hodge decomposition harmonic residual feedback cycle message",
        "Mobius inversion nested cycle aggregation poset",
        "curvature budget differentiable knapsack adaptive cycle selection",
        "hourglass persistence birth contraction bidirectional cycle message",
        "directed cycle gauge field holonomy learned node potential",
        "cycle space cut space orthogonal residual network",
        "conformal prediction reliable cycle selection",
        "topological phase transition adaptive maximum cycle length",
    ),
    3: (
        "nested hypergraph scale tower zigzag persistence contrastive labels",
        "cross-scale Gromov-Wasserstein hyperedge transport",
        "scale derivative velocity acceleration contrastive representation",
        "node hypergraph edge hypergraph operator commutator",
        "hypergraph Laplacian wavelet packet frequency contrastive",
        "scale intervention Shapley causal fusion",
        "hyperedge birth death memory bank persistence interval positives",
        "multiscale hypergraph mixture experts Sinkhorn load balancing",
        "topological counterfactual cycle insertion scale consistency",
        "hypergraph neural ODE continuous scale flow incidence",
    ),
    4: (
        "view subset lattice Mobius inversion interaction fusion",
        "Shapley DPP diverse view subset selection",
        "views as nodes view subsets as hyperedges fusion",
        "noncommutative ordered view fusion commutator",
        "reliability tensor train sample view scale granularity",
        "counterfactual view ablation hypernetwork fusion operator",
        "partial information atoms Hodge cut harmonic subspace coupling multi-view graph",
        "invertible normalizing flow tree structured multi-view fusion",
        "subjective logic conflict mass explicit conflict view",
        "curriculum view combination synergy confidence lower bound",
    ),
    5: (
        "Mondrian conformal reliability node edge cycle graph bidirectional",
        "Dirichlet evidence conflict cascade graph granularity",
        "PAC-Bayes complexity budget granularity routing",
        "jackknife after bootstrap structural reliability graph",
        "reliability mass conservation Sinkhorn node edge cycle graph",
        "local Lipschitz stability information graph granularity weighting",
        "mutual information bootstrap confidence lower bound routing",
        "Kalman filter dynamic reliability multi-granularity",
        "counterfactual recoverability unique stable information reliability",
        "Shapley interaction graph reliability granularity",
    ),
    6: (
        "Wasserstein barycenter graph augmentation distributions positive sample",
        "Frechet mean mixed curvature manifold augmented views positive",
        "DPP view selection Set Transformer aggregate positive",
        "diffusion denoising latent posterior multiple augmentations positive",
        "Weiszfeld geometric median topology weighted augmentations",
        "augmentation views hypergraph shared substructure pooling positive",
        "capsule optimal transport part matching augmented graph views",
        "minimal sufficient information bottleneck aggregate positive views",
        "Dirichlet opinion credal intersection positive representation",
        "augmentation trajectory geodesic spline zero-noise extrapolation",
    ),
    7: (
        "sample view clean matrix Beta posterior graph noisy data",
        "heterogeneous representation jury co-teaching stop gradient graph cleaning",
        "training dynamics time reversal view-specific noisy sample",
        "influence function prototype topology drift data cleaning graph",
        "conformal three-state clean uncertain sample graph training",
        "noncommuting attribute topology cleaning operators commutator graph contrastive",
        "topological phase continuity multiscale noisy sample cleaning",
        "dual memory bank hysteresis clean noisy sample exchange",
        "minimum description length graph grammar noisy sample weighting",
        "bilevel game data cleaner augmentation corruption exposure",
    ),
    8: (
        "bidirectional relation distillation augmentation positive intrinsic neighbor",
        "semantic topology retrieval path homotopy filling area positive pair graph",
        "causal substructure swap validate positive pair graph",
        "dual Beta posterior product experts positive pair correctness",
        "positive generator discriminator reciprocal graph contrastive",
        "invertible augmentation equivariant bidirectional supervision graph",
        "curriculum reciprocal feedback real pseudo positive samples",
        "dual optimal transport pairing potentials bidirectional supervision",
        "four-valued paraconsistent logic positive pairs contrastive learning",
        "gradient projection true pseudo positive samples graph contrastive",
    ),
    9: (
        "chain complex boundary squared zero node edge cycle graph contrastive",
        "node edge graph optimal transport cycle consistency cross-level",
        "partial information decomposition node edge cycle graph representations",
        "edge representation translator node graph bidirectional prediction",
        "discrete gradient curl harmonic energy cross-level contrastive",
        "dynamic reliability directed distillation node edge cycle graph",
        "counterfactual node edge cycle intervention ANOVA graph representation",
        "node edge cycle graph prototypes simplex barycentric coordinates",
        "masked granularity mutual prediction residual contrastive graph",
        "Koopman modes message passing node edge cycle cross-level",
    ),
    10: (
        "carre du champ neural intrinsic metric graph heat kernel tangent",
        "curvature transition chain Hodge harmonic compression graph representation",
        "diffusion coordinates graph representation contrastive training-only neighbor graph",
        "stratified manifold persistent topology graph representation",
        "Hodge denoising sample relation graph representation curl",
        "semi-relaxed Gromov-Wasserstein graph low-dimensional projection",
        "heat geodesic classification geodesic Pareto metric curvature correction",
        "invertible normalizing flow dimensionality reduction graph contrastive",
        "graph scattering orthogonal learnable residual contrastive",
        "persistence path coordinates sparse dictionary semantic prototypes",
    ),
    11: (
        "reliability recoverability conditioned inverse masking graph autoencoder",
        "node edge cycle conservative mask chain complex Euler characteristic",
        "conformal calibrated difficulty curriculum masking graph",
        "counterfactual minimal sufficient necessary dual mask graph",
        "edge message bandwidth spectral frequency mask cross reconstruction",
        "Shapley DPP adversarial graph masking diverse motifs",
        "Kalman memory recoverability adaptive motif masking curriculum",
        "observed existence counterfactual dual decoder masked graph",
        "Mobius substructure lattice mask higher-order interaction reconstruction",
        "cross-scale two-sided decoder proof reliable masked reconstruction",
    ),
}


def parse_catalog() -> dict[str, dict[str, str]]:
    records = {}
    pattern = re.compile(
        r"\|\s*(D(\d+)-I(\d+))\s*\|\s*\*\*(.*?)\*\*：(.*?)\|\s*(.*?)\|\s*(.*?)\|"
    )
    for line in CATALOG.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            idea_id, family, variant, title, chain, connection, initial = match.groups()
            records[idea_id] = {
                "family": int(family), "variant": int(variant), "title": title.strip(),
                "chain": chain.strip(), "connection": connection.strip(), "initial": initial.strip(),
            }
    if len(records) != 110:
        raise RuntimeError(f"目录应包含 110 个 idea，实际解析到 {len(records)}")
    return records


def query_openalex(query: str, retries: int = 8) -> list[dict]:
    params = {
        "search": query,
        "filter": "from_publication_date:2010-01-01",
        "per-page": 6,
        "select": "id,display_name,publication_year,doi,primary_location,cited_by_count,type",
        "mailto": "literature-audit@example.com",
    }
    for attempt in range(retries):
        response = requests.get("https://api.openalex.org/works", params=params, timeout=30)
        if response.status_code == 200:
            results = []
            for rank, work in enumerate(response.json().get("results", []), start=1):
                location = work.get("primary_location") or {}
                results.append({
                    "rank": rank,
                    "title": work.get("display_name"),
                    "year": work.get("publication_year"),
                    "doi": work.get("doi"),
                    "landing_page": location.get("landing_page_url"),
                    "venue": (location.get("source") or {}).get("display_name"),
                    "cited_by_count": work.get("cited_by_count"),
                    "openalex_id": work.get("id"),
                })
            return results
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
        retry_after = int(response.headers.get("Retry-After", "0") or 0)
        time.sleep(max(retry_after, min(60, 3 * (attempt + 1))))
    raise RuntimeError(f"OpenAlex 连续 {retries} 次请求失败：{query}")


def query_crossref(query: str, retries: int = 5) -> list[dict]:
    params = {
        "query.bibliographic": query,
        "filter": "from-pub-date:2010-01-01",
        "rows": 6,
        "select": "DOI,title,published,container-title,URL,score,type,is-referenced-by-count",
        "mailto": "literature-audit@example.com",
    }
    headers = {
        "User-Agent": "ProjectXMLG-literature-audit/1.0 (mailto:literature-audit@example.com)"
    }
    for attempt in range(retries):
        response = requests.get(
            "https://api.crossref.org/works", params=params, headers=headers, timeout=30
        )
        if response.status_code == 200:
            results = []
            for rank, work in enumerate(response.json()["message"].get("items", []), start=1):
                date_parts = (work.get("published") or {}).get("date-parts", [[None]])
                doi = work.get("DOI")
                results.append({
                    "rank": rank,
                    "title": (work.get("title") or [None])[0],
                    "year": date_parts[0][0] if date_parts and date_parts[0] else None,
                    "doi": f"https://doi.org/{doi}" if doi else None,
                    "landing_page": work.get("URL"),
                    "venue": (work.get("container-title") or [None])[0],
                    "cited_by_count": work.get("is-referenced-by-count"),
                    "crossref_score": work.get("score"),
                })
            return results
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
        time.sleep(min(30, 2 * (attempt + 1)))
    raise RuntimeError(f"Crossref 连续 {retries} 次请求失败：{query}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records = parse_catalog()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    previous = {}
    if output.exists():
        previous = {
            item["idea_id"]: item
            for item in json.loads(output.read_text(encoding="utf-8")).get("ideas", [])
        }
    audited = []
    for idea_id, item in tqdm(records.items(), desc="110 个 idea 外部近邻检索", unit="idea"):
        if idea_id in previous and previous[idea_id].get("candidate_groups"):
            audited.append(previous[idea_id])
            continue
        family, variant = item["family"], item["variant"]
        chain_query = VARIANT_QUERIES[family][variant - 1]
        tokens = chain_query.split()
        component_queries = [
            " ".join(tokens[:3] + ["graph"]),
            " ".join(tokens[-3:] + ["graph"]),
        ]
        candidate_groups = []
        for query in component_queries:
            candidate_groups.append({"query": query, "candidates": query_crossref(query)})
            time.sleep(0.2)
        audited.append({
            "idea_id": idea_id,
            "title": item["title"],
            "complete_chain_query": f"{FAMILY_QUERIES[family]} {chain_query}",
            "candidate_scope": "Crossref 2010-present two-component-pair top-6 relevance screen",
            "candidate_groups": candidate_groups,
            "verdict": "pending_manual_chain_review",
        })
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "metadata": {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "idea_count": len(audited),
                "warning": "高召回近邻筛选，不等价于新颖性结论；需回查论文完整方法链。",
            },
            "ideas": audited,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已完成 {len(audited)} 条外部近邻检索：{output}")


if __name__ == "__main__":
    main()
