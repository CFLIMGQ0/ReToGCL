"""基于 BalanceGCL 的 110 个研究 idea 最小可执行原型。

本模块把 ``research/IDEA_CATALOG.md`` 中的 11 个方向分别实现为一个研究族，
每个研究族包含 10 条彼此不同的表示连接或辅助目标。这里的目标是快速、统一地
验证梯度、显存和训练协议；精确持久同调、GW、共形校准等昂贵算子在首轮使用
可微代理，不能把冒烟结果表述为对应完整算法的最终实验结果。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import scatter

from src.models.idea_parameter_registry import parameter_space
from src.models.sota_graph_models import BalancedGCL


FAMILY_NAMES = (
    "多空间本质相似度",
    "环与高阶连接",
    "多视图多尺度超边",
    "多视图子集融合",
    "多粒度可靠性",
    "综合正样本生成",
    "分维度自动清洗",
    "真假正样本双向监督",
    "点边环图跨层对比",
    "高维空间整理与降维",
    "可学习掩码",
)

IDEA_IDS = tuple(f"D{family}-I{variant:02d}" for family in range(1, 12) for variant in range(1, 11))
REVISED_IDEA_IDS = frozenset({"D1-I04", "D4-I07", "D7-I06", "D8-I02", "D10-I02"})
STRENGTHENED_IDEA_IDS = frozenset({"D11-I05"})


@dataclass(frozen=True)
class IdeaSpec:
    idea_id: str
    family: int
    variant: int
    family_name: str
    fidelity: str = "differentiable_prototype"
    revision: str = "v1"


def get_idea_spec(idea_id: str) -> IdeaSpec:
    match = re.fullmatch(r"D(\d+)-I(\d{2})", idea_id.upper())
    if match is None:
        raise ValueError(f"无效 idea ID：{idea_id}")
    family, variant = map(int, match.groups())
    if not 1 <= family <= 11 or not 1 <= variant <= 10:
        raise ValueError(f"idea ID 超出范围：{idea_id}")
    normalized = idea_id.upper()
    if normalized in REVISED_IDEA_IDS:
        revision = "v2_external_audit"
    elif normalized in STRENGTHENED_IDEA_IDS:
        revision = "v1_strengthened_external_audit"
    else:
        revision = "v1_external_audit"
    return IdeaSpec(normalized, family, variant, FAMILY_NAMES[family - 1], revision=revision)


def _standardize(values: Tensor, dim: int = -1) -> Tensor:
    return (values - values.mean(dim=dim, keepdim=True)) / values.std(
        dim=dim, keepdim=True, unbiased=False
    ).clamp_min(1e-5)


def _symmetric_nce(first: Tensor, second: Tensor, weights: Tensor | None = None, temperature: float = 0.2) -> Tensor:
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    logits = first @ second.t() / temperature
    labels = torch.arange(first.size(0), device=first.device)
    forward = F.cross_entropy(logits, labels, reduction="none")
    backward = F.cross_entropy(logits.t(), labels, reduction="none")
    losses = 0.5 * (forward + backward)
    if weights is None:
        return losses.mean()
    weights = weights.detach().clamp_min(0)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def _off_diagonal_mean(matrix: Tensor) -> Tensor:
    if matrix.size(0) < 2:
        return matrix.new_zeros(())
    mask = ~torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)
    return matrix[mask].mean()


class ResearchIdeaBalanceGCL(BalancedGCL):
    """共享 BalanceGCL 主干、按 ID 切换新连接算子的统一模型。"""

    def __init__(
        self,
        idea_id: str,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        auxiliary_weight: float = 0.25,
        idea_temperature: float = 0.2,
        operator_mix: float = 1.0,
        idea_parameters: dict[str, float] | None = None,
    ):
        super().__init__(in_dim, hidden_dim, layers, num_classes)
        self.spec = get_idea_spec(idea_id)
        dim = self.output_dim
        self.auxiliary_weight = auxiliary_weight
        if idea_temperature <= 0:
            raise ValueError("idea_temperature 必须大于 0")
        if operator_mix < 0:
            raise ValueError("operator_mix 必须大于等于 0")
        self.idea_temperature = float(idea_temperature)
        self.operator_mix = float(operator_mix)
        space = parameter_space(self.spec.idea_id)
        supplied = {} if idea_parameters is None else dict(idea_parameters)
        unknown = sorted(set(supplied) - set(space.defaults))
        if unknown:
            raise ValueError(f"{self.spec.idea_id} 收到非专属参数：{unknown}")
        self.idea_parameters = {**space.defaults, **supplied}
        self.idea_parameter_names = tuple(parameter.name for parameter in space.parameters)
        multipliers = []
        for parameter in space.parameters:
            value = float(self.idea_parameters[parameter.name])
            if not any(math.isclose(value, candidate, rel_tol=1e-9, abs_tol=1e-12) for candidate in parameter.values):
                raise ValueError(
                    f"{self.spec.idea_id}/{parameter.name}={value} 不在候选值 {parameter.values} 中"
                )
            # 三个专属参数分别控制 idea 残差剂量、关系分辨率和辅助约束剂量。
            # 默认点的倍率为 1；含 0 的消融点会把对应新增子组件置零。
            if math.isclose(parameter.default, 0.0, abs_tol=1e-12):
                candidate_index = min(
                    range(len(parameter.values)),
                    key=lambda index: abs(parameter.values[index] - value),
                )
                default_index = min(
                    range(len(parameter.values)),
                    key=lambda index: abs(parameter.values[index] - parameter.default),
                )
                radius = max(default_index, len(parameter.values) - 1 - default_index, 1)
                multipliers.append(1.0 + (candidate_index - default_index) / (2.0 * radius))
            else:
                multipliers.append(value / parameter.default)
        self.idea_parameter_multipliers = tuple(float(value) for value in multipliers)
        self.view_heads = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        self.scale_heads = nn.ModuleList([nn.Linear(hidden_dim, dim) for _ in range(3)])
        self.fusion_gate = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 4))
        self.reconstructor = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim))
        self.bottleneck = nn.Sequential(nn.Linear(dim, max(8, dim // 4)), nn.Tanh(), nn.Linear(max(8, dim // 4), dim))
        self.flow_scale = nn.Parameter(torch.zeros(dim))
        self.metric = nn.Parameter(torch.zeros(4, dim))
        self.subset_logits = nn.Parameter(torch.zeros(7))
        self.mask_logits = nn.Parameter(torch.zeros(dim))
        self.register_buffer("reliability_ema", torch.full((4,), 0.5))
        self.register_buffer("clean_ema", torch.full((4,), 0.5))

    def _bundle_from_history(self, data: Data, history: Tensor) -> dict[str, Tensor]:
        """用共享编码历史构造 idea 专属视图，供多模块联合模型复用主干。"""

        nodes = history.flatten(1)
        graph_count = int(data.batch.max().item()) + 1
        graph = global_add_pool(nodes, data.batch, size=graph_count)

        source, target = data.edge_index
        if source.numel():
            edge_nodes = 0.5 * (nodes[source] + nodes[target])
            edge_graph = data.batch[source]
            edge = global_mean_pool(edge_nodes, edge_graph, size=graph_count)
            degree = scatter(
                torch.ones_like(source, dtype=nodes.dtype), source, dim=0,
                dim_size=nodes.size(0), reduce="sum",
            ).clamp_min(1)
            neighbor = scatter(nodes[target], source, dim=0, dim_size=nodes.size(0), reduce="sum")
            neighbor = neighbor / degree.unsqueeze(-1)
            # 二阶闭合残差是环/高阶信息的廉价可微代理。
            cycle_nodes = (nodes - neighbor).abs() / degree.sqrt().unsqueeze(-1)
            cycle = global_mean_pool(cycle_nodes, data.batch, size=graph_count)
        else:
            edge = graph.new_zeros(graph.shape)
            cycle = graph.new_zeros(graph.shape)

        node = global_mean_pool(nodes, data.batch, size=graph_count)
        raw_views = (node, edge, cycle, graph)
        views = torch.stack(
            [F.normalize(head(value), dim=-1) for head, value in zip(self.view_heads, raw_views)],
            dim=1,
        )

        layer_indices = (0, history.size(1) // 2, history.size(1) - 1)
        scales = []
        for head, index in zip(self.scale_heads, layer_indices):
            pooled = global_mean_pool(history[:, index], data.batch, size=graph_count)
            scales.append(F.normalize(head(pooled), dim=-1))
        return {
            "graph": graph,
            "nodes": nodes,
            "views": views,
            "scales": torch.stack(scales, dim=1),
        }

    def _bundle(self, data: Data) -> dict[str, Tensor]:
        return self._bundle_from_history(data, self.encoder(data))

    def encode(self, data: Data) -> Tensor:
        return self._bundle(data)["graph"]

    @staticmethod
    def _view_evidence(first: Tensor, second: Tensor) -> Tensor:
        agreement = (F.normalize(first, dim=-1) * F.normalize(second, dim=-1)).sum(-1)
        stability = torch.exp(-(first - second).pow(2).mean(-1))
        diversity = first.var(-1, unbiased=False).sqrt().clamp_max(1)
        return torch.stack((agreement, stability, diversity), dim=-1)

    def _family1(self, a: dict[str, Tensor], b: dict[str, Tensor], variant: int) -> tuple[Tensor, Tensor, Tensor]:
        va, vb = a["views"][:, :3], b["views"][:, :3]
        evidence = self._view_evidence(va, vb)
        similarity, stability, spread = evidence.unbind(-1)
        if variant == 1:  # 边界条件 Wasserstein 重心代理
            entropy = -(similarity.softmax(-1) * similarity.log_softmax(-1)).sum(-1, keepdim=True)
            weights = (similarity + entropy * torch.tensor([0.0, 0.5, 1.0], device=va.device)).softmax(-1)
        elif variant == 2:  # holonomy 冲突门代理
            loop_residual = (va[:, 0] - va[:, 1] + va[:, 2] - vb[:, 2]).abs().mean(-1)
            weights = torch.stack((similarity[:, 0], similarity[:, 1] - loop_residual, similarity[:, 2] + loop_residual), -1).softmax(-1)
        elif variant == 3:  # 共形化三距离
            weights = torch.sigmoid(_standardize(similarity, 0)).div(torch.sigmoid(_standardize(similarity, 0)).sum(-1, keepdim=True))
        elif variant == 4:  # 曲率反事实仅定义干预，响应失配才决定专家可信度
            curvature_intervention = (
                (va[:, 2] - va[:, 1]).norm(dim=-1)
                - (vb[:, 2] - vb[:, 1]).norm(dim=-1)
            ).abs().clamp_min(1e-3)
            expert_response = (va - vb).norm(dim=-1) / curvature_intervention.unsqueeze(-1)
            observed_response = (
                a["scales"][:, -1] - b["scales"][:, -1]
            ).norm(dim=-1) / curvature_intervention
            response_mismatch = (
                _standardize(expert_response, 0)
                - _standardize(observed_response, 0).unsqueeze(-1)
            ).abs()
            weights = (similarity - response_mismatch).softmax(-1)
        elif variant == 5:  # Koopman 轨迹稳定性
            dynamics = (a["scales"][:, 2] - a["scales"][:, 1] - a["scales"][:, 1] + a["scales"][:, 0]).pow(2).mean(-1)
            weights = torch.stack((similarity[:, 0], similarity[:, 1], similarity[:, 2] - dynamics), -1).softmax(-1)
        elif variant == 6:  # 热带 max-plus 门
            tropical = va.amax(-1) - va.amin(-1)
            weights = (similarity - tropical + tropical.amax(-1, keepdim=True)).softmax(-1)
        elif variant == 7:  # Bures 二阶代理
            second_order = (va.pow(2).mean(-1).sqrt() - vb.pow(2).mean(-1).sqrt()).abs()
            weights = (similarity - second_order).softmax(-1)
        elif variant == 8:  # 局部维数失配否决门
            effective_dim = va.abs().sum(-1).pow(2) / va.pow(2).sum(-1).clamp_min(1e-6)
            mismatch = (effective_dim - effective_dim.roll(1, 0)).abs() / va.size(-1)
            weights = (similarity - mismatch).softmax(-1)
        elif variant == 9:  # 三角闭合门
            triangle = (va - va.roll(1, 0)).norm(dim=-1) + (va.roll(1, 0) - va.roll(2, 0)).norm(dim=-1)
            direct = (va - va.roll(2, 0)).norm(dim=-1)
            weights = (similarity - F.relu(direct - triangle)).softmax(-1)
        else:  # 最小反事实路径一致
            path = (va - vb).abs().cumsum(dim=1)
            weights = (similarity - path.mean(-1)).softmax(-1)
        za = (weights.unsqueeze(-1) * va).sum(1)
        zb = (weights.unsqueeze(-1) * vb).sum(1)
        auxiliary = -(weights * similarity).sum(-1).mean() + 0.01 * (weights * weights.clamp_min(1e-8).log()).sum(-1).mean()
        return za, zb, auxiliary

    def _family2(self, a: dict[str, Tensor], b: dict[str, Tensor], variant: int) -> tuple[Tensor, Tensor, Tensor]:
        va, vb = a["views"], b["views"]
        node_a, edge_a, cycle_a, graph_a = va.unbind(1)
        node_b, edge_b, cycle_b, graph_b = vb.unbind(1)
        if variant == 1:  # 持久寿命权
            weight = torch.sigmoid((cycle_a - edge_a).norm(dim=-1, keepdim=True))
            za, zb = graph_a + weight * cycle_a, graph_b + weight * cycle_b
        elif variant == 2:  # 节点/边超图 incidence OT 代理
            coupling = torch.sigmoid((node_a * edge_a).sum(-1, keepdim=True))
            za, zb = graph_a + coupling * (node_a + edge_a), graph_b + coupling * (node_b + edge_b)
        elif variant == 3:  # Hodge 残差反馈
            residual_a, residual_b = cycle_a - (edge_a - node_a), cycle_b - (edge_b - node_b)
            za, zb = graph_a + residual_a, graph_b + residual_b
        elif variant == 4:  # Möbius 去重
            za, zb = graph_a + cycle_a - edge_a + node_a, graph_b + cycle_b - edge_b + node_b
        elif variant == 5:  # 曲率/成本预算
            gain = (cycle_a * graph_a).sum(-1, keepdim=True)
            cost = (cycle_a - edge_a).norm(dim=-1, keepdim=True)
            gate = torch.sigmoid(gain - cost)
            za, zb = graph_a + gate * cycle_a, graph_b + gate * cycle_b
        elif variant == 6:  # Hourglass 双向一致
            born_a, die_a = cycle_a + edge_a, cycle_a - edge_a
            born_b, die_b = cycle_b + edge_b, cycle_b - edge_b
            gate = torch.sigmoid((born_a * die_a).sum(-1, keepdim=True))
            za, zb = graph_a + gate * (born_a + die_a), graph_b + gate * (born_b + die_b)
        elif variant == 7:  # 规范不变 holonomy
            za, zb = graph_a + (cycle_a * edge_a.sign()).abs(), graph_b + (cycle_b * edge_b.sign()).abs()
        elif variant == 8:  # cycle/cut 正交残差
            projection_a = (cycle_a * edge_a).sum(-1, keepdim=True) * edge_a
            projection_b = (cycle_b * edge_b).sum(-1, keepdim=True) * edge_b
            za, zb = graph_a + cycle_a - projection_a, graph_b + cycle_b - projection_b
        elif variant == 9:  # 环共形筛选代理
            score = _standardize((cycle_a - cycle_b).pow(2).mean(-1), 0)
            gate = torch.sigmoid(-score).unsqueeze(-1)
            za, zb = graph_a + gate * cycle_a, graph_b + gate * cycle_b
        else:  # 相变点双尺度
            gap_a, gap_b = (cycle_a - edge_a).abs(), (cycle_b - edge_b).abs()
            gate = torch.sigmoid(_standardize(gap_a.mean(-1), 0)).unsqueeze(-1)
            za, zb = graph_a + gate * cycle_a + (1 - gate) * edge_a, graph_b + gate * cycle_b + (1 - gate) * edge_b
        auxiliary = 1 - F.cosine_similarity(za, zb).mean()
        return za, zb, auxiliary

    def _family3(self, a: dict[str, Tensor], b: dict[str, Tensor], variant: int) -> tuple[Tensor, Tensor, Tensor]:
        sa, sb = a["scales"], b["scales"]
        delta_a, delta_b = sa[:, 1:] - sa[:, :-1], sb[:, 1:] - sb[:, :-1]
        if variant == 1:  # zigzag：稳定事件吸引、突变事件排斥
            event = torch.exp(-delta_a.pow(2).mean(-1))
            weights = torch.cat((event, event[:, -1:]), 1).softmax(-1)
        elif variant == 2:  # 跨尺度 GW 软耦合
            cost = torch.cdist(sa, sb).mean(0)
            weights = (-cost.diag()).softmax(0).expand(sa.size(0), -1)
        elif variant == 3:  # 尺度速度/加速度
            acceleration = (delta_a[:, 1] - delta_a[:, 0]).pow(2).mean(-1, keepdim=True)
            weights = torch.cat((delta_a.norm(dim=-1), acceleration), 1).neg().softmax(-1)
        elif variant == 4:  # 节点/边算子交换子代理
            commutator = (a["views"][:, 0] * a["views"][:, 1].roll(1, -1) - a["views"][:, 1] * a["views"][:, 0].roll(1, -1)).abs().mean(-1, keepdim=True)
            weights = torch.cat((1 - commutator, commutator, torch.ones_like(commutator)), 1).softmax(-1)
        elif variant == 5:  # 小波频带
            energy = torch.stack((sa[:, 0].pow(2).mean(-1), delta_a[:, 0].pow(2).mean(-1), delta_a[:, 1].pow(2).mean(-1)), -1)
            weights = energy.softmax(-1)
        elif variant == 6:  # Shapley 缺失贡献
            total = sa.mean(1)
            contribution = torch.stack([(total - (sa.sum(1) - sa[:, k]) / 2).norm(dim=-1) for k in range(3)], -1)
            weights = contribution.softmax(-1)
        elif variant == 7:  # 出生—死亡区间重叠代理
            interval = torch.stack((sa.amin(-1), sa.amax(-1)), -1)
            overlap = (torch.minimum(interval[..., 1], interval.roll(1, 1)[..., 1]) - torch.maximum(interval[..., 0], interval.roll(1, 1)[..., 0])).clamp_min(0)
            weights = overlap.softmax(-1)
        elif variant == 8:  # Sinkhorn 负载平衡路由的一步投影
            logits = (sa * sb).sum(-1)
            weights = logits.softmax(-1)
            weights = weights / weights.mean(0, keepdim=True).clamp_min(1e-6)
            weights = weights / weights.sum(-1, keepdim=True)
        elif variant == 9:  # 拓扑真实/平凡反事实
            response = delta_a.norm(dim=-1)
            weights = torch.cat((response[:, :1], response, ), 1).softmax(-1)
        else:  # 连续尺度 ODE 梯形积分
            velocity = torch.cat(
                (delta_a[:, :1], 0.5 * (delta_a[:, :1] + delta_a[:, 1:]), delta_a[:, 1:]),
                1,
            )
            weights = velocity.norm(dim=-1).neg().softmax(-1)
        za, zb = (weights.unsqueeze(-1) * sa).sum(1), (weights.unsqueeze(-1) * sb).sum(1)
        auxiliary = F.mse_loss(delta_a, delta_b) + 0.05 * weights.mean(0).var(unbiased=False)
        return za, zb, auxiliary

    def _subset_terms(self, views: Tensor) -> Tensor:
        a, b, c = views[:, 0], views[:, 1], views[:, 2]
        return torch.stack((a, b, c, (a + b) / 2, (a + c) / 2, (b + c) / 2, (a + b + c) / 3), 1)

    def _family4(self, a: dict[str, Tensor], b: dict[str, Tensor], variant: int) -> tuple[Tensor, Tensor, Tensor]:
        xa, xb = self._subset_terms(a["views"]), self._subset_terms(b["views"])
        if variant == 1:  # 子集格 Möbius
            xa = xa.clone(); xb = xb.clone()
            xa[:, 3:6] -= 0.5 * (xa[:, :3] + xa[:, :3].roll(-1, 1))
            xb[:, 3:6] -= 0.5 * (xb[:, :3] + xb[:, :3].roll(-1, 1))
            weights = torch.ones_like(self.subset_logits).softmax(0)
        elif variant == 2:  # Shapley × DPP 多样性
            utility = (xa * xb).sum(-1).mean(0)
            redundancy = torch.einsum("bkd,bjd->kj", xa, xa) / xa.size(0)
            weights = (utility - 0.1 * redundancy.abs().mean(-1)).softmax(0)
        elif variant == 3:  # 组合超图消息
            incidence = xa.new_tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]], dtype=xa.dtype)
            weights = incidence.sum(-1).softmax(0)
            xa = xa + torch.einsum("kv,bvd->bkd", incidence / incidence.sum(-1, keepdim=True), a["views"][:, :3])
            xb = xb + torch.einsum("kv,bvd->bkd", incidence / incidence.sum(-1, keepdim=True), b["views"][:, :3])
        elif variant == 4:  # 非交换顺序融合
            commutator_a = a["views"][:, 0] * a["views"][:, 1].roll(1, -1) - a["views"][:, 1] * a["views"][:, 0].roll(1, -1)
            commutator_b = b["views"][:, 0] * b["views"][:, 1].roll(1, -1) - b["views"][:, 1] * b["views"][:, 0].roll(1, -1)
            xa[:, 3] += commutator_a; xb[:, 3] += commutator_b
            weights = self.subset_logits.softmax(0)
        elif variant == 5:  # tensor-train 低秩可靠性代理
            left = xa.mean(-1); right = xa.var(-1, unbiased=False)
            weights = (left.mean(0) * right.mean(0)).softmax(0)
        elif variant == 6:  # 反事实遮视图路由
            full = xa[:, -1]
            effects = torch.stack([(full - xa[:, k]).norm(dim=-1) for k in range(7)], -1)
            weights = effects.mean(0).softmax(0)
        elif variant == 7:  # PID 原子不直接加权，而与 Hodge 槽位按类型耦合
            def pid_hodge_slots(views: Tensor) -> Tensor:
                node, edge, cycle = views[:, 0], views[:, 1], views[:, 2]
                base = views[:, :3]
                magnitude = base.abs().amin(1)
                redundant = magnitude * base.mean(1).sign()
                unique = base - redundant.unsqueeze(1)
                gradient = edge - node
                gradient_unit = F.normalize(gradient, dim=-1)
                cut_slot = (redundant * gradient_unit).sum(-1, keepdim=True) * gradient_unit
                harmonic = cycle - (cycle * gradient_unit).sum(-1, keepdim=True) * gradient_unit
                synergy = (node * edge + edge * cycle + cycle * node) / 3 - redundant
                synergy_harmonic = synergy - (
                    synergy * gradient_unit
                ).sum(-1, keepdim=True) * gradient_unit + harmonic
                semantic_residual = unique.mean(1) - cut_slot
                return torch.stack((
                    unique[:, 0], unique[:, 1], unique[:, 2], cut_slot,
                    synergy_harmonic, semantic_residual,
                    redundant + synergy_harmonic,
                ), 1)

            xa, xb = pid_hodge_slots(a["views"]), pid_hodge_slots(b["views"])
            slot_agreement = F.cosine_similarity(xa, xb, dim=-1).mean(0)
            weights = (self.subset_logits + slot_agreement).softmax(0)
        elif variant == 8:  # 可逆树融合
            scale = self.flow_scale.tanh()
            xa = xa * torch.exp(scale) + xa.roll(1, -1)
            xb = xb * torch.exp(scale) + xb.roll(1, -1)
            weights = self.subset_logits.softmax(0)
        elif variant == 9:  # 冲突质量作为第七组合增强
            conflict_a = a["views"][:, :3].var(1, unbiased=False)
            conflict_b = b["views"][:, :3].var(1, unbiased=False)
            xa[:, -1] += conflict_a; xb[:, -1] += conflict_b
            weights = self.subset_logits.softmax(0)
        else:  # 数据驱动组合课程
            confidence = (xa * xb).sum(-1).mean(0)
            unlocked = torch.sigmoid(5 * (confidence - confidence.mean()))
            weights = (self.subset_logits.softmax(0) * unlocked).div((self.subset_logits.softmax(0) * unlocked).sum())
        za, zb = (xa * weights[None, :, None]).sum(1), (xb * weights[None, :, None]).sum(1)
        auxiliary = 1 - F.cosine_similarity(za, zb).mean()
        return za, zb, auxiliary

    def _family5(self, a: dict[str, Tensor], b: dict[str, Tensor], variant: int) -> tuple[Tensor, Tensor, Tensor]:
        va, vb = a["views"], b["views"]
        evidence = self._view_evidence(va, vb)
        agreement, stability, diversity = evidence.unbind(-1)
        if variant == 1:  # Mondrian 共形代理
            reliability = torch.sigmoid(_standardize(agreement + stability, 0))
        elif variant == 2:  # 证据冲突瀑布
            conflict = (va - va.roll(1, 1)).abs().mean(-1)
            reliability = torch.cumprod(torch.sigmoid(stability - conflict), dim=1)
        elif variant == 3:  # PAC-Bayes 风险/复杂度预算
            complexity = torch.stack([head.weight.pow(2).mean() for head in self.view_heads])
            reliability = torch.exp(-(1 - agreement) - complexity.unsqueeze(0))
        elif variant == 4:  # jackknife 稳定性
            influence = (va - va.roll(1, 0)).pow(2).mean(-1)
            reliability = 1 / (1 + influence)
        elif variant == 5:  # 守恒流
            mass = stability.softmax(-1)
            reliability = mass / mass.sum(-1, keepdim=True)
        elif variant == 6:  # 局部 Lipschitz
            lipschitz = (va - vb).norm(dim=-1) / (a["scales"][:, -1:].norm(dim=-1) + 1e-4)
            reliability = diversity / (1 + lipschitz)
        elif variant == 7:  # MI 置信下界代理
            lower = agreement - 1.96 * (1 - stability) / math.sqrt(max(va.size(0), 1))
            reliability = torch.sigmoid(lower)
        elif variant == 8:  # Kalman 可靠性
            observation = stability.detach().mean(0)
            if self.training:
                self.reliability_ema.mul_(0.9).add_(0.1 * observation)
            reliability = self.reliability_ema.unsqueeze(0).expand_as(stability)
        elif variant == 9:  # 独有性 × 稳定性
            recoverable = F.cosine_similarity(va, va.mean(1, keepdim=True), dim=-1).abs()
            reliability = (1 - recoverable) * stability
        else:  # Shapley 交互图
            interaction = torch.einsum("bvd,bwd->bvw", va, va).mean(-1)
            reliability = torch.sigmoid(agreement + interaction)
        reliability = reliability / reliability.sum(-1, keepdim=True).clamp_min(1e-6)
        za, zb = (reliability.unsqueeze(-1) * va).sum(1), (reliability.unsqueeze(-1) * vb).sum(1)
        auxiliary = -(reliability * stability).sum(-1).mean() + 0.02 * reliability.mean(0).var(unbiased=False)
        return za, zb, auxiliary

    def _family6(self, a: dict[str, Tensor], b: dict[str, Tensor], variant: int) -> tuple[Tensor, Tensor, Tensor]:
        # 每张图的三个尺度充当三种增强观测，生成一个综合正样本。
        views = b["scales"]
        anchor = a["graph"]
        distance = (views - anchor.unsqueeze(1)).norm(dim=-1)
        if variant == 1:  # Wasserstein 重心代理
            weights = (-distance).softmax(-1)
            positive = (weights.unsqueeze(-1) * views).sum(1)
        elif variant == 2:  # Fréchet 均值的一步切空间更新
            center = F.normalize(views.mean(1), dim=-1)
            positive = F.normalize(center + (views - center.unsqueeze(1)).mean(1), dim=-1)
        elif variant == 3:  # DPP 多样子集 + set attention
            diversity = 1 - torch.einsum("bvd,bwd->bvw", views, views).abs().mean(-1)
            weights = (distance.neg() + diversity).softmax(-1)
            positive = (weights.unsqueeze(-1) * views).sum(1)
        elif variant == 4:  # 条件扩散后验一步去噪
            noise = views - views.mean(1, keepdim=True)
            positive = views.mean(1) - 0.25 * self.reconstructor(noise.mean(1))
        elif variant == 5:  # Weiszfeld 几何中位数一步
            center = views.mean(1, keepdim=True)
            weights = 1 / (views - center).norm(dim=-1).clamp_min(1e-3)
            positive = (weights.unsqueeze(-1) * views).sum(1) / weights.sum(-1, keepdim=True)
        elif variant == 6:  # 正样本超图池化
            incidence = torch.einsum("bvd,bwd->bvw", views, views).softmax(-1)
            positive = torch.einsum("bvw,bwd->bvd", incidence, views).mean(1)
        elif variant == 7:  # capsule 动态路由
            logits = torch.zeros_like(distance)
            for _ in range(2):
                weights = logits.softmax(-1)
                positive = F.normalize((weights.unsqueeze(-1) * views).sum(1), dim=-1)
                logits = logits + (views * positive.unsqueeze(1)).sum(-1)
        elif variant == 8:  # 最小充分信息瓶颈
            positive = self.bottleneck(views.mean(1))
        elif variant == 9:  # credal 意见交集
            lower = views.amin(1); upper = views.amax(1)
            positive = torch.where(lower.sign() == upper.sign(), 0.5 * (lower + upper), torch.zeros_like(lower))
        else:  # 增强轨迹二次样条零点外推
            positive = 3 * views[:, 0] - 3 * views[:, 1] + views[:, 2]
        auxiliary = 1 - F.cosine_similarity(anchor, positive).mean()
        return anchor, positive, auxiliary

    def _family7(self, a: dict[str, Tensor], b: dict[str, Tensor], variant: int) -> tuple[Tensor, Tensor, Tensor]:
        va, vb = a["views"], b["views"]
        evidence = self._view_evidence(va, vb)
        agreement, stability, diversity = evidence.unbind(-1)
        if variant == 1:  # Beta 后验均值
            clean = (1 + stability) / (2 + stability + (1 - agreement).clamp_min(0))
        elif variant == 2:  # 2/3 异构陪审
            votes = torch.stack((agreement > 0, stability > 0.5, diversity > diversity.median(0).values), -1)
            clean = (votes.float().sum(-1) >= 2).float().detach() + 0.05
        elif variant == 3:  # 时间反转记忆代理
            observation = stability.detach().mean(0)
            trend = observation - self.clean_ema
            if self.training:
                self.clean_ema.mul_(0.9).add_(0.1 * observation)
            clean = torch.sigmoid(stability + trend)
        elif variant == 4:  # 反事实原型/拓扑双影响
            prototype_effect = (va - va.mean(0, keepdim=True)).norm(dim=-1)
            topology_effect = (a["scales"][:, -1] - a["scales"][:, 0]).norm(dim=-1, keepdim=True)
            clean = torch.exp(-prototype_effect * topology_effect)
        elif variant == 5:  # 共形三态
            score = _standardize(1 - stability, 0)
            clean = torch.where(score < -0.5, torch.ones_like(score), torch.where(score > 0.5, 0.1 * torch.ones_like(score), 0.5 * torch.ones_like(score)))
        elif variant == 6:  # 属性/拓扑去污次序的非交换子定位跨视图污染
            def attribute_cleaner(values: Tensor) -> Tensor:
                center = values.median(dim=1, keepdim=True).values
                residual = values - center
                threshold = 0.25 * residual.abs().mean(-1, keepdim=True)
                shrunk = residual.sign() * F.relu(residual.abs() - threshold)
                return center + shrunk

            def topology_cleaner(values: Tensor, reference: Tensor) -> Tensor:
                cycle_reference = reference[:, 2:3]
                support = torch.sigmoid(
                    F.cosine_similarity(values, cycle_reference, dim=-1)
                ).unsqueeze(-1)
                return support * values + (1 - support) * cycle_reference

            attr_after_topo_a = attribute_cleaner(topology_cleaner(va, va))
            topo_after_attr_a = topology_cleaner(attribute_cleaner(va), va)
            attr_after_topo_b = attribute_cleaner(topology_cleaner(vb, vb))
            topo_after_attr_b = topology_cleaner(attribute_cleaner(vb), vb)
            commutator = 0.5 * (
                (attr_after_topo_a - topo_after_attr_a).abs().mean(-1)
                + (attr_after_topo_b - topo_after_attr_b).abs().mean(-1)
            )
            va = 0.5 * (attr_after_topo_a + topo_after_attr_a)
            vb = 0.5 * (attr_after_topo_b + topo_after_attr_b)
            clean = torch.exp(-_standardize(commutator, 0).abs())
        elif variant == 7:  # 多尺度相位连续
            phase_jump = (a["scales"][:, 2] - 2 * a["scales"][:, 1] + a["scales"][:, 0]).abs().mean(-1, keepdim=True)
            clean = torch.exp(-phase_jump).expand_as(stability)
        elif variant == 8:  # 双库迟滞
            high = stability > 0.65; low = stability < 0.35
            state = self.clean_ema.unsqueeze(0).expand_as(stability)
            clean = torch.where(high, torch.ones_like(state), torch.where(low, torch.zeros_like(state), state)) + 0.05
        elif variant == 9:  # MDL 额外码长
            code_length = -torch.log2(stability.clamp_min(1e-5)) + diversity
            clean = torch.exp(-code_length)
        else:  # 清洗器—增强器博弈代理
            clean = torch.sigmoid(agreement + stability - diversity)
            clean = clean * (1 + 0.1 * (1 - clean).detach())
        weights = clean / clean.sum(-1, keepdim=True).clamp_min(1e-6)
        za, zb = (weights.unsqueeze(-1) * va).sum(1), (weights.unsqueeze(-1) * vb).sum(1)
        auxiliary = -(weights * agreement).sum(-1).mean()
        return za, zb, auxiliary

    def _family8(self, a: dict[str, Tensor], b: dict[str, Tensor], variant: int) -> tuple[Tensor, Tensor, Tensor]:
        true_a, true_b = a["graph"], b["graph"]
        fake_b = b["views"][:, :3].mean(1).roll(1, 0)
        true_score = F.cosine_similarity(true_a, true_b)
        fake_score = F.cosine_similarity(true_a, fake_b)
        if variant == 1:  # 双向可信关系蒸馏
            confidence = torch.minimum(torch.sigmoid(true_score), torch.sigmoid(fake_score))
            auxiliary = F.mse_loss(true_score, fake_score.detach()) + F.mse_loss(fake_score, true_score.detach())
        elif variant == 2:  # 语义/拓扑双证据路径的离散同伦代理
            semantic_path = (
                F.normalize(true_a, dim=-1) @ F.normalize(fake_b, dim=-1).t() / 0.2
            ).softmax(-1)
            topology_path = (
                F.normalize(a["views"][:, 2], dim=-1)
                @ F.normalize(b["views"][:, 2].roll(1, 0), dim=-1).t() / 0.2
            ).softmax(-1)
            midpoint = 0.5 * (semantic_path + topology_path)
            filling_area = 0.5 * (
                (semantic_path * (semantic_path.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(-1)
                + (topology_path * (topology_path.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(-1)
            )
            closed_mass = (semantic_path @ topology_path.t()).diagonal()
            confidence = (closed_mass * torch.exp(-filling_area)).clamp(0.05, 0.95)
            hard_negative = (semantic_path * (1 - topology_path)).sum(-1)
            auxiliary = -confidence.clamp_min(1e-6).log().mean() + 0.1 * hard_negative.mean()
        elif variant == 3:  # 因果交换干预
            intervention = 0.5 * (fake_b + true_b.roll(1, 0))
            effect = (F.cosine_similarity(true_a, intervention) - true_score).abs()
            confidence = torch.exp(-effect); auxiliary = effect.mean()
        elif variant == 4:  # 双 Beta product-of-experts
            p = (1 + torch.sigmoid(true_score)) / 3
            q = (1 + torch.sigmoid(fake_score)) / 3
            confidence = p * q / (p * q + (1 - p) * (1 - q)).clamp_min(1e-6)
            auxiliary = -(confidence * (true_score + fake_score)).mean()
        elif variant == 5:  # 生成—鉴别互惠
            confidence = torch.sigmoid(true_score - fake_score.detach())
            entropy_floor = -(confidence * confidence.clamp_min(1e-6).log() + (1-confidence) * (1-confidence).clamp_min(1e-6).log())
            auxiliary = -fake_score.mean() + F.relu(0.3 - entropy_floor).mean()
        elif variant == 6:  # 顺逆等变
            inverse = self.reconstructor(fake_b)
            confidence = torch.exp(-(inverse - true_a).pow(2).mean(-1))
            auxiliary = F.mse_loss(inverse, true_a)
        elif variant == 7:  # 课程反哺
            confidence = torch.sigmoid(5 * (true_score.detach() - fake_score.detach()))
            auxiliary = (confidence * (true_score - fake_score).abs()).mean()
        elif variant == 8:  # 对偶 OT 势一致代理
            cost = torch.cdist(F.normalize(true_a, dim=-1), F.normalize(fake_b, dim=-1))
            forward_potential = -torch.logsumexp(-cost, dim=1)
            backward_potential = -torch.logsumexp(-cost, dim=0)
            confidence = torch.sigmoid(-(forward_potential - backward_potential).abs())
            auxiliary = F.mse_loss(forward_potential, backward_potential)
        elif variant == 9:  # 四值逻辑
            support, oppose = true_score > 0, fake_score < 0
            confidence = torch.where(support & ~oppose, torch.ones_like(true_score), torch.where(~support & oppose, torch.zeros_like(true_score), 0.5 * torch.ones_like(true_score)))
            contradiction = support & oppose
            auxiliary = (contradiction.float() * (true_score * fake_score).abs()).mean()
        else:  # 真正样本约束下的表示空间梯度投影代理
            true_direction = true_b - true_a
            fake_direction = fake_b - true_a
            conflict = (true_direction * fake_direction).sum(-1, keepdim=True) < 0
            projected = fake_direction - (fake_direction * true_direction).sum(-1, keepdim=True) / true_direction.pow(2).sum(-1, keepdim=True).clamp_min(1e-6) * true_direction
            fake_b = true_a + torch.where(conflict, projected, fake_direction)
            confidence = torch.ones_like(true_score); auxiliary = (conflict.float()).mean()
        positive = confidence.unsqueeze(-1) * true_b + (1 - confidence.unsqueeze(-1)) * fake_b
        return true_a, positive, auxiliary

    def _family9(self, a: dict[str, Tensor], b: dict[str, Tensor], variant: int) -> tuple[Tensor, Tensor, Tensor]:
        va, vb = a["views"], b["views"]
        node_a, edge_a, cycle_a, graph_a = va.unbind(1)
        node_b, edge_b, cycle_b, graph_b = vb.unbind(1)
        levels_a, levels_b = va, vb
        if variant == 1:  # boundary∘boundary=0
            boundary_a = cycle_a - edge_a + node_a
            boundary_b = cycle_b - edge_b + node_b
            auxiliary = F.mse_loss(boundary_a, torch.zeros_like(boundary_a)) + F.mse_loss(boundary_b, torch.zeros_like(boundary_b))
        elif variant == 2:  # 跨层 OT 循环
            transport = torch.einsum("bvd,bwd->bvw", levels_a, levels_b).softmax(-1)
            recovered = torch.einsum("bvw,bwd->bvd", transport, levels_b)
            auxiliary = F.mse_loss(recovered, levels_a)
        elif variant == 3:  # PID 槽
            redundant = levels_a.amin(1)
            unique = levels_a - redundant.unsqueeze(1)
            auxiliary = unique.mean(1).pow(2).mean() - F.cosine_similarity(redundant, levels_b.amin(1)).mean()
        elif variant == 4:  # 边翻译器
            auxiliary = F.mse_loss(edge_a, 0.5 * (node_a + graph_a)) + F.mse_loss(edge_b, 0.5 * (node_b + graph_b))
        elif variant == 5:  # 梯度—旋度—调和能量
            gradient = edge_a - node_a; curl = cycle_a - edge_a; harmonic = graph_a - cycle_a
            auxiliary = (gradient.sum(-1) + curl.sum(-1) + harmonic.sum(-1)).pow(2).mean()
        elif variant == 6:  # 可靠层单向蒸馏
            reliability = self._view_evidence(levels_a, levels_b)[..., 1]
            teacher = (reliability.softmax(-1).unsqueeze(-1) * levels_a).sum(1).detach()
            auxiliary = F.mse_loss(levels_b, teacher.unsqueeze(1).expand_as(levels_b))
        elif variant == 7:  # 阶层干预 ANOVA
            main = levels_a.mean(1); interaction = (node_a * edge_a + edge_a * cycle_a + cycle_a * graph_a) / 3
            auxiliary = 1 - F.cosine_similarity(main + interaction, levels_b.mean(1)).mean()
        elif variant == 8:  # 原型单纯形重心坐标
            prototypes = F.normalize(self.semantic_prototypes[: min(4, self.semantic_prototypes.size(0))], dim=-1)
            coordinate = torch.einsum("bvd,kd->bvk", levels_a, prototypes).softmax(-1)
            auxiliary = F.mse_loss(coordinate, torch.einsum("bvd,kd->bvk", levels_b, prototypes).softmax(-1))
        elif variant == 9:  # 遮粒度互预测残差
            predicted = (levels_a.sum(1, keepdim=True) - levels_a) / 3
            residual = levels_a - predicted
            auxiliary = _symmetric_nce(residual.mean(1), (levels_b - (levels_b.sum(1, keepdim=True) - levels_b) / 3).mean(1))
        else:  # Koopman 跨层动力学
            mode_a = torch.fft.rfft(levels_a, dim=1).abs(); mode_b = torch.fft.rfft(levels_b, dim=1).abs()
            auxiliary = F.mse_loss(mode_a, mode_b)
        return graph_a + levels_a[:, :3].mean(1), graph_b + levels_b[:, :3].mean(1), auxiliary

    def _family10(self, a: dict[str, Tensor], b: dict[str, Tensor], variant: int) -> tuple[Tensor, Tensor, Tensor]:
        za, zb = a["graph"], b["graph"]
        if variant == 1:  # carré-du-champ 对角度量
            metric = F.softplus(self.metric[0])
            pa, pb = za * metric, zb * metric
        elif variant == 2:  # 曲率跃迁写成尺度 1-chain，仅压缩调和跃迁签名
            def curvature_transition(bundle: dict[str, Tensor]) -> Tensor:
                scales = bundle["scales"]
                transitions = scales[:, 1:] - scales[:, :-1]
                exact = transitions.mean(1)
                harmonic = 0.5 * (transitions[:, 1] - transitions[:, 0])
                curvature_jump = (
                    bundle["views"][:, 2] - bundle["views"][:, 1]
                ).norm(dim=-1, keepdim=True)
                harmonic_signature = harmonic * torch.sigmoid(curvature_jump)
                semantic = bundle["graph"]
                exact_unit = F.normalize(exact, dim=-1)
                semantic_residual = semantic - (
                    semantic * exact_unit
                ).sum(-1, keepdim=True) * exact_unit
                return semantic_residual + self.bottleneck(harmonic_signature)

            pa, pb = curvature_transition(a), curvature_transition(b)
        elif variant == 3:  # 扩散坐标一步
            kernel = torch.exp(-torch.cdist(za, za).pow(2) / za.size(-1)).softmax(-1)
            pa, pb = kernel @ za, kernel @ zb
        elif variant == 4:  # 拓扑 strata 条件切空间
            strata = a["views"][:, 2].mean(-1, keepdim=True).sign()
            pa, pb = self.bottleneck(za) * strata, self.bottleneck(zb) * strata
        elif variant == 5:  # 样本关系 Hodge 去旋度代理
            flow = za.roll(-1, 0) - za
            curl = flow + flow.roll(1, 0)
            pa, pb = za - 0.5 * curl, zb - 0.5 * curl
        elif variant == 6:  # 半松弛 GW 结构先验
            structure = torch.cdist(a["views"][:, 0], a["views"][:, 2]).detach()
            representation = torch.cdist(za, za)
            scale = (structure.mean() / representation.mean().clamp_min(1e-6)).detach()
            pa, pb = self.bottleneck(za) * scale, self.bottleneck(zb) * scale
        elif variant == 7:  # 热测地/分类测地 Pareto
            heat = a["scales"].mean(1); task = self.bottleneck(za)
            angle = F.cosine_similarity(heat, task).unsqueeze(-1)
            pa, pb = angle * heat + (1-angle) * task, angle * b["scales"].mean(1) + (1-angle) * self.bottleneck(zb)
        elif variant == 8:  # 可逆仿射流
            scale = self.flow_scale.tanh()
            pa, pb = za * torch.exp(scale), zb * torch.exp(scale)
        elif variant == 9:  # 固定散射代理 + 正交可学习残差
            scattering_a = a["scales"].mean(1).detach(); scattering_b = b["scales"].mean(1).detach()
            residual_a = za - (za * scattering_a).sum(-1, keepdim=True) * scattering_a
            residual_b = zb - (zb * scattering_b).sum(-1, keepdim=True) * scattering_b
            pa, pb = scattering_a + residual_a, scattering_b + residual_b
        else:  # 持久路径稀疏字典代理
            path_a = a["scales"][:, 0] - 2*a["scales"][:, 1] + a["scales"][:, 2]
            path_b = b["scales"][:, 0] - 2*b["scales"][:, 1] + b["scales"][:, 2]
            pa, pb = self.bottleneck(path_a), self.bottleneck(path_b)
        auxiliary = F.mse_loss(torch.cdist(F.normalize(pa, dim=-1), F.normalize(pa, dim=-1)), torch.cdist(F.normalize(a["views"][:, 0], dim=-1), F.normalize(a["views"][:, 2], dim=-1)).detach())
        return pa, pb, auxiliary

    def _family11(self, a: dict[str, Tensor], b: dict[str, Tensor], variant: int) -> tuple[Tensor, Tensor, Tensor]:
        za, zb = a["graph"], b["graph"]
        probability = self.mask_logits.sigmoid().unsqueeze(0).expand_as(za)
        cross_reconstruction = za.new_zeros(())
        if variant == 1:  # 可靠×可恢复反向掩码
            reliability = F.cosine_similarity(za, zb).unsqueeze(-1)
            probability = probability * torch.sigmoid(reliability)
        elif variant == 2:  # 点边环守恒预算
            conservation = a["views"][:, 0] - a["views"][:, 1] + a["views"][:, 2]
            probability = probability * torch.sigmoid(-conservation.abs())
        elif variant == 3:  # 共形难度课程
            score = _standardize((za - zb).abs(), 0)
            probability = probability * torch.sigmoid(-score)
        elif variant == 4:  # 最小充分/必要双掩码
            sufficient = torch.sigmoid(za * zb); necessary = torch.sigmoid((za-zb).abs())
            probability = probability * (sufficient + necessary) / 2
        elif variant == 5:  # 连续空间带宽与谱频带互相预测后再联合掩码
            spatial_bandwidth = torch.sigmoid(
                (a["views"][:, 1] - a["views"][:, 2]).abs()
            )
            spectrum = torch.fft.rfft(za, dim=-1).abs()
            spectral_band = torch.sigmoid(spectrum - spectrum.mean(-1, keepdim=True))
            spectral_signal = torch.fft.irfft(spectral_band, n=za.size(-1), dim=-1)
            probability = probability * torch.sigmoid(spatial_bandwidth + spectral_signal)
            cross_reconstruction = (
                F.mse_loss(self.reconstructor(spatial_bandwidth), spectral_signal.detach())
                + F.mse_loss(self.reconstructor(spectral_signal), spatial_bandwidth.detach())
            )
        elif variant == 6:  # Shapley × DPP 代理
            marginal = (za * zb).abs(); diversity = (za - za.roll(1, 0)).abs()
            probability = probability * torch.sigmoid(marginal + diversity)
        elif variant == 7:  # Kalman 可恢复性记忆
            state = self.reliability_ema.mean().detach()
            probability = probability * state
        elif variant == 8:  # 观测/存在双解码
            existence = torch.sigmoid(self.reconstructor(za))
            probability = probability * existence
        elif variant == 9:  # Möbius 高阶交互
            mobius = a["views"][:, 2] - a["views"][:, 1] + a["views"][:, 0]
            probability = probability * torch.sigmoid(mobius)
        else:  # 相邻尺度双侧证明
            proof = F.cosine_similarity(a["scales"][:, 0], a["scales"][:, 2]).unsqueeze(-1)
            probability = probability * torch.sigmoid(proof)
        if self.training:
            hard = (torch.rand_like(probability) > probability).to(za.dtype)
            keep = hard + (1 - probability) - (1 - probability).detach()
        else:
            keep = 1 - probability
        masked = zb * keep
        reconstructed = self.reconstructor(masked)
        auxiliary = (
            F.mse_loss(reconstructed, zb)
            + cross_reconstruction
            + 0.01 * (probability.mean() - 0.3).pow(2)
        )
        return za, reconstructed, auxiliary

    def idea_branch_terms(
        self,
        first: dict[str, Tensor],
        second: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        """返回单个 idea 的对比分支、辅助分支及诊断，不重复计算主损失。"""

        family_method = getattr(self, f"_family{self.spec.family}")
        transformed_first, transformed_second, auxiliary = family_method(
            first, second, self.spec.variant
        )
        parameter_a_gain, parameter_b_resolution, parameter_c_gain = self.idea_parameter_multipliers
        # A、B、C 只进入对应 idea 分支。A 控制新表示残差剂量；B 控制该
        # idea 关系分布的分辨率；C 控制该 idea 专属辅助约束，不改主干配置。
        if math.isclose(parameter_a_gain, 0.0, abs_tol=1e-12):
            # 与 C=0 同理，A=0 必须完全旁路新增表示，不能让 ``0 * NaN``
            # 从已关闭的分支传播到主干对比损失。
            transformed_first = first["graph"]
            transformed_second = second["graph"]
        else:
            transformed_first = (
                first["graph"]
                + self.operator_mix * parameter_a_gain * (transformed_first - first["graph"])
            )
            transformed_second = (
                second["graph"]
                + self.operator_mix * parameter_a_gain * (transformed_second - second["graph"])
            )
        resolved_temperature = self.idea_temperature / math.sqrt(max(parameter_b_resolution, 1e-3))
        idea_contrast = _symmetric_nce(
            transformed_first, transformed_second, temperature=resolved_temperature
        )
        # 0 是专属子组件消融。不能写成 ``0 * auxiliary``，因为辅助分支若
        # 产生 Inf/NaN，IEEE 浮点运算仍会把乘积变成 NaN，导致本应关闭的
        # 子组件污染总损失。
        auxiliary_term = (
            auxiliary.new_zeros(())
            if math.isclose(parameter_c_gain, 0.0, abs_tol=1e-12)
            else parameter_c_gain * auxiliary
        )
        diagnostics = {
            "idea_contrast": float(idea_contrast.detach()),
            "auxiliary": float(auxiliary.detach()),
            "idea_loss_weight": self.auxiliary_weight,
            "idea_temperature": self.idea_temperature,
            "operator_mix": self.operator_mix,
            "resolved_idea_temperature": resolved_temperature,
            "parameter_a_gain": parameter_a_gain,
            "parameter_b_resolution": parameter_b_resolution,
            "parameter_c_gain": parameter_c_gain,
        }
        return idea_contrast, auxiliary_term, diagnostics

    def idea_loss(self, first_data: Data, second_data: Data) -> tuple[Tensor, dict[str, float]]:
        first, second = self._bundle(first_data), self._bundle(second_data)
        idea_contrast, auxiliary_term, diagnostics = self.idea_branch_terms(first, second)
        base = super().contrastive_loss(first["graph"], second["graph"], 0.2)
        total = base + self.auxiliary_weight * (idea_contrast + auxiliary_term)
        diagnostics = {"base": float(base.detach()), **diagnostics}
        return total, diagnostics


def build_research_idea_model(
    idea_id: str,
    in_dim: int,
    hidden_dim: int,
    layers: int,
    num_classes: int,
    auxiliary_weight: float = 0.25,
    idea_temperature: float = 0.2,
    operator_mix: float = 1.0,
    idea_parameters: dict[str, float] | None = None,
) -> ResearchIdeaBalanceGCL:
    return ResearchIdeaBalanceGCL(
        idea_id, in_dim, hidden_dim, layers, num_classes,
        auxiliary_weight, idea_temperature, operator_mix, idea_parameters,
    )
