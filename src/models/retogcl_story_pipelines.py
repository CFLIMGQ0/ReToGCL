"""围绕 ReToGCL 三个创新模块构造的八套闭环 Pipeline。

三阶段分别为：

1. D7-I10（M1）产生增强视图与增强轨迹；
2. D7-I01（M4）估计样本×视图洁净度；
3. D10-I04（M7）生成多尺度拓扑投影。

每套策略都保留原 ReToGCL 主路径，并把三阶段链接路径作为可拒绝残差。候选
路径中的随机正样本不会推进全局随机数状态，保证下一批增强序列与基线路径一致。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data

from src.models.retogcl_pipeline_strategies import ReToGCLPipelineStrategy


@dataclass(frozen=True)
class StoryPipeline:
    pipeline_id: str
    name: str
    story: str
    mode: str
    max_mix: float
    auxiliary_weight: float = 0.003


STORY_PIPELINES: dict[str, StoryPipeline] = {
    "S01": StoryPipeline(
        "S01", "三证据否决链",
        "增强保真、视图洁净和拓扑稳定任一不可靠时，链接路径自动退回原路径。",
        "tri_veto", 0.18,
    ),
    "S02": StoryPipeline(
        "S02", "反事实收益回路",
        "把链接路径视为一次干预，仅在其对比能量优于原路径时接受该干预。",
        "counterfactual_accept", 0.24,
    ),
    "S03": StoryPipeline(
        "S03", "跨视图可恢复闭环",
        "拓扑投影必须能够恢复增强轨迹和洁净表示，可恢复性同时控制残差强度。",
        "recoverability_cycle", 0.18, auxiliary_weight=0.005,
    ),
    "S04": StoryPipeline(
        "S04", "拓扑慢反馈增强",
        "当前批次的拓扑可信度形成慢变量，反向调节下一批次的增强强度。",
        "topology_feedback", 0.20,
    ),
    "S05": StoryPipeline(
        "S05", "洁净边界救援",
        "原型边界样本只有在洁净且拓扑一致时才获得更强的链接修正。",
        "boundary_rescue", 0.20,
    ),
    "S06": StoryPipeline(
        "S06", "多尺度可信课程",
        "由浅到深逐步开放拓扑尺度，并用已开放尺度的一致性控制连接。",
        "scale_curriculum", 0.22,
    ),
    "S07": StoryPipeline(
        "S07", "充分性—多样性帕累托桥",
        "增强既不能破坏语义也不能退化为恒等映射，仅在二者平衡区间连接。",
        "sufficiency_diversity", 0.16, auxiliary_weight=0.002,
    ),
    "S08": StoryPipeline(
        "S08", "点边环互惠协作",
        "点、边、环的增强保真度与洁净权重、拓扑尺度逐级对齐后共同路由。",
        "node_edge_cycle_reciprocity", 0.22,
    ),
}


class ReToGCLStoryPipeline(ReToGCLPipelineStrategy):
    """原 ReToGCL 主路径与八类故事化闭环残差的统一实现。"""

    protocol = "retogcl_story_pipelines_v1"

    def __init__(
        self,
        pipeline_id: str,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
    ) -> None:
        if pipeline_id not in STORY_PIPELINES:
            raise ValueError(f"未知故事 Pipeline：{pipeline_id}")
        # 复用 P05 的稳定链接算子，但不复用 P05 的门控和损失组合。
        super().__init__(
            "P05", in_dim, hidden_dim, layers, num_classes, idea_parameters
        )
        self.story_pipeline = STORY_PIPELINES[pipeline_id]

        # 新模块初始化后恢复 RNG，避免改变基线主干和后续增强的随机序列。
        cpu_rng_state = torch.random.get_rng_state()
        self.trace_decoder = nn.Sequential(
            nn.Linear(self.output_dim, max(16, self.output_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(16, self.output_dim // 2), 5),
        )
        self.clean_decoder = nn.Linear(self.output_dim, self.output_dim)
        self.reciprocal_gate = nn.Sequential(
            nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, 1)
        )
        with torch.no_grad():
            nn.init.eye_(self.clean_decoder.weight)
            nn.init.zeros_(self.clean_decoder.bias)
            nn.init.zeros_(self.reciprocal_gate[-1].weight)
            self.reciprocal_gate[-1].bias.fill_(-1.0)
        torch.random.set_rng_state(cpu_rng_state)
        self.register_buffer("topology_trust_ema", torch.tensor(0.75))

    def prepare_views(self, batch: Data) -> tuple[Data, Data]:
        first, second = super().prepare_views(batch)
        if self.story_pipeline.mode != "topology_feedback":
            return first, second

        # 上一批拓扑可信度只调节属性/节点扰动幅度；边子图仍保持离散结构。
        strength = float((0.80 + 0.20 * self.topology_trust_ema).clamp(0.80, 1.0))
        first.x = batch.x + strength * (first.x - batch.x)
        second.x = batch.x + strength * (second.x - batch.x)
        first.augmentation_trace = self._augmentation_trace(batch, first)
        second.augmentation_trace = self._augmentation_trace(batch, second)
        return first, second

    def _faithful_base_path(
        self,
        first_graph: Tensor,
        second_graph: Tensor,
        first_clean_bundle: dict[str, Tensor],
        second_clean_bundle: dict[str, Tensor],
        first_topology_bundle: dict[str, Tensor],
        second_topology_bundle: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """执行原三模块模型的完整顺序，包括其被 M7 覆盖的正样本构造。"""

        routed_first, _, clean_auxiliary = self.second._module_transform(
            first_clean_bundle, second_clean_bundle
        )
        routing = self.first.semantic_router(
            routed_first, self.first.semantic_prototypes
        )
        projected_second = F.normalize(self.first.projector(second_graph), dim=-1)
        # 原实现会构造该正样本，随后 M7 用 second_graph 覆盖它；保留调用是为了
        # 让每批随机数消耗与原 ReToGCL 完全一致。
        self.first.positive_builder(
            first_graph, second_graph, projected_second, routing
        )
        hard = self.first.negative_builder(first_graph, routing)
        anchor, positive, topology_auxiliary = self.third._module_transform(
            self._replace_graph(first_topology_bundle, first_graph),
            self._replace_graph(second_topology_bundle, second_graph),
        )
        auxiliary = (
            self.second.auxiliary_weight
            * self.second._parameter_controls()[2]
            * clean_auxiliary
            + self.third.auxiliary_weight
            * self.third._parameter_controls()[2]
            * topology_auxiliary
        )
        return anchor, positive, hard, auxiliary

    @staticmethod
    def _rng_state(reference: Tensor) -> tuple[Tensor, Tensor | None]:
        cpu_state = torch.random.get_rng_state()
        cuda_state = (
            torch.cuda.get_rng_state(reference.device)
            if reference.is_cuda else None
        )
        return cpu_state, cuda_state

    @staticmethod
    def _restore_rng(reference: Tensor, state: tuple[Tensor, Tensor | None]) -> None:
        cpu_state, cuda_state = state
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, reference.device)

    @staticmethod
    def _topology_agreement(
        first_bundle: dict[str, Tensor], second_bundle: dict[str, Tensor]
    ) -> Tensor:
        first = F.normalize(first_bundle["scales"], dim=-1)
        second = F.normalize(second_bundle["scales"], dim=-1)
        return ((first * second).sum(-1) + 1).mul(0.5).clamp(0, 1)

    @staticmethod
    def _contrastive_utility(anchor: Tensor, positive: Tensor, hard: Tensor) -> Tensor:
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        hard = F.normalize(hard, dim=-1)
        positive_score = (anchor * positive).sum(-1)
        hard_score = torch.einsum("bd,bkd->bk", anchor, hard).amax(-1)
        return positive_score - hard_score

    def _routing_entropy(self, representation: Tensor) -> Tensor:
        prototypes = F.normalize(self.first.semantic_prototypes, dim=-1)
        logits = F.normalize(representation, dim=-1) @ prototypes.t()
        probabilities = logits.softmax(-1)
        denominator = math.log(max(probabilities.size(-1), 2))
        return -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(-1) / denominator

    def _story_mix(
        self,
        base_anchor: Tensor,
        base_positive: Tensor,
        base_hard: Tensor,
        linked_anchor: Tensor,
        linked_positive: Tensor,
        linked_hard: Tensor,
        first_clean: Tensor,
        second_clean: Tensor,
        weights: Tensor,
        confidence: Tensor,
        agreement: Tensor,
        first_trace: Tensor,
        second_trace: Tensor,
        topology_agreement: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        spec = self.story_pipeline
        trace = 0.5 * (first_trace + second_trace)
        retention = trace[:, 3].clamp(0, 1)
        diversity = trace[:, 4].clamp(0, 1)
        view_trust = ((agreement.mean(-1) + 1) * 0.5).clamp(0, 1)
        topology_trust = topology_agreement.mean(-1).clamp(0, 1)
        topology_floor = topology_agreement.amin(-1).clamp(0, 1)
        auxiliary = confidence.new_zeros(())

        if spec.mode == "tri_veto":
            consensus = torch.stack(
                (confidence, retention, view_trust, topology_floor), -1
            ).amin(-1)
            accept = torch.sigmoid((consensus - 0.68) / 0.06)
            mix = spec.max_mix * confidence * accept

        elif spec.mode == "counterfactual_accept":
            base_utility = self._contrastive_utility(
                base_anchor, base_positive, base_hard
            )
            linked_utility = self._contrastive_utility(
                linked_anchor, linked_positive, linked_hard
            )
            advantage = linked_utility - base_utility
            accept = torch.sigmoid((advantage - 0.01) / 0.08)
            mix = spec.max_mix * confidence * topology_trust * accept

        elif spec.mode == "recoverability_cycle":
            linked_center = F.normalize(
                0.5 * (linked_anchor + linked_positive), dim=-1
            )
            trace_target = trace.detach()
            trace_prediction = torch.sigmoid(self.trace_decoder(linked_center))
            trace_error = (trace_prediction - trace_target).pow(2).mean(-1)
            clean_target = F.normalize(
                0.5 * (first_clean + second_clean), dim=-1
            ).detach()
            clean_prediction = F.normalize(
                self.clean_decoder(linked_center), dim=-1
            )
            clean_error = (clean_prediction - clean_target).pow(2).mean(-1)
            reconstruction_error = trace_error + clean_error
            recoverability = torch.exp(-4 * reconstruction_error).clamp(0, 1)
            mix = (
                spec.max_mix * confidence * topology_trust * recoverability
            )
            auxiliary = reconstruction_error.mean()

        elif spec.mode == "topology_feedback":
            trust = (view_trust * topology_trust).sqrt()
            if self.training:
                with torch.no_grad():
                    self.topology_trust_ema.mul_(0.95).add_(0.05 * trust.mean())
            slow_trust = self.topology_trust_ema.detach().expand_as(trust)
            mix = spec.max_mix * confidence * (trust * slow_trust).sqrt()

        elif spec.mode == "boundary_rescue":
            boundary = self._routing_entropy(first_clean)
            safe = confidence * view_trust * topology_trust
            need = 0.20 + 0.80 * boundary
            mix = spec.max_mix * safe * need

        elif spec.mode == "scale_curriculum":
            if self.current_epoch <= 10:
                open_scales = 1
            elif self.current_epoch <= 25:
                open_scales = 2
            else:
                open_scales = 3
            scale_trust = topology_agreement[:, :open_scales].mean(-1)
            curriculum = 0.50 + 0.50 * min(1.0, self.current_epoch / 30.0)
            mix = spec.max_mix * confidence * scale_trust * curriculum

        elif spec.mode == "sufficiency_diversity":
            # 以约 15% 扰动为平衡点：过弱无新信息，过强容易语义漂移。
            pareto = torch.exp(-((diversity - 0.15) / 0.10).pow(2))
            mix = spec.max_mix * confidence * topology_trust * pareto
            residual = F.normalize(linked_anchor - base_anchor, dim=-1)
            base_direction = F.normalize(base_anchor, dim=-1)
            auxiliary = (residual * base_direction).sum(-1).abs().mean()

        elif spec.mode == "node_edge_cycle_reciprocity":
            augmentation = trace[:, :3]
            cleanliness = weights[:, :3]
            topology = topology_agreement
            reciprocity = augmentation * cleanliness * topology
            gate_input = torch.cat(
                (augmentation, cleanliness, topology, reciprocity), dim=-1
            )
            learned = torch.sigmoid(self.reciprocal_gate(gate_input)).squeeze(-1)
            reciprocal_trust = reciprocity.sum(-1).clamp(0, 1)
            mix = spec.max_mix * learned * reciprocal_trust
            # 门网络只拟合三阶段共同认可的可信度，不直接接受对比目标梯度。
            auxiliary = F.mse_loss(learned, reciprocal_trust.detach())

        else:
            raise RuntimeError(spec.mode)

        # 门只负责是否接受链接干预，不允许下游目标篡改证据本身。
        mix = mix.detach().clamp(0, spec.max_mix)
        diagnostics = {
            "mean_retention": retention.mean(),
            "mean_view_trust": view_trust.mean(),
            "mean_topology_trust": topology_trust.mean(),
        }
        return mix, auxiliary, diagnostics

    def idea_loss(
        self, first_data: Data, second_data: Data
    ) -> tuple[Tensor, dict[str, float]]:
        first_history = self.encoder(first_data)
        second_history = self.encoder(second_data)
        first_graph_bundle = self.first._bundle_from_history(first_data, first_history)
        second_graph_bundle = self.first._bundle_from_history(second_data, second_history)
        first_clean_bundle = self.second._bundle_from_history(first_data, first_history)
        second_clean_bundle = self.second._bundle_from_history(second_data, second_history)
        first_topology_bundle = self.third._bundle_from_history(first_data, first_history)
        second_topology_bundle = self.third._bundle_from_history(second_data, second_history)
        first_graph = first_graph_bundle["graph"]
        second_graph = second_graph_bundle["graph"]

        base_anchor, base_positive, base_hard, base_auxiliary = (
            self._faithful_base_path(
                first_graph, second_graph,
                first_clean_bundle, second_clean_bundle,
                first_topology_bundle, second_topology_bundle,
            )
        )
        first_clean, second_clean, weights, confidence, agreement, clean_auxiliary = (
            self._trace_conditioned_clean(
                first_clean_bundle, second_clean_bundle,
                first_data.augmentation_trace, second_data.augmentation_trace,
            )
        )

        # 候选路径可以使用随机语义噪声，但不能改变下一批基线增强的 RNG 状态。
        rng_state = self._rng_state(first_graph)
        linked_anchor, linked_positive, linked_hard, linked_auxiliary, _ = (
            self._linked_path(
                first_clean, second_clean, weights, confidence, agreement,
                first_data.augmentation_trace, second_data.augmentation_trace,
                first_topology_bundle, second_topology_bundle,
            )
        )
        self._restore_rng(first_graph, rng_state)

        topology_agreement = self._topology_agreement(
            first_topology_bundle, second_topology_bundle
        )
        mix, story_auxiliary, story_diagnostics = self._story_mix(
            base_anchor, base_positive, base_hard,
            linked_anchor, linked_positive, linked_hard,
            first_clean, second_clean, weights, confidence, agreement,
            first_data.augmentation_trace, second_data.augmentation_trace,
            topology_agreement,
        )
        anchor = base_anchor + mix.unsqueeze(-1) * (linked_anchor - base_anchor)
        positive = base_positive + mix.unsqueeze(-1) * (
            linked_positive - base_positive
        )
        hard = base_hard + mix.view(-1, 1, 1) * (linked_hard - base_hard)
        objective = self.first.objective(
            F.normalize(anchor, dim=-1),
            F.normalize(positive, dim=-1),
            F.normalize(hard, dim=-1),
            self._temperature(),
        )
        linked_regularizer = linked_auxiliary + 0.1 * clean_auxiliary
        extra_auxiliary = self.story_pipeline.auxiliary_weight * (
            linked_regularizer + story_auxiliary
        )
        total = objective + base_auxiliary + extra_auxiliary
        return total, {
            "module_objective": float(objective.detach()),
            "base_auxiliary": float(base_auxiliary.detach()),
            "linked_auxiliary": float(extra_auxiliary.detach()),
            "mean_mix": float(mix.mean()),
            "mean_clean_confidence": float(confidence.mean().detach()),
            **{
                key: float(value.detach())
                for key, value in story_diagnostics.items()
            },
        }


def build_retogcl_story_pipeline(
    pipeline_id: str,
    in_dim: int,
    hidden_dim: int,
    layers: int,
    num_classes: int,
    idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
) -> ReToGCLStoryPipeline:
    return ReToGCLStoryPipeline(
        pipeline_id, in_dim, hidden_dim, layers, num_classes, idea_parameters
    )
