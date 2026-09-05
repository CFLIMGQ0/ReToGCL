"""只针对 ACVG、RDTG、LTCP 的匹配消融；Full 直接复用未修改的原模型。"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from src.baselines.gcl_baselines import augment_graph
from src.models.modular_triple_research_ideas import ModularTripleResearchIdeaBalanceGCL
from src.models.research_ideas import _standardize


IDEA_IDS = ("D11-I01", "D2-I09", "D10-I10")
CONFIGS = {
    "Full": (True, True, True),
    "w/o ACVG": (False, True, True),
    "w/o RDTG": (True, False, True),
    "w/o LTCP": (True, True, False),
    "Backbone": (False, False, False),
    "ACVG only": (True, False, False),
    "RDTG only": (False, True, False),
    "LTCP only": (False, False, True),
}


class AReTGCLAblation(ModularTripleResearchIdeaBalanceGCL):
    """保留原参数构造顺序、残差混合、路由、负样本和温度。"""

    def __init__(self, in_dim, hidden_dim, layers, num_classes, idea_parameters,
                 use_acvg=True, use_rdtg=True, use_ltcp=True):
        super().__init__(IDEA_IDS, in_dim, hidden_dim, layers, num_classes, idea_parameters)
        self.use_acvg = use_acvg
        self.use_rdtg = use_rdtg
        self.use_ltcp = use_ltcp

    def prepare_views(self, batch):
        if self.use_acvg:
            return super().prepare_views(batch)
        # 原方法先执行两个随后被覆盖的增强；保留其随机数消耗以匹配 Full。
        augment_graph(batch, "edge_drop", 0.2)
        augment_graph(batch, "attr_mask", 0.2)
        a, b, c = self.first._parameter_controls()
        strength = min(0.8, max(0.01, 0.2 * a))
        secondary = min(0.8, max(0.01, strength * max(c, 0.05) / math.sqrt(max(b, 1e-3))))
        # 两支都使用已有 node_drop，保留原来逐图 5%/2% 的节点扰动预算。
        return augment_graph(batch, "node_drop", strength), augment_graph(batch, "node_drop", secondary)

    def _apply_idea_transform(self, model, first_bundle, second_bundle, first_graph, second_graph):
        if model.target_module == "M2" and not self.use_rdtg:
            # 去掉 gate * cycle，只保留同一个归一化 graph head 与外层残差混合。
            # 直接返回原始 sum-pool 会把原来的 -0.5 系数变成 +1，产生尺度/符号混杂。
            a = model._parameter_controls()[0]
            first = first_graph + a * (first_bundle["views"][:, 3] - first_graph)
            second = second_graph + a * (second_bundle["views"][:, 3] - second_graph)
            return first, second, first.new_zeros(())
        if model.target_module == "M7" and not self.use_ltcp:
            # 三个同容量 head 都只读取最终层，由 _bundle_from_history 的局部包装提供。
            # 三个终点坐标平均，再通过原来的瓶颈网络；参数数量与原投影完全相同。
            a = model._parameter_controls()[0]
            pa = model.bottleneck(first_bundle["scales"].mean(1))
            pb = model.bottleneck(second_bundle["scales"].mean(1))
            return first_graph + a * (pa - first_graph), second_graph + a * (pb - second_graph), pa.new_zeros(())
        return super()._apply_idea_transform(model, first_bundle, second_bundle, first_graph, second_graph)

    def _weighted_auxiliary(self, model, auxiliary):
        if (model.target_module == "M2" and not self.use_rdtg) or (model.target_module == "M7" and not self.use_ltcp):
            return auxiliary.new_zeros(())
        return super()._weighted_auxiliary(model, auxiliary)

    def idea_loss(self, first_data, second_data):
        if self.use_ltcp:
            return super().idea_loss(first_data, second_data)
        original = self.third._bundle_from_history

        def endpoint_bundle(data, history):
            # 原层数及 graph readout 保持不变，只把 M7 的三个 scale head 输入换成终点。
            bundle = original(data, history)
            from torch_geometric.nn import global_mean_pool
            pooled = global_mean_pool(history[:, -1], data.batch, size=bundle["graph"].size(0))
            bundle["scales"] = torch.stack([F.normalize(head(pooled), dim=-1) for head in self.third.scale_heads], dim=1)
            return bundle

        self.third._bundle_from_history = endpoint_bundle
        try:
            return super().idea_loss(first_data, second_data)
        finally:
            self.third._bundle_from_history = original

    @torch.no_grad()
    def mechanism_diagnostics(self, first_data, second_data):
        """独立评估模式记录真实内部值，不消耗随机数，不更新 BN 或训练目标。"""
        was_training = self.training
        self.eval()
        h1, h2 = self.encoder(first_data), self.encoder(second_data)
        a = self.second._bundle_from_history(first_data, h1)
        b = self.second._bundle_from_history(second_data, h2)
        d = (a["views"][:, 2] - b["views"][:, 2]).square().mean(-1)
        z = _standardize(d, 0)
        g = torch.sigmoid(-z)
        raw = a["graph"]
        ungated = raw + self.second._parameter_controls()[0] * (a["views"][:, 3] - raw)
        gated, _, _ = self.second._module_transform(a, b)
        c = self.third._bundle_from_history(first_data, h1)
        trajectory = self.third.bottleneck(c["scales"][:, 0] - 2*c["scales"][:, 1] + c["scales"][:, 2])
        from torch_geometric.nn import global_mean_pool
        end = global_mean_pool(h1[:, -1], first_data.batch)
        endpoint = self.third.bottleneck(torch.stack([F.normalize(head(end), dim=-1) for head in self.third.scale_heads], 1).mean(1))
        def stats(x):
            x = x.float().flatten()
            return {"mean": x.mean().item(), "std": x.std(unbiased=False).item(), "min": x.min().item(), "max": x.max().item(), "quantiles": torch.quantile(x, torch.tensor([0., .25, .5, .75, 1.], device=x.device)).tolist(), "values": x.tolist()}
        result = {"d_i": stats(d), "standardized_disagreement": stats(z), "g_i": stats(g),
                  "raw_graph_norm": stats(raw.norm(dim=-1)), "ungated_graph_norm": stats(ungated.norm(dim=-1)),
                  "gated_graph_norm": stats(gated.norm(dim=-1)), "gated_to_ungated_norm_ratio": stats(gated.norm(dim=-1) / ungated.norm(dim=-1).clamp_min(1e-12)),
                  "trajectory_endpoint_l2": stats((trajectory-endpoint).norm(dim=-1)),
                  "trajectory_endpoint_cosine": stats(F.cosine_similarity(trajectory, endpoint))}
        self.train(was_training)
        return result
