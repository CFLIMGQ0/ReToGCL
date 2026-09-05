# CITA-GCL 模型命名与实验追溯

## 正式名称

- 模型名称：CITA-GCL；简称：CITA。
- 英文全称：Cleanliness-Informed Topological Alignment for Graph Contrastive Learning。
- 中文名称：洁净度引导的拓扑对齐图对比学习。
- 历史实验编号：`PL069`；部分调度与汇总脚本中的内部方法键为 `pl069`。

论文、结果表和对外说明统一使用 CITA-GCL。`PL069` 仅作为兼容旧实验的标识保留；它不是另一种模型，也不应与 CITA-GCL 重复计入对比。

## 对应的实际模型

CITA-GCL 对应已有的 `PL069` 实现，并非本次重新设计或训练的模型。主体保留 `D7-I10`、`D7-I01`、`D10-I04` 三项设计及 T077 的基础改动：

1. 互补图增强与共享 GIN 编码生成两种视图的图表示。
2. 样本与视图的软洁净表征参与语义路由，影响困难负样本构建。
3. 拓扑条件映射处理对比表示，困难负样本也通过对应的拓扑映射。
4. 锚点额外引入映射前表征的弱旁路，修正量停止梯度；随后进行归一化与对比学习。

PL069 专属配置保持为：`primary_action=topology_anchor_bypass`、`primary_strength=0.065`、`primary_gate=fixed`、`secondary_action=none`、`secondary_strength=0.0`、`secondary_gate=fixed`、`gradient_policy=detach_both`、`schedule=constant`。基础温度为 `0.045`，困难负样本对齐增益为 `4.0`，辅助项系数为 `1.0`，权重衰减为 `5e-4`。

该旁路使用固定系数，不是按样本学习的自适应门控。名称描述已有机制，不代表额外的理论保证或新颖性证明。

## 兼容与结果说明

- 原始 ReToGCL 的名称、实现和结果继续保留，不能直接替换为 CITA-GCL。
- 实现仍位于 `src/models/retogcl_t077_adaptive_pipeline.py`；该文件还支持其他候选配置，只有 `PL069` 对应这里命名的 CITA-GCL。
- 运行入口仍使用 `--config-id PL069`；JSON 方法键、任务编号、检查点、结果目录和文件名不迁移。
- 最新对外结果见 [table.md](table.md)；两份表格生成脚本会将旧编号显示为 CITA-GCL，统计计算不变。
- `model.yaml` 保持通用模块化基线配置，不因模型更名而切换默认模型。
- `paper_retogcl/` 保留原始 ReToGCL 草稿，不将原始模型的公式和实验冒充为 CITA-GCL。
- 本次仅统一命名，不重新训练、不调整参数、不修改已有实验指标，也不更改 GitHub 仓库地址。
