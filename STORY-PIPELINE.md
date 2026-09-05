# ReToGCL 故事化 Pipeline 设计

## 目标

把三个已有创新点从“并列模块”改成有因果顺序的数据流：

```text
D7-I10 增强与扰动轨迹
        ↓ 提供增强保真证据
D7-I01 样本×视图洁净估计
        ↓ 提供洁净表示与可靠度
D10-I04 多尺度拓扑投影
        ↓ 提供跨尺度稳定性
安全接受/反馈机制
        ↘ 不可信时退回原 ReToGCL
```

所有策略都保留原 ReToGCL 主路径，避免强制连接在某类数据上破坏表示。

## 近年研究依据

- ICML 2023 的 [Graph Contrastive Saliency](https://proceedings.mlr.press/v202/wei23c.html) 指出随机增强会破坏语义，提出用显著性筛选语义相关子结构。
- NeurIPS 2024 的 [GOUDA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/41efc12982eca6f8bb5e48dc3a84b843-Abstract-Conference.html) 同时强调增强的一致性与多样性。
- UAI 2024 的 [GCVR](https://proceedings.mlr.press/v244/wen24a.html) 用跨视图重构保证表示的充分性与鲁棒性。
- CVPR 2024 的 [HEATS](https://openaccess.thecvf.com/content/CVPR2024/html/Zhuo_Improving_Graph_Contrastive_Learning_via_Adaptive_Positive_Sampling_CVPR_2024_paper.html) 通过自适应正样本处理盲目采样问题。
- ICML 2024 的 [BAT](https://proceedings.mlr.press/v235/liu24ay.html) 从拓扑角度而非单纯重采样处理类别不平衡。
- 2024 年的 [TopoGCL](https://arxiv.org/abs/2406.17251) 将多分辨率拓扑不变量用于图对比学习。
- UAI 2025 的 [IGCL-CS](https://proceedings.mlr.press/v286/chen25a.html) 表明社区结构可以提供更可靠的对比监督。
- ICML 2025 的 [模糊边界图对比学习](https://openreview.net/forum?id=1b94nzUAvt) 关注局部聚集、全局稀疏和边界样本。
- NeurIPS 2025 的 [混合协作增强](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a845c35c57947ce27d7c3f5c2ed5efef-Abstract-Conference.html) 强调点、边以及多粒度增强之间的协作。
- NeurIPS 2025 的 [扩散引导图增强](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e4e723536170f9c72e04a8cc535cea18-Abstract-Conference.html) 只选择真正能从增强中获益的样本。

这些工作分别研究增强、重构、拓扑或样本选择；本项目的差异是让增强证据、洁净证据和拓扑证据在同一图级 Pipeline 中前向传递并反向否决。

## 八套候选策略

| 编号 | 名称 | 完整故事线 | 重点风险 |
|---|---|---|---|
| S01 | 三证据否决链 | 增强、洁净、拓扑三方共同表决，任一证据不足就退回原路径 | 过度保守 |
| S02 | 反事实收益回路 | 同时运行原路径与链接路径，把链接看作干预，仅接受能改善对比能量的干预 | 训练早期收益估计有噪声 |
| S03 | 跨视图可恢复闭环 | 拓扑表示反向恢复增强轨迹和洁净表示，可恢复才说明前向连接没有丢失关键信息 | 重构目标可能偏强 |
| S04 | 拓扑慢反馈增强 | 当前批拓扑可信度形成指数滑动慢变量，用于调节下一批增强强度 | 批次顺序敏感 |
| S05 | 洁净边界救援 | 原型边界样本需要更强修正，但只有洁净且拓扑一致时才允许救援 | 原型尚未稳定时边界不准 |
| S06 | 多尺度可信课程 | 先使用浅层拓扑，再逐步开放中层和深层拓扑，避免训练初期强结构约束 | 后期尺度可能相互冲突 |
| S07 | 充分性—多样性帕累托桥 | 增强过弱或过强都拒绝，只接受兼具信息充分性和视图多样性的连接 | 最优扰动区间依赖数据 |
| S08 | 点边环互惠协作 | 点、边、环的增强保真、洁净权重和拓扑尺度一一对应，再联合决定路由 | 多证据乘积可能偏小 |

## 公平实验约束

- 固定原 ReToGCL 的编码器、优化器、学习率、训练轮数与五折探针协议。
- Pipeline 只改变三个创新模块之间的信息连接。
- 候选分支的随机语义噪声不会推进全局 RNG，下一批增强与基线保持一致。
- 结果独立保存，不覆盖原 ReToGCL、P05 或上一轮十套 Pipeline。
