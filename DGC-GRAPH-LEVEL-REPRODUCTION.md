# 六种深度图聚类方法的图级复现说明

## 任务口径

THESAURUS、ARLN、MHGC、SCGC、RDSA、DCGC 的原始任务都是“单张属性图中的节点聚类”。
本项目的 18 个数据集则由多张独立图构成，目标是图级分类。因此，官方训练脚本不能原样读取这些
数据集。本次实验采用以下统一且无标签泄漏的适配口径：

1. 使用论文的核心无监督目标预训练图编码器，预训练阶段不读取图标签；
2. 把论文中的节点聚类表示对象改为独立图的图表示；
3. 冻结表示后使用相同随机种子进行分层五折线性评测；
4. 统一报告 Accuracy、NMI、ARI 和 Macro-F1 的五折均值与样本标准差；
5. SIDER 和 Tox21 按官方多任务标签逐任务五折，再做任务宏平均。

这些结果必须称为 **graph-level adaptation**，不能称为原论文节点聚类协议下的官方结果。

## 论文与源码

| 方法 | 本次对应论文 | 原始任务 | 源码使用情况 |
|---|---|---|---|
| THESAURUS | THESAURUS: Contrastive Graph Clustering by Swapping Fused Gromov-Wasserstein Couplings，AAAI 2025 | 节点聚类 | 论文没有公开官方仓库；依据论文复现语义原型、平衡分配、交换预测、原型图和边际动量更新 |
| ARLN | Deep Graph Clustering via Aligning Representation Learning，Neural Networks 2025 | 节点聚类 | 论文页面没有公开仓库；依据论文复现 AE/GAE 双视图、实例/特征/分配概率三重对齐 |
| MHGC | MHGC: Multi-scale Hard Sample Mining for Contrastive Deep Graph Clustering，IP&M 2025 | 节点聚类 | 使用作者仓库并适配多尺度对比、Beta 混合难样本后验和聚类目标分布 |
| SCGC | Simple Contrastive Graph Clustering，IEEE TNNLS 2023 | 节点聚类 | 使用作者仓库并适配拉普拉斯平滑、双线性头、高斯扰动和结构相似度重构 |
| RDSA | RDSA: A Robust Deep Graph Clustering Framework via Dual Soft Assignment，DASFAA 2025 | 节点聚类 | 使用作者仓库并适配属性/拓扑融合、结构软分配和地标软分配 |
| DCGC | Dual-Center Graph Clustering with Neighbor Distribution，ECAI 2025 | 节点聚类 | 使用作者仓库并适配低/高频滤波、邻居难负样本、特征中心和邻居分布中心 |

第三方源码保存在 `src/sota_sources_dgc/`，其中还保留 PyDGC 作为统一深度图聚类基准参考。
第三方目录不做本地修改，各仓库固定提交如下：

- MHGC：`875bb84d38ccb03a6abb9e8eb5fab436e0015f0b`
- SCGC：`ab029e452e6fc91c29163ebf4ff2050208351839`
- RDSA：`0d6a14f9a0144f602698659bae7e0e8091f1ef31`
- DCGC：`d7282bcc2a0d0b252d08e3151895ec59641f9e7f`
- PyDGC：`7f4ea24d7c421eb4206dc61ac11a74d252e9b651`

## 正式实验配置

- 18 个数据集：NCI1、PROTEINS、COLLAB、MUTAG、COLORS-3、PTC_MR、Mutagenicity、
  ClinTox、BACE、BBBP、ABIDE、ADHD200、SIDER、HIV、AIDS、Tox21、BZR、COX2；
- 无监督预训练：40 轮；
- 隐藏维度：64；
- GNN 层数：3；
- 评测：分层五折线性分类器，每折 200 轮；
- 随机种子：42；
- 结果目录：`outputs/dgc_graph_level_6x18/`。

本机单任务示例：

```bash
python src/scripts/run_dgc_graph_level_five_fold.py \
  --methods mhgc \
  --datasets MUTAG \
  --device cuda:0
```

并行分片示例：

```bash
python src/scripts/run_dgc_graph_level_parallel.py \
  --devices 0 1 \
  --jobs-per-device 2 \
  --shard-count 6 \
  --shard-indices 0 1 \
  --tag host202
```

实时查看单个调度器状态：

```bash
watch -n 5 'cat outputs/dgc_graph_level_6x18/launcher_status_host202.json'
```

查看跨主机最终审计状态：

```bash
cat outputs/dgc_graph_level_6x18/execution_status.json
```
