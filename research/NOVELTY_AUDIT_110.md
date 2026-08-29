# 110 个研究 idea 外部近邻与完整链审查

- 生成时间：2026-08-12T01:22:11.622957+00:00
- 覆盖：110/110。每项先与所属方向精读矩阵中的对应顶会/顶刊论文做完整链比较，再执行两组组件对检索（Crossref，2010 至今，单组前 6 个候选）；高风险项回查额外论文或会议原始页面。
- 判定：93 项初步通过；12 项条件通过并要求强制消融；5 项原方案否决后已改写为 v2。
- 边界：这是论文近邻与完整方法链审查，不等同于专利自由实施意见，也不能证明全球范围绝对无人提出。正式投稿前仍需按目标会议截稿日补检索。

## 判定规则

- `通过`：发现相同组件，但未发现同样的有序完整链与相同子模块联系方式。
- `条件通过`：宏观组合或关键组件邻近，只有保留指定连接并完成去连接/单组件/完整模型消融后才可主张创新。
- `v2 重写通过`：原链已有强先例，旧实验作废；表中记录重写后的不同机制。

## 逐项结果

| ID | idea | 结论 | 最近风险证据 | 完整链判定 |
|---|---|---|---|---|
| D1-I01 | 边界条件的三几何重心门 | 通过 | [Mitigating Local Cohesion and Global Sparseness in GCL with Fuzzy Boundaries](https://openreview.net/forum?id=1b94nzUAvt) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D1-I02 | 余层回路一致门 | 通过 | [A New Mechanism for Eliminating Implicit Conflict in GCL](https://ojs.aaai.org/index.php/AAAI/article/view/29125) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D1-I03 | 切空间—GW 双校准相似度 | 通过 | [ProGCL: Rethinking Hard Negative Mining in GCL](https://proceedings.mlr.press/v162/xia22b.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D1-I04 | 曲率反事实响应校准门 | v2 重写通过 | [GeoMoE：Geometric Mixture-of-Experts with Curvature-Guided Adaptive Routing](https://arxiv.org/abs/2603.22317) | 原链以曲率直接路由专家，与 GeoMoE 的核心机制强冲突；已改为曲率反事实响应校准。 |
| D1-I05 | 增强轨迹 Koopman 相似度 | 通过 | [Similarity Preserving Adversarial Graph Contrastive Learning](https://dl.acm.org/doi/10.1145/3580305.3599383) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D1-I06 | 热带投影边界门 | 通过 | [InfoGCL: Information-Aware Graph Contrastive Learning](https://proceedings.neurips.cc/paper/2021/hash/ff1e68e74c6b16a1a7b5d958b95e120c-Abstract.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D1-I07 | Bures 协方差语义门 | 通过 | [Adversarial Graph Augmentation to Improve GCL](https://proceedings.neurips.cc/paper_files/paper/2021/hash/854f1fb6f65734d9e49f708d6cd84ad6-Abstract.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D1-I08 | 局部维数失配门 | 通过 | [Graph Contrastive Learning with Augmentations](https://proceedings.neurips.cc/paper/2020/hash/3fe230348e9a12c13120749e3f9fa4cd-Abstract.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D1-I09 | 三角闭合关系门 | 通过 | [Unified Graph Augmentations for Generalized Contrastive Learning on Graphs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/41efc12982eca6f8bb5e48dc3a84b843-Abstract-Conference.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D1-I10 | 反事实编辑本质映射 | 通过 | [Teacher-Guided Graph Contrastive Learning](https://openreview.net/forum?id=15tjpSHI15) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D2-I01 | 持久寿命加权环胞腔 | 通过 | [Weisfeiler and Lehman Go Cellular: CW Networks](https://proceedings.neurips.cc/paper/2021/hash/157792e4abb490f99dbd738483e0d2d4-Abstract.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D2-I02 | 环的边超图—节点超图对偶 | 通过 | [Weisfeiler and Lehman Go Topological: MPSN](https://arxiv.org/abs/2103.03212) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D2-I03 | Hodge 残差反馈胞腔 | 通过 | [Learning From Simplicial Data Based on Random Walks and 1D Convolutions](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fa30825df7ceaba452d5533538ea29c2-Abstract-Conference.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D2-I04 | Möbius 反演环聚合 | 通过 | [Weisfeiler and Lehman Go Paths](https://ojs.aaai.org/index.php/AAAI/article/view/29463) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D2-I05 | 曲率预算的自适应环上限 | 通过 | [Topological Neural Networks go Persistent, Equivariant, and Continuous](https://proceedings.mlr.press/v235/verma24a.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D2-I06 | Hourglass 环消息 | 通过 | [Demystifying Topological Message-Passing with Relational Structures](https://openreview.net/forum?id=QC2qE1tcmd) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D2-I07 | 有向环的规范场提升 | 通过 | [TopInG: Topologically Interpretable Graph Learning](https://proceedings.mlr.press/v267/xin25b.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D2-I08 | 环—割互补网络 | 通过 | [Contraction and Hourglass Persistence](https://openreview.net/forum?id=mWN6cpA6Wr) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D2-I09 | 环可靠性共形筛选 | 通过 | [Copresheaf Topological Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2025/hash/dc62cd4ec77f3bbec8e6245d0bd91d08-Abstract-Conference.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D2-I10 | 拓扑相变温度提升 | 通过 | [Directed Semi-Simplicial Learning](https://openreview.net/forum?id=YR3CNvFfCr) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D3-I01 | 嵌套尺度超图塔 | 通过 | [TriCL: Tri-directional Contrastive Learning on Hypergraphs](https://ojs.aaai.org/index.php/AAAI/article/view/26019) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D3-I02 | 尺度间 Gromov-Wasserstein 输运 | 通过 | [Augmentations in Hypergraph Contrastive Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0cd1eec0eeaf5ce1bf6d8875a7c1d095-Abstract-Conference.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D3-I03 | 尺度导数对比 | 通过 | [Hypergraph Contrastive Collaborative Filtering](https://dl.acm.org/doi/10.1145/3477495.3532058) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D3-I04 | 节点—边超图交换子 | 通过 | [Self-Supervised Multi-Channel Hypergraph Convolutional Network](https://dl.acm.org/doi/10.1145/3442381.3449940) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D3-I05 | 超边尺度小波包 | 通过 | [Hypergraph-Enhanced Contrastive MVC with Hyper-Laplacian](https://openreview.net/forum?id=L5VPvuPMKs) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D3-I06 | 尺度因果干预门 | 通过 | [Hypergraph-Based Multi-View Multi-Label Classification via Adaptive High-Order Semantic Fusion](https://ojs.aaai.org/index.php/AAAI/article/view/39719) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D3-I07 | 超边出生—死亡记忆库 | 通过 | [Heterophily-aware Contrastive Learning for Heterophilic Hypergraphs](https://ojs.aaai.org/index.php/AAAI/article/view/39473) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D3-I08 | 多尺度超图专家的负载平衡路由 | 条件通过 | [Dynamic Hypergraph Structure Learning for Traffic Flow Forecasting](https://doi.org/10.1109/ICDE55515.2023.00179) | 未发现完整链同构，但宏观形式接近；必须保留关键连接并做四组消融。 |
| D3-I09 | 尺度反事实一致性 | 通过 | [Deep Heterogeneous Contrastive Hyper-Graph Learning](https://openreview.net/forum?id=HOwJ5YfJQh) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D3-I10 | 超图神经 ODE 尺度流 | 通过 | [Hypergraph Learning for Unsupervised Graph Alignment via Optimal Transport](https://ojs.aaai.org/index.php/AAAI/article/view/35498) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D4-I01 | 视图子集 Möbius 融合 | 条件通过 | [Self-Weighted Contrastive Learning among Multiple Views](https://proceedings.neurips.cc/paper_files/paper/2023/hash/03b13b0db740b95cb741e007178ef5e5-Abstract-Conference.html) | 未发现完整链同构，但宏观形式接近；必须保留关键连接并做四组消融。 |
| D4-I02 | Shapley—DPP 组合选择 | 通过 | [Contrastive Multi-View Representation Learning on Graphs](https://proceedings.mlr.press/v119/hassani20a.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D4-I03 | 视图组合超图 | 条件通过 | [CoCo: A Coupled Contrastive Framework](https://proceedings.mlr.press/v202/yin23a.html) | 未发现完整链同构，但宏观形式接近；必须保留关键连接并做四组消融。 |
| D4-I04 | 非交换顺序融合 | 通过 | [AF-UMC: Alignment-Free Fusion for Unaligned Multi-View Clustering](https://proceedings.neurips.cc/paper_files/paper/2025/hash/965484d5b2b2624ba17295612a3ba7e8-Abstract-Conference.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D4-I05 | 可靠性张量列车融合 | 通过 | [MMPG: MoE-based Adaptive Multi-Perspective Graph Fusion](https://ojs.aaai.org/index.php/AAAI/article/view/37096) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D4-I06 | 视图反事实路由器 | 通过 | [Reliable Conflictive Multi-View Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29546) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D4-I07 | PID 原子—Hodge 槽位耦合 | v2 重写通过 | [I2MoE：Interpretable Multimodal Interaction-aware Mixture-of-Experts](https://arxiv.org/abs/2505.19190) | 原链以 PID 冗余/独有/协同信息构造并加权专家，与 I2MoE 强冲突；已改为 PID 原子与 Hodge 槽位守恒耦合。 |
| D4-I08 | 树结构融合的可逆流 | 通过 | [Self-supervised Trusted Contrastive Multi-view Clustering](https://ojs.aaai.org/index.php/AAAI/article/view/33902) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D4-I09 | 冲突质量守恒融合 | 通过 | [Adaptive Evolutionary Fusion for Multi-View Clustering](https://ojs.aaai.org/index.php/AAAI/article/view/40110) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D4-I10 | 组合课程融合 | 条件通过 | [Hierarchical mutual distillation for multi-view fusion: Learning from all possible view combinations](https://www.sciencedirect.com/science/article/abs/pii/S0031320326003973) | 未发现完整链同构，但宏观形式接近；必须保留关键连接并做四组消融。 |
| D5-I01 | 节点—边—环—图共形可靠性张量 | 通过 | [Uncertainty Quantification over Graph with Conformalized GNNs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D5-I02 | 证据冲突瀑布 | 通过 | [What Makes Graph Neural Networks Miscalibrated?](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5975754c7650dfee0682e06e1fec0522-Abstract-Conference.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D5-I03 | PAC-Bayes 粒度预算 | 通过 | [Be Confident! Towards Trustworthy GNNs via Confidence Calibration](https://proceedings.neurips.cc/paper/2021/hash/c7a9f13a6c0940277d46706c7ca32601-Abstract.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D5-I04 | Jackknife-after-bootstrap 结构可靠性 | 通过 | [Accurate and Scalable Estimation of Epistemic Uncertainty for GNNs](https://openreview.net/forum?id=ZL6yd6N1S2) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D5-I05 | 可靠性守恒流 | 通过 | [Evidential Uncertainty Probes for GNNs](https://proceedings.mlr.press/v258/yu25a.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D5-I06 | 局部 Lipschitz 可靠性 | 通过 | [Similarity-Navigated Conformal Prediction for Graphs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/571c7e164fb1ffbcf2f84a63784451ec-Abstract-Conference.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D5-I07 | 互信息下界置信区间门 | 通过 | [Trusted Multi-View Deep Learning with Opinion Aggregation](https://ojs.aaai.org/index.php/AAAI/article/view/20724) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D5-I08 | 可靠性 Kalman 滤波 | 通过 | [Enhancing Multi-View Classification Reliability with Adaptive Rejection](https://ojs.aaai.org/index.php/AAAI/article/view/34088) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D5-I09 | 反事实可恢复可靠性 | 通过 | [Training Uncertainty-Aware Classifiers with Conformalized Deep Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8c96b559340daa7bb29f56ccfbbc9c2f-Abstract-Conference.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D5-I10 | 可靠性 Shapley 交互图 | 通过 | [Diagnostic Uncertainty Calibration](https://proceedings.mlr.press/v130/mimori21a.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D6-I01 | 增强集合 Wasserstein 重心正样本 | 条件通过 | [SwAV: Unsupervised Learning by Contrasting Cluster Assignments](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html) | 未发现完整链同构，但宏观形式接近；必须保留关键连接并做四组消融。 |
| D6-I02 | 流形 Fréchet 综合正样本 | 通过 | [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D6-I03 | DPP 子集—Set Transformer 正样本 | 通过 | [NNCLR: Nearest-Neighbor Contrastive Learning](https://proceedings.neurips.cc/paper/2021/hash/14f2ebeab937ca128186e7ba876faef9-Abstract.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D6-I04 | 扩散去噪正样本 | 条件通过 | [Prototypical Contrastive Learning of Unsupervised Representations](https://openreview.net/forum?id=KmykpuSrjcq) | 未发现完整链同构，但宏观形式接近；必须保留关键连接并做四组消融。 |
| D6-I05 | 鲁棒几何中位数正样本 | 通过 | [You Never Cluster Alone](https://proceedings.neurips.cc/paper/2021/hash/e96ed478dab8595a7dbda4cbcbee168f-Abstract.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D6-I06 | 正样本超图池化 | 条件通过 | [HyperGCL：Hypergraph Contrastive Learning for Graph Classification](https://arxiv.org/abs/2502.13277) | 未发现完整链同构，但宏观形式接近；必须保留关键连接并做四组消融。 |
| D6-I07 | Barycentric capsule 正样本 | 通过 | [CLEAR: Cluster-Enhanced Contrast for Graph Representation](https://doi.org/10.1109/TNNLS.2022.3177775) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D6-I08 | 最小充分综合正样本 | 通过 | [Unsupervised Graph-Level Representation Learning with Hierarchical Contrasts](https://doi.org/10.1016/j.neunet.2022.10.016) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D6-I09 | 多视图意见交集正样本 | 通过 | [Hierarchically Contrastive Hard Sample Mining for Graph Pretraining](https://doi.org/10.1109/TNNLS.2023.3297607) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D6-I10 | 增强轨迹样条正样本 | 通过 | [Contrastive Learning Meets Homophily: Two Birds with One Stone](https://proceedings.mlr.press/v202/he23c.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D7-I01 | 样本×视图软洁净矩阵 | 通过 | [NoisyGL: A Comprehensive Benchmark for GNNs under Label Noise](https://proceedings.neurips.cc/paper_files/paper/2024/hash/436ffa18e7e17be336fd884f8ebb5748-Abstract-Datasets_and_Benchmarks_Track.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D7-I02 | 跨空间陪审团清洗 | 通过 | [Mitigating Label Noise on Graphs via Topological Sample Selection](https://proceedings.mlr.press/v235/wu24ae.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D7-I03 | 时间反转记忆清洗 | 通过 | [GraphCleaner: Detecting Mislabelled Samples in Graph Benchmarks](https://proceedings.mlr.press/v202/li23ai.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D7-I04 | 反事实影响清洗 | 通过 | [NRGNN: Learning a Label Noise Resistant GNN](https://dl.acm.org/doi/10.1145/3447548.3467364) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D7-I05 | 洁净性共形控制器 | 条件通过 | [Robust Training of GNNs via Noise Governance](https://dl.acm.org/doi/10.1145/3539597.3570410) | 未发现完整链同构，但宏观形式接近；必须保留关键连接并做四组消融。 |
| D7-I06 | 非交换双去污算子 | v2 重写通过 | [Noise-Disentangled GCL via Low-Rank and Sparse Subspace Decomposition](https://sigport.org/documents/noise-disentangled-graph-contrastive-learning-low-rank-and-sparse-subspace-decomposition) | 原链的低秩+稀疏图对比去噪已有直接先例；已改为属性/拓扑去污算子的非交换子。 |
| D7-I07 | 拓扑相位一致清洗 | 通过 | [GDeR: Prototypical Graph Pruning](https://openreview.net/forum?id=O97BzlN9Wh) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D7-I08 | 双记忆库洁净交换 | 通过 | [United We Stand, Divided We Fall: N2G for Robust Graph Classification](https://proceedings.mlr.press/v231/zhen24a.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D7-I09 | 最小描述长度清洗 | 通过 | [Contrastive Learning of Graphs under Label Noise](https://doi.org/10.1016/j.neunet.2024.106113) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D7-I10 | 清洗—增强双层博弈 | 通过 | [Prototype-Guided Supervision for Graph Learning with Noisy and Sparse Labels](https://ojs.aaai.org/index.php/AAAI/article/view/39477) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D8-I01 | 双向可信关系蒸馏 | 通过 | [Difficult Examples Hurt Unsupervised Contrastive Learning](https://openreview.net/forum?id=5LMdnUdAoy) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D8-I02 | 双证据路径同伦配对 | v2 重写通过 | [Cycle-Contrast for Self-Supervised Video Representation Learning](https://arxiv.org/abs/2010.14810) | 循环一致检索挖正样本在相邻领域已有先例；已改为语义/拓扑双闭路的离散同伦填充判据。 |
| D8-I03 | 因果交换正样本 | 通过 | [Debiased Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D8-I04 | 双 Beta 关系后验 | 通过 | [Contrastive Learning with Hard Negative Samples](https://openreview.net/forum?id=CR1XOQ0UTh-) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D8-I05 | 正样本生成器—鉴别器互惠 | 通过 | [Bootstrap Graph Representation Learning](https://openreview.net/forum?id=0UXT6PpRpW) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D8-I06 | 顺逆增强等变监督 | 通过 | [Augmentation-Free Self-Supervised Learning on Graphs](https://ojs.aaai.org/index.php/AAAI/article/view/20353) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D8-I07 | 可靠正样本课程反哺 | 通过 | [Mutual Contrastive Learning for Visual Representation](https://ojs.aaai.org/index.php/AAAI/article/view/20211) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D8-I08 | 对偶 OT 配对监督 | 通过 | [Robust Contrastive Multi-view Clustering against Noisy Correspondence](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dbe81b08f7dc4dd8b43bc62dedfd9662-Abstract-Conference.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D8-I09 | 四值逻辑正样本 | 通过 | [CaliGCL: Calibrated Graph Contrastive Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b84cbefc3bebbb88811e2ead03c9ef0c-Abstract-Conference.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D8-I10 | 真假正样本梯度投影 | 条件通过 | [Noise Is Also Useful: Negative Correlation-Steered Latent Contrastive Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Yan_Noise_Is_Also_Useful_Negative_Correlation-Steered_Latent_Contrastive_Learning_CVPR_2022_paper.html) | 未发现完整链同构，但宏观形式接近；必须保留关键连接并做四组消融。 |
| D9-I01 | 链复形对比守恒 | 条件通过 | [Edge Contrastive Learning: An Augmentation-Free GCL Model](https://ojs.aaai.org/index.php/AAAI/article/view/34044) | 未发现完整链同构，但宏观形式接近；必须保留关键连接并做四组消融。 |
| D9-I02 | 跨层 OT 循环 | 通过 | [InfoGraph](https://openreview.net/forum?id=r1lfF2NYvH) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D9-I03 | 信息预算分解 | 通过 | [GCC: Graph Contrastive Coding](https://dl.acm.org/doi/10.1145/3394486.3403168) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D9-I04 | 边作为点—图翻译器 | 通过 | [Sub-Graph Contrast for Scalable Self-Supervised Graph Representation](https://doi.org/10.1109/ICDM50108.2020.00031) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D9-I05 | 梯度流一致跨层对比 | 通过 | [SUGAR: Subgraph Neural Network with Reinforcement Pooling](https://dl.acm.org/doi/10.1145/3442381.3449820) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D9-I06 | 跨层可靠性路由 | 通过 | [Multi-Scale Contrastive Siamese Networks](https://www.ijcai.org/proceedings/2021/0204.pdf) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D9-I07 | 跨层反事实交换 | 通过 | [Multi-Scale Subgraph Contrastive Learning](https://www.ijcai.org/proceedings/2023/0247.pdf) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D9-I08 | 多层原型单纯形 | 通过 | [Contrastive Cross-Scale Graph Knowledge Synergy](https://dl.acm.org/doi/10.1145/3580305.3599271) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D9-I09 | 多分辨率互预测编码 | 条件通过 | [Graph-Aware Contrasting for Multivariate Time-Series](https://ojs.aaai.org/index.php/AAAI/article/view/29501) | 未发现完整链同构，但宏观形式接近；必须保留关键连接并做四组消融。 |
| D9-I10 | 跨层 Koopman 交换 | 通过 | [Improving GCL with Community Structure](https://proceedings.mlr.press/v286/chen25a.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D10-I01 | 图条件 carré-du-champ 度量 | 通过 | [Riemannian Metric Matching for Scalable Geometric Modeling](https://openreview.net/forum?id=KVzXnWPLgX) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D10-I02 | 曲率跃迁链复形压缩 | v2 重写通过 | [SelfMGNN：A Self-Supervised Mixed-Curvature Graph Neural Network](https://ojs.aaai.org/index.php/AAAI/article/view/20333) | 原链的自监督混合曲率乘积空间已有直接先例；已改为曲率跃迁 1-chain 的 Hodge 调和压缩。 |
| D10-I03 | 扩散坐标后再对比 | 通过 | [A Heat Diffusion Perspective on Geodesic Preserving DR](https://openreview.net/forum?id=HNd4qTJxkW) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D10-I04 | 先拓扑分层、后流形降维 | 通过 | [Scaling Riemannian Diffusion Models](https://openreview.net/forum?id=FLTg8uA5xI) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D10-I05 | 高维表示的 Hodge 去噪 | 通过 | [Geometric Wavelet Scattering Networks on Manifolds](https://proceedings.mlr.press/v107/perlmutter20a.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D10-I06 | 半松弛 GW 监督投影头 | 通过 | [The Manifold Scattering Transform for High-Dimensional Point Clouds](https://proceedings.mlr.press/v196/chew22a.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D10-I07 | 热测地—分类测地双度量 | 通过 | [Generalized Dimension Reduction Using Semi-Relaxed GW](https://ojs.aaai.org/index.php/AAAI/article/view/33766) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D10-I08 | 可逆压缩后对比 | 通过 | [Graph Contrastive Learning with Stable and Scalable Spectral Encoding](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8e9a6582caa59fda0302349702965171-Abstract-Conference.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D10-I09 | 散射—可学习残差分解 | 通过 | [Spectral Augmentations for Graph Contrastive Learning](https://proceedings.mlr.press/v206/ghose23a.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D10-I10 | 持久路径坐标压缩 | 通过 | [A Persistent Weisfeiler-Lehman Procedure for Graph Classification](https://proceedings.mlr.press/v97/rieck19a.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D11-I01 | 可靠性反向掩码 | 通过 | [GraphMAE: Self-Supervised Masked Graph Autoencoders](https://dl.acm.org/doi/10.1145/3534678.3539321) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D11-I02 | 点—边—环守恒掩码 | 通过 | [GraphMAE2: A Decoding-Enhanced Masked Self-Supervised GNN](https://dl.acm.org/doi/10.1145/3543507.3583379) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D11-I03 | 共形难度课程掩码 | 通过 | [MaskGAE: Masked Graph Modeling Meets Graph Autoencoders](https://dl.acm.org/doi/10.1145/3511808.3557269) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D11-I04 | 反事实最小充分掩码 | 通过 | [S2GAE: Self-Supervised Graph Autoencoders are Generalizable Learners](https://dl.acm.org/doi/10.1145/3539597.3570404) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D11-I05 | 连续带宽×频带掩码 | 条件通过 | [Bandana：Masked Graph Autoencoder with Non-discrete Bandwidths](https://openreview.net/forum?id=0iwNrRRIiZ) | 未发现完整链同构，但宏观形式接近；必须保留关键连接并做四组消融。 |
| D11-I06 | 掩码博弈 Shapley 平衡 | 通过 | [Rethinking Graph Masked Autoencoders through Alignment and Uniformity](https://ojs.aaai.org/index.php/AAAI/article/view/29479) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D11-I07 | 时间一致记忆掩码 | 通过 | [Masked Graph Autoencoder with Non-discrete Bandwidths](https://openreview.net/forum?id=0iwNrRRIiZ) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D11-I08 | 真假掩码双解码器 | 通过 | [Self-Supervised Masked Graph Autoencoder via Structure-Aware Curriculum](https://proceedings.mlr.press/v267/li25ct.html) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D11-I09 | Möbius 子结构掩码 | 通过 | [Teacher-Guided Edge Discriminator for Personalized GMAE](https://ojs.aaai.org/index.php/AAAI/article/view/33448) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |
| D11-I10 | 掩码恢复的跨尺度证明 | 通过 | [Dynamic and Chemical Constraints for Molecular Masked Graph Autoencoders](https://openreview.net/forum?id=tv2McpEdyH) | 候选仅覆盖部分组件；未发现相同有序完整链与相同联系方式。 |

## 实验准入要求

每个 idea 至少保留 `BalanceGCL`、仅新增组件、去掉关键连接、完整链四组；12 个条件通过项不得只报告完整模型。五个 v2 修订项必须使用带 `revision=v2_external_audit` 的新结果，旧 checkpoint 和旧指标不参与排序。

自动检索原始记录：`outputs/research_ideas/novelty_screen_crossref.json`。
