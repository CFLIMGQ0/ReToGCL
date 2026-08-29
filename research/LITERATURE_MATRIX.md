# 图表示学习十一方向文献问题矩阵

## 使用口径

- 检索截止日期：2026-08-11。
- “顶级论文”优先指 ICLR、ICML、NeurIPS、KDD、WWW、AAAI、IJCAI、AISTATS、UAI、SIGIR、WSDM、TMLR、TKDE、TNNLS 等正式论文；2026 年已公开但仍在审的工作只用于查重，不作为创新有效性的主要证据。
- 每个方向列出 10 篇独立论文，共 110 篇；表中的“剩余问题”是后续构造新方案的入口，并不表示原论文存在错误。
- 阅读重点是论文的问题定义、方法主链、损失或连接机制及局限，而不是只比较最终性能。

## 方向 1：多空间相似度、边界样本与门控

| # | 论文（会议/年份） | 解决的问题与核心创新 | 剩余问题 |
|---:|---|---|---|
| 1 | [Mitigating Local Cohesion and Global Sparseness in GCL with Fuzzy Boundaries](https://openreview.net/forum?id=1b94nzUAvt)（ICML 2025） | 指出局部小团簇和全局稀疏，构造并收缩模糊边界。 | 边界只在单一潜空间定义，缺少结构空间的交叉验证。 |
| 2 | [A New Mechanism for Eliminating Implicit Conflict in GCL](https://ojs.aaai.org/index.php/AAAI/article/view/29125)（AAAI 2024） | 用相似度梯度检测消息传递与 InfoNCE 的隐式冲突并忽略冲突项。 | 忽略冲突没有恢复被丢失的正确关系。 |
| 3 | [ProGCL: Rethinking Hard Negative Mining in GCL](https://proceedings.mlr.press/v162/xia22b.html)（ICML 2022） | 用混合分布估计困难负样本为真负样本的概率。 | 估计主要依赖表示相似度，结构证据有限。 |
| 4 | [A Probability Contrastive Learning Framework for 3D Molecular Representation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6adaf0cbeba11705d4ea67a62044f63d-Abstract-Conference.html)（NeurIPS 2024） | 用贝叶斯权重分布与随机 EM 同时缓解假正、假负样本。 | 概率权重没有显式建模分类边界和多种内在几何。 |
| 5 | [Similarity Preserving Adversarial Graph Contrastive Learning](https://dl.acm.org/doi/10.1145/3580305.3599383)（KDD 2023） | 以相似度保持的对抗扰动生成困难且语义稳定的视图。 | 相似度本身仍由单一表示决定。 |
| 6 | [InfoGCL: Information-Aware Graph Contrastive Learning](https://proceedings.neurips.cc/paper/2021/hash/ff1e68e74c6b16a1a7b5d958b95e120c-Abstract.html)（NeurIPS 2021） | 用信息瓶颈刻画最优视图与最优编码器。 | 信息量原则不能直接判断单个样本对是否可靠。 |
| 7 | [Adversarial Graph Augmentation to Improve GCL](https://proceedings.neurips.cc/paper_files/paper/2021/hash/854f1fb6f65734d9e49f708d6cd84ad6-Abstract.html)（NeurIPS 2021） | 学习最具挑战但保留任务信息的边删减增强。 | 对抗增强可能在边界样本上破坏稀有结构。 |
| 8 | [Graph Contrastive Learning with Augmentations](https://proceedings.neurips.cc/paper/2020/hash/3fe230348e9a12c13120749e3f9fa4cd-Abstract.html)（NeurIPS 2020） | 系统建立图级增强与对比学习范式。 | 固定增强组合无法按图或样本对自适应。 |
| 9 | [Unified Graph Augmentations for Generalized Contrastive Learning on Graphs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/41efc12982eca6f8bb5e48dc3a84b843-Abstract-Conference.html)（NeurIPS 2024） | 从消息传递角度统一节点、边、属性和子图增强，并约束一致性与多样性。 | 统一增强不等于统一相似度，也未显式修复分类空间错位。 |
| 10 | [Teacher-Guided Graph Contrastive Learning](https://openreview.net/forum?id=15tjpSHI15)（TMLR 2024） | 用教师的软伪标签替代所有增强对都“同等为正”的硬假设。 | 教师与学生共享表示偏差时容易相互确认错误。 |

## 方向 2：环、胞腔、单纯形与高阶提升

| # | 论文（会议/年份） | 解决的问题与核心创新 | 剩余问题 |
|---:|---|---|---|
| 1 | [Weisfeiler and Lehman Go Cellular: CW Networks](https://proceedings.neurips.cc/paper/2021/hash/157792e4abb490f99dbd738483e0d2d4-Abstract.html)（NeurIPS 2021） | 将图按环提升为正规胞腔复形，增强表达力并缩短节点距离。 | 环大小阈值固定，所有环被近似同等对待。 |
| 2 | [Weisfeiler and Lehman Go Topological: MPSN](https://arxiv.org/abs/2103.03212)（ICLR GTRL 2021） | 在单纯形复形上定义高阶消息传递并给出 SWL 表达力。 | clique 提升可能产生组合爆炸且不能自然表达非团环。 |
| 3 | [Learning From Simplicial Data Based on Random Walks and 1D Convolutions](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fa30825df7ceaba452d5533538ea29c2-Abstract-Conference.html)（ICLR 2024） | SCRaWl 用单纯形随机游走与一维卷积控制计算成本。 | 随机游走会稀释罕见但判别性强的环。 |
| 4 | [Weisfeiler and Lehman Go Paths](https://ojs.aaai.org/index.php/AAAI/article/view/29463)（AAAI 2024） | 将图提升到路径复形，统一部分单纯形与胞腔方法。 | 路径冗余高，缺少任务相关的路径筛选。 |
| 5 | [Topological Neural Networks go Persistent, Equivariant, and Continuous](https://proceedings.mlr.press/v235/verma24a.html)（ICML 2024） | TopNets 统一持久同调与高阶消息传递并支持几何等变性。 | 持久特征通常作为附加描述符，和环消息的因果联系弱。 |
| 6 | [Demystifying Topological Message-Passing with Relational Structures](https://openreview.net/forum?id=QC2qE1tcmd)（ICLR 2025） | 用关系结构统一图和高阶消息传递并分析、缓解过挤压。 | 重连目标侧重传输，不区分有益环与噪声环。 |
| 7 | [TopInG: Topologically Interpretable Graph Learning](https://proceedings.mlr.press/v267/xin25b.html)（ICML 2025） | 用持久理由过滤生成可解释子图并施加拓扑差异约束。 | 理由过滤和高阶消息传递没有闭环共同优化。 |
| 8 | [Contraction and Hourglass Persistence](https://openreview.net/forum?id=mWN6cpA6Wr)（ICLR 2026） | 交替包含与收缩，构造比单向过滤更强的 Hourglass Persistence。 | 尚未按分类不确定性自适应选择收缩路径。 |
| 9 | [Copresheaf Topological Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2025/hash/dc62cd4ec77f3bbec8e6245d0bd91d08-Abstract-Conference.html)（NeurIPS 2025） | 用余层统一方向性、各向异性的拓扑消息。 | 余层映射参数多，缺少轻量的图级环实例化。 |
| 10 | [Directed Semi-Simplicial Learning](https://openreview.net/forum?id=YR3CNvFfCr)（ICLR 2026） | 以半单纯形集合编码有向高阶 motif 及方向关系。 | TU 无向图上的方向应如何可靠诱导仍未解决。 |

## 方向 3：多视图、多尺度节点/边超图与对比学习

| # | 论文（会议/年份） | 解决的问题与核心创新 | 剩余问题 |
|---:|---|---|---|
| 1 | [TriCL: Tri-directional Contrastive Learning on Hypergraphs](https://ojs.aaai.org/index.php/AAAI/article/view/26019)（AAAI 2023） | 同时对比节点—节点、超边—超边、节点—超边。 | 单一超图尺度，未表达环大小形成的层级。 |
| 2 | [Augmentations in Hypergraph Contrastive Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0cd1eec0eeaf5ce1bf6d8875a7c1d095-Abstract-Conference.html)（NeurIPS 2022） | 提出人工和生成式超边增强，显示超边增强最关键。 | 生成器没有显式约束跨尺度拓扑一致性。 |
| 3 | [Hypergraph Contrastive Collaborative Filtering](https://dl.acm.org/doi/10.1145/3477495.3532058)（SIGIR 2022） | 用超图增强全局协同关系，并与局部图视图对比。 | 高阶关系由协同数据定义，缺少结构环语义。 |
| 4 | [Self-Supervised Multi-Channel Hypergraph Convolutional Network](https://dl.acm.org/doi/10.1145/3442381.3449940)（WWW 2021） | 构造多通道超图并用自监督任务协调通道。 | 通道融合是预设的，尺度可靠性不可辨识。 |
| 5 | [Hypergraph-Enhanced Contrastive MVC with Hyper-Laplacian](https://openreview.net/forum?id=L5VPvuPMKs)（NeurIPS 2025） | 结合跨视图、超边内对比和超拉普拉斯流形保持。 | 只约束同尺度超边内关系，跨尺度对应关系未建模。 |
| 6 | [Hypergraph-Based Multi-View Multi-Label Classification via Adaptive High-Order Semantic Fusion](https://ojs.aaai.org/index.php/AAAI/article/view/39719)（AAAI 2026） | 建立视图内语义超边和跨视图超边，再做原型对比。 | 依赖标签原型，不适合完全自监督图分类预训练。 |
| 7 | [Heterophily-aware Contrastive Learning for Heterophilic Hypergraphs](https://ojs.aaai.org/index.php/AAAI/article/view/39473)（AAAI 2026） | 多跳编码配合标签感知与结构感知的多粒度对比。 | 多粒度来自 hop，而非高阶单元的尺度谱。 |
| 8 | [Dynamic Hypergraph Structure Learning for Traffic Flow Forecasting](https://doi.org/10.1109/ICDE55515.2023.00179)（ICDE 2023） | 端到端动态学习超边以适应时变高阶关系。 | 动态超边缺少拓扑稳定性和嵌套约束。 |
| 9 | [Deep Heterogeneous Contrastive Hyper-Graph Learning](https://openreview.net/forum?id=HOwJ5YfJQh)（IMWUT 2023） | 三类异质子超图用专门卷积，并以对比约束节点异质性。 | 专用分支难共享统计强度，视图数增加时扩展性差。 |
| 10 | [Hypergraph Learning for Unsupervised Graph Alignment via Optimal Transport](https://ojs.aaai.org/index.php/AAAI/article/view/35498)（AAAI 2025） | 用最优传输自适应学习高阶超边，并用 Dirichlet 能量协调结构与特征。 | 只对单层超图求传输，未处理多尺度超边间的输运。 |

## 方向 4：ABC、AB、AC、BC 等多视图多融合

| # | 论文（会议/年份） | 解决的问题与核心创新 | 剩余问题 |
|---:|---|---|---|
| 1 | [Self-Weighted Contrastive Learning among Multiple Views](https://proceedings.neurips.cc/paper_files/paper/2023/hash/03b13b0db740b95cb741e007178ef5e5-Abstract-Conference.html)（NeurIPS 2023） | 按视图对差异自加权并用重构抑制表示退化。 | 只学习边权，未显式建模高阶视图子集的协同或冗余。 |
| 2 | [Contrastive Multi-View Representation Learning on Graphs](https://proceedings.mlr.press/v119/hassani20a.html)（ICML 2020） | MVGRL 对比局部/全局和邻接/扩散两个结构视图。 | 两视图设定不能覆盖 ABC 与成对子集的交互。 |
| 3 | [CoCo: A Coupled Contrastive Framework](https://proceedings.mlr.press/v202/yin23a.html)（ICML 2023） | GNN 与层次图核分别隐式、显式建模拓扑，再耦合对比。 | 分支耦合固定，不能根据样本改变组合路径。 |
| 4 | [AF-UMC: Alignment-Free Fusion for Unaligned Multi-View Clustering](https://proceedings.neurips.cc/paper_files/paper/2025/hash/965484d5b2b2624ba17295612a3ba7e8-Abstract-Conference.html)（NeurIPS 2025） | 在共同基空间直接提取一致表示，绕过不可靠的逐样本对齐。 | 一致基可能吞掉稀有但互补的视图特有信息。 |
| 5 | [MMPG: MoE-based Adaptive Multi-Perspective Graph Fusion](https://ojs.aaai.org/index.php/AAAI/article/view/37096)（AAAI 2026） | 专家自动分化为单视图、两两协同和全局共识。 | 专家路由不保证组合覆盖，也没有组合复杂度惩罚。 |
| 6 | [Reliable Conflictive Multi-View Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29546)（AAAI 2024） | 学习视图证据并用冲突意见聚合区分共性与特有可靠性。 | 证据融合只在决策端发生，不能反向塑造编码器视图。 |
| 7 | [Deep Fuzzy Multi-View Learning for Reliable Classification](https://openreview.net/forum?id=ZzuaeYvLsJ)（ICML 2025） | 以可能性/必然性及双可靠融合处理冲突视图。 | 没有结构化搜索 ABC/AB/AC/BC 组合。 |
| 8 | [Self-supervised Trusted Contrastive Multi-view Clustering](https://ojs.aaai.org/index.php/AAAI/article/view/33902)（AAAI 2025） | Dirichlet 证据与 Dempster-Shafer 融合产生不确定性，反哺伪标签对比。 | DS 融合在高冲突时可能数值极化。 |
| 9 | [Adaptive Evolutionary Fusion for Multi-View Clustering](https://ojs.aaai.org/index.php/AAAI/article/view/40110)（AAAI 2026） | 用进化算法搜索树形融合，而非预设串联/拼接。 | 离散搜索成本高且缺少图结构归纳偏置。 |
| 10 | [What Makes for Good Views for Contrastive Learning?](https://proceedings.neurips.cc/paper/2020/hash/4c2e5eaae9152079b9e95845750bb9ab-Abstract.html)（NeurIPS 2020） | 给出保留任务信息同时降低视图互信息的“甜点区”原则。 | 原则不能直接判断多于两个视图的最优子集。 |

## 方向 5：多粒度可靠性及其权重

| # | 论文（会议/年份） | 解决的问题与核心创新 | 剩余问题 |
|---:|---|---|---|
| 1 | [Uncertainty Quantification over Graph with Conformalized GNNs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html)（NeurIPS 2023） | 建立图数据上的保覆盖共形预测，并用拓扑修正缩小预测集。 | 只校准最终预测，没有给每个粒度分配可靠性。 |
| 2 | [What Makes Graph Neural Networks Miscalibrated?](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5975754c7650dfee0682e06e1fec0522-Abstract-Conference.html)（NeurIPS 2022） | 识别五类失校准因素，提出节点级图注意温度缩放。 | 温度缩放不解释哪一结构尺度导致失校准。 |
| 3 | [Be Confident! Towards Trustworthy GNNs via Confidence Calibration](https://proceedings.neurips.cc/paper/2021/hash/c7a9f13a6c0940277d46706c7ca32601-Abstract.html)（NeurIPS 2021） | 发现 GNN 常欠置信，并用拓扑感知校准 GNN。 | 依赖置信同质性，对异质图和图级批次不稳。 |
| 4 | [Accurate and Scalable Estimation of Epistemic Uncertainty for GNNs](https://openreview.net/forum?id=ZL6yd6N1S2)（ICLR 2024） | 以距离感知确定性网络高效估计分布偏移下认知不确定性。 | 统一不确定度无法定位节点、边、子图哪一级失效。 |
| 5 | [Evidential Uncertainty Probes for GNNs](https://proceedings.mlr.press/v258/yu25a.html)（AISTATS 2025） | 用轻量证据探针为冻结 GNN 分离认知/偶然不确定性。 | 探针是读出端附加头，不能主动调整多粒度传播。 |
| 6 | [Similarity-Navigated Conformal Prediction for Graphs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/571c7e164fb1ffbcf2f84a63784451ec-Abstract-Conference.html)（NeurIPS 2024） | 利用图相似邻域构造更局部、更高效的共形校准。 | 相似邻域本身可能在错误空间中定义。 |
| 7 | [Trusted Multi-View Deep Learning with Opinion Aggregation](https://ojs.aaai.org/index.php/AAAI/article/view/20724)（AAAI 2022） | 将每视图输出建模为主观意见，再可靠聚合。 | 意见可靠性与表示粒度没有因果约束。 |
| 8 | [Enhancing Multi-View Classification Reliability with Adaptive Rejection](https://ojs.aaai.org/index.php/AAAI/article/view/34088)（AAAI 2025） | 以分布无关的自适应拒绝阻断低质量视图。 | 硬拒绝浪费部分可用信息，不能细化到图内部粒度。 |
| 9 | [Training Uncertainty-Aware Classifiers with Conformalized Deep Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8c96b559340daa7bb29f56ccfbbc9c2f-Abstract-Conference.html)（NeurIPS 2022） | 将共形思想写入训练目标以抑制过置信。 | i.i.d. 假设下的目标未显式适配图结构与多尺度依赖。 |
| 10 | [Diagnostic Uncertainty Calibration](https://proceedings.mlr.press/v130/mimori21a.html)（AISTATS 2021） | 校准类别概率估计的高阶统计与标注者分歧。 | 校准对象是输出分布，不覆盖结构证据间的冲突。 |

## 方向 6：多个增强正样本聚合成综合正样本

| # | 论文（会议/年份） | 解决的问题与核心创新 | 剩余问题 |
|---:|---|---|---|
| 1 | [SwAV: Unsupervised Learning by Contrasting Cluster Assignments](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html)（NeurIPS 2020） | 通过在线聚类和跨视图交换预测，用原型替代大量样本对。 | 单个原型会平均掉同一实例多视图中的互补信息。 |
| 2 | [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)（NeurIPS 2020） | 将同类的多个样本同时作为正样本。 | 标签正样本集合不等价于实例的多视图综合正样本。 |
| 3 | [NNCLR: Nearest-Neighbor Contrastive Learning](https://proceedings.neurips.cc/paper/2021/hash/14f2ebeab937ca128186e7ba876faef9-Abstract.html)（NeurIPS 2021） | 用支持集中的最近邻代替增强视图作为正样本。 | 最近邻选择是离散的，易把边界邻居误并入。 |
| 4 | [Prototypical Contrastive Learning of Unsupervised Representations](https://openreview.net/forum?id=KmykpuSrjcq)（ICLR 2021） | 多粒度聚类生成潜在类别原型并执行原型对比。 | 原型表示群体中心，不保留单实例各增强的合成轨迹。 |
| 5 | [You Never Cluster Alone](https://proceedings.neurips.cc/paper/2021/hash/e96ed478dab8595a7dbda4cbcbee168f-Abstract.html)（NeurIPS 2021） | 协同学习样本特征与聚类原型，避免聚类退化。 | 聚合仍是类级而不是视图条件的实例级。 |
| 6 | [Graph Communal Contrastive Learning](https://dl.acm.org/doi/10.1145/3485447.3512207)（WWW 2022） | 以社区作为中间语义单元进行图对比。 | 社区正样本无法表达跨增强的高阶合成关系。 |
| 7 | [CLEAR: Cluster-Enhanced Contrast for Graph Representation](https://doi.org/10.1109/TNNLS.2022.3177775)（TNNLS 2022） | 用聚类伪标签扩展图级正样本集合。 | 伪标签错误会一次性污染整个正样本集合。 |
| 8 | [Unsupervised Graph-Level Representation Learning with Hierarchical Contrasts](https://doi.org/10.1016/j.neunet.2022.10.016)（Neural Networks 2023） | 在节点、子图、图的层级之间联合对比。 | 多层正样本独立进入损失，未生成统一的综合视图。 |
| 9 | [Hierarchically Contrastive Hard Sample Mining for Graph Pretraining](https://doi.org/10.1109/TNNLS.2023.3297607)（TNNLS 2023） | 在不同层级挖掘困难样本以加强预训练。 | 重点在困难度，未约束综合正样本保留各视图的独有成分。 |
| 10 | [Contrastive Learning Meets Homophily: Two Birds with One Stone](https://proceedings.mlr.press/v202/he23c.html)（ICML 2023） | 联合更新邻居集合，使消息传递与正样本发现相互促进。 | 邻居正样本仍逐个使用，没有学习集合到单一综合正样本的映射。 |

## 方向 7：自动识别干净/脏数据并动态降权

| # | 论文（会议/年份） | 解决的问题与核心创新 | 剩余问题 |
|---:|---|---|---|
| 1 | [NoisyGL: A Comprehensive Benchmark for GNNs under Label Noise](https://proceedings.neurips.cc/paper_files/paper/2024/hash/436ffa18e7e17be336fd884f8ebb5748-Abstract-Datasets_and_Benchmarks_Track.html)（NeurIPS 2024） | 统一评测 17 种方法，揭示成对噪声和稀疏图中的传播危害。 | 主要是节点标签噪声，图级样本的多维局部污染仍缺标准协议。 |
| 2 | [Mitigating Label Noise on Graphs via Topological Sample Selection](https://proceedings.mlr.press/v235/wu24ae.html)（ICML 2024） | 以拓扑信息保留边界样本并选择干净节点。 | 把每个节点判成相对整体的干净/噪声，未允许不同维度部分污染。 |
| 3 | [GraphCleaner: Detecting Mislabelled Samples in Graph Benchmarks](https://proceedings.mlr.press/v202/li23ai.html)（ICML 2023） | 用类别条件图传播检测和修正基准中的错标节点。 | 依赖少量已知干净标签且面向单大图节点。 |
| 4 | [NRGNN: Learning a Label Noise Resistant GNN](https://dl.acm.org/doi/10.1145/3447548.3467364)（KDD 2021） | 预测可靠伪标签和边，缓解稀疏、噪声监督。 | 结构扩充可能把局部污染传播得更远。 |
| 5 | [Robust Training of GNNs via Noise Governance](https://dl.acm.org/doi/10.1145/3539597.3570410)（WSDM 2023） | 估计标签可信度并治理噪声传播。 | 信任分数主要由单模型产生，缺乏独立证据校验。 |
| 6 | [OMG: Towards Effective Graph Classification Against Label Noise](https://doi.org/10.1109/TKDE.2023.3271677)（TKDE 2023） | 为图级标签噪声设计鲁棒图分类与样本处理。 | 关注整体错标，不能处理“标签对但某一结构/属性视图脏”。 |
| 7 | [GDeR: Prototypical Graph Pruning](https://openreview.net/forum?id=O97BzlN9Wh)（NeurIPS 2024） | 用动态原型软裁剪兼顾效率、类别平衡和噪声鲁棒性。 | 原型距离易遗漏少数类的合法离群结构。 |
| 8 | [United We Stand, Divided We Fall: N2G for Robust Graph Classification](https://proceedings.mlr.press/v231/zhen24a.html)（LoG 2024） | 将网络集合抽象为更高层图，提高图标签腐败下的鲁棒性。 | 抽象依赖群体关系，单样本内部污染定位不足。 |
| 9 | [Contrastive Learning of Graphs under Label Noise](https://doi.org/10.1016/j.neunet.2024.106113)（Neural Networks 2024） | 用半监督对比降低噪声标签对图表示的影响。 | 干净/脏的判断仍受早期表示质量影响。 |
| 10 | [Prototype-Guided Supervision for Graph Learning with Noisy and Sparse Labels](https://ojs.aaai.org/index.php/AAAI/article/view/39477)（AAAI 2026） | 以类原型监督替代直接使用可能污染的节点标签。 | 原型也可能由系统性噪声偏移，缺少反事实检查。 |

## 方向 8：假正样本与真正样本双向监督

| # | 论文（会议/年份） | 解决的问题与核心创新 | 剩余问题 |
|---:|---|---|---|
| 1 | [Difficult Examples Hurt Unsupervised Contrastive Learning](https://openreview.net/forum?id=5LMdnUdAoy)（ICLR 2026） | 理论说明无监督对比中的边界困难样本会伤害泛化，并提出移除机制。 | 移除困难样本没有利用其可恢复信息。 |
| 2 | [Discovering Global False Negatives On the Fly](https://proceedings.mlr.press/v267/balmaseda25a.html)（ICML 2025） | 为每个锚点在线学习全局假负阈值，复杂度不随数据集增长。 | 只纠正负样本，不处理增强导致的假正样本。 |
| 3 | [Debiased Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html)（NeurIPS 2020） | 对潜在假负样本进行无监督偏差校正。 | 使用全局类先验，不能实现样本对级双向监督。 |
| 4 | [Contrastive Learning with Hard Negative Samples](https://openreview.net/forum?id=CR1XOQ0UTh-)（ICLR 2021） | 通过重要性采样合成更困难负样本。 | 困难度与真实性未分离，图数据上假负风险更高。 |
| 5 | [Bootstrap Graph Representation Learning](https://openreview.net/forum?id=0UXT6PpRpW)（ICLR 2022） | BGRL 用在线/目标网络去除显式负样本。 | 避开假负却仍假设同实例增强一定是真正样本。 |
| 6 | [Augmentation-Free Self-Supervised Learning on Graphs](https://ojs.aaai.org/index.php/AAAI/article/view/20353)（AAAI 2022） | AFGRL 用局部与全局语义发现正样本，不依赖随机增强。 | 邻居发现和编码器共偏，容易自我确认。 |
| 7 | [Mutual Contrastive Learning for Visual Representation](https://ojs.aaai.org/index.php/AAAI/article/view/20211)（AAAI 2022） | 多网络互相蒸馏完整的对比分布。 | 网络间交换的是分布，不是“真/假正样本”的可解释双向纠错。 |
| 8 | [Robust Contrastive Multi-view Clustering against Noisy Correspondence](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dbe81b08f7dc4dd8b43bc62dedfd9662-Abstract-Conference.html)（NeurIPS 2024） | CANDY 用上下文语义和谱去噪同时处理跨视图假正、假负。 | 跨视图对应去噪尚未迁移到图结构增强的配对机制。 |
| 9 | [CaliGCL: Calibrated Graph Contrastive Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b84cbefc3bebbb88811e2ead03c9ef0c-Abstract-Conference.html)（NeurIPS 2025） | 校准图对比并把同质邻居纳入附加正样本。 | 校准是单向修正，未让可靠真实对反过来监督候选对生成器。 |
| 10 | [Noise Is Also Useful: Negative Correlation-Steered Latent Contrastive Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Yan_Noise_Is_Also_Useful_Negative_Correlation-Steered_Latent_Contrastive_Learning_CVPR_2022_paper.html)（CVPR 2022） | 标签空间筛样本、度量空间挖掘噪声的负相关，并以跨空间相似一致性连接两者。 | 两空间仍共享单向筛选结果，缺少防止双方共同确认错误的锚点。 |

## 方向 9：点、边、子图、图之间的跨层对比

| # | 论文（会议/年份） | 解决的问题与核心创新 | 剩余问题 |
|---:|---|---|---|
| 1 | [Edge Contrastive Learning: An Augmentation-Free GCL Model](https://ojs.aaai.org/index.php/AAAI/article/view/34044)（AAAI 2025） | 从端点拼接生成边表示，显式执行边—边对比。 | 边表示未和节点、图表示组成闭环一致性。 |
| 2 | [InfoGraph](https://openreview.net/forum?id=r1lfF2NYvH)（ICLR 2020） | 最大化图表示与不同尺度子结构表示之间的互信息。 | 主要是局部—全局两级，边只被隐含使用。 |
| 3 | [GCC: Graph Contrastive Coding](https://dl.acm.org/doi/10.1145/3394486.3403168)（KDD 2020） | 以随机游走子图作为通用结构实例完成跨图预训练。 | 子图采样无法保证覆盖关键边或环。 |
| 4 | [Sub-Graph Contrast for Scalable Self-Supervised Graph Representation](https://doi.org/10.1109/ICDM50108.2020.00031)（ICDM 2020） | 对比中心节点、子图和上下文，兼顾可扩展性。 | 层级间以固定目标相连，没有方向可靠性。 |
| 5 | [SUGAR: Subgraph Neural Network with Reinforcement Pooling](https://dl.acm.org/doi/10.1145/3442381.3449820)（WWW 2021） | 强化学习选择子图并最大化子图—全图互信息。 | 选择器可能只偏向易分类区域，遗漏边界结构。 |
| 6 | [Multi-Scale Contrastive Siamese Networks](https://www.ijcai.org/proceedings/2021/0204.pdf)（IJCAI 2021） | 同时建立节点、局部和全局尺度的孪生对比。 | 多尺度损失并列，缺乏显式的点→边→图信息守恒。 |
| 7 | [Multi-Scale Subgraph Contrastive Learning](https://www.ijcai.org/proceedings/2023/0247.pdf)（IJCAI 2023） | 通过多尺度子图构建细粒度结构对比。 | 仍以子图为中介，边级语义没有独立监督。 |
| 8 | [Contrastive Cross-Scale Graph Knowledge Synergy](https://dl.acm.org/doi/10.1145/3580305.3599271)（KDD 2023） | 在不同图尺度之间协同迁移知识。 | 协同是表示对齐，未检查跨层信息是否相互矛盾。 |
| 9 | [Graph-Aware Contrasting for Multivariate Time-Series](https://ojs.aaai.org/index.php/AAAI/article/view/29501)（AAAI 2024） | 用节点/边增强及节点—图对比保持空间一致性。 | 缺少显式边编码器和边—图对比。 |
| 10 | [Improving GCL with Community Structure](https://proceedings.mlr.press/v286/chen25a.html)（UAI 2025） | 用社区替代穷举节点对，获得低秩高效的中粒度对比。 | 社区是软中尺度，未与微观边和宏观图联合校准。 |

## 方向 10：高维流形整理、几何保持与降维

| # | 论文（会议/年份） | 解决的问题与核心创新 | 剩余问题 |
|---:|---|---|---|
| 1 | [Riemannian Metric Matching for Scalable Geometric Modeling](https://openreview.net/forum?id=KVzXnWPLgX)（ICML 2026） | 学习 carré du champ 算子的神经代理，在高维中摊销估计内在黎曼度量。 | 面向一般样本流形，尚未把图内拓扑作为度量条件。 |
| 2 | [Hyperbolic Diffusion Embedding and Distance](https://proceedings.mlr.press/v202/lin23a.html)（ICML 2023） | 将扩散几何的多尺度密度嵌入双曲空间，恢复层级距离。 | 预设树状层级倾向，平坦或混合曲率图可能失真。 |
| 3 | [A Heat Diffusion Perspective on Geodesic Preserving DR](https://openreview.net/forum?id=HNd4qTJxkW)（NeurIPS 2023） | 从黎曼热核推导可保持测地距离的热测地不相似度。 | 全局带宽难适配不同图样本的局部密度。 |
| 4 | [Scaling Riemannian Diffusion Models](https://openreview.net/forum?id=FLTg8uA5xI)（NeurIPS 2023） | 利用对称空间提高流形扩散的精度与高维可扩展性，并缓解投影头坍塌。 | 扩散生成成本仍高，不适合直接遍历每个小图。 |
| 5 | [Geometric Wavelet Scattering Networks on Manifolds](https://proceedings.mlr.press/v107/perlmutter20a.html)（MSML 2020） | 构造对局部等距不变且对形变稳定的流形散射。 | 固定滤波器不能按下游类别调整频带。 |
| 6 | [The Manifold Scattering Transform for High-Dimensional Point Clouds](https://proceedings.mlr.press/v196/chew22a.html)（TAG-ML 2022） | 用扩散图数值实现高维点云上的流形散射。 | 依赖邻图质量，高维近邻失真会进入全部尺度。 |
| 7 | [Generalized Dimension Reduction Using Semi-Relaxed GW](https://ojs.aaai.org/index.php/AAAI/article/view/33766)（AAAI 2025） | 以半松弛 Gromov-Wasserstein 将点云嵌入任意目标度量空间。 | 未同时保持标签边界和图的组合结构。 |
| 8 | [Graph Contrastive Learning with Stable and Scalable Spectral Encoding](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8e9a6582caa59fda0302349702965171-Abstract-Conference.html)（NeurIPS 2023） | EigenMLP 消除特征向量旋转/反射不稳定，并对比空间与谱视图。 | 谱和空间只做一致性，未区分应保留的互补成分。 |
| 9 | [Spectral Augmentations for Graph Contrastive Learning](https://proceedings.mlr.press/v206/ghose23a.html)（AISTATS 2023） | 以谱裁剪、频率重排和随机游走对齐生成结构视图。 | 离散增强可能跨过类别相关的拓扑相变点。 |
| 10 | [A Persistent Weisfeiler-Lehman Procedure for Graph Classification](https://proceedings.mlr.press/v97/rieck19a.html)（ICML 2019） | 用持久同调补足 WL 无法捕获连通分量和环的问题。 | 核特征与端到端高维表示缺少双向耦合。 |

## 方向 11：掩码策略、掩码对象与重构目标

| # | 论文（会议/年份） | 解决的问题与核心创新 | 剩余问题 |
|---:|---|---|---|
| 1 | [GraphMAE: Self-Supervised Masked Graph Autoencoders](https://dl.acm.org/doi/10.1145/3534678.3539321)（KDD 2022） | 以节点特征掩码、重掩码解码和缩放余弦误差建立图掩码范式。 | 随机节点掩码没有利用图级判别结构。 |
| 2 | [GraphMAE2: A Decoding-Enhanced Masked Self-Supervised GNN](https://dl.acm.org/doi/10.1145/3543507.3583379)（WWW 2023） | 多视图随机重掩码与潜变量预测提高解码和泛化。 | 多视图掩码仍未对样本难度和可靠性校准。 |
| 3 | [MaskGAE: Masked Graph Modeling Meets Graph Autoencoders](https://dl.acm.org/doi/10.1145/3511808.3557269)（CIKM 2022） | 掩蔽边并执行路径/边重构，避免普通 GAE 的恒等捷径。 | 二值边掩码会完全阻断消息流。 |
| 4 | [S2GAE: Self-Supervised Graph Autoencoders are Generalizable Learners](https://dl.acm.org/doi/10.1145/3539597.3570404)（WSDM 2023） | 用掩蔽边重构和多层表示提升迁移能力。 | 掩码比例与图结构复杂度无关。 |
| 5 | [Rethinking Tokenizer and Decoder in Masked Graph Modeling](https://proceedings.neurips.cc/paper_files/paper/2023/hash/51fd9a7d1706023cb9f8210cc6ac357c-Abstract-Conference.html)（NeurIPS 2023） | SimSGT 表明子图 tokenizer、强解码器和重掩码比只改编码器更重要。 | 化学子图 tokenizer 难直接迁移到无领域知识的 TU 图。 |
| 6 | [Rethinking Graph Masked Autoencoders through Alignment and Uniformity](https://ojs.aaai.org/index.php/AAAI/article/view/29479)（AAAI 2024） | 将掩码前后对齐和表示均匀性显式加入 GraphMAE。 | 全局均匀性可能排斥合法的类内多模态。 |
| 7 | [Masked Graph Autoencoder with Non-discrete Bandwidths](https://openreview.net/forum?id=0iwNrRRIiZ)（WWW 2024） | 用连续带宽掩码代替 Bernoulli 边删除，并逐层预测带宽。 | 连续掩码只控制消息量，不辨别信息来源是否可靠。 |
| 8 | [Self-Supervised Masked Graph Autoencoder via Structure-Aware Curriculum](https://proceedings.mlr.press/v267/li25ct.html)（ICML 2025） | 估计边重构难度并按由易到难的课程进行掩码。 | 单调课程不能适应训练中难度次序反转。 |
| 9 | [Teacher-Guided Edge Discriminator for Personalized GMAE](https://ojs.aaai.org/index.php/AAAI/article/view/33448)（AAAI 2025） | 教师区分同质/异质边，再为每图个性化掩码、编码和重构。 | 教师误差会直接决定掩码，缺少无教师反事实校验。 |
| 10 | [Dynamic and Chemical Constraints for Molecular Masked Graph Autoencoders](https://openreview.net/forum?id=tv2McpEdyH)（NeurIPS 2025） | 以图信息瓶颈动态调节掩码并使用化学软标签重构。 | 依赖化学约束；一般图上需要可学习但不泄漏标签的替代约束。 |

## 跨方向归纳出的共同问题

1. 多数方法只在单一潜空间判断关系，空间本身错误时会自我确认。
2. 多视图研究通常只学习视图权重，没有显式搜索并校准视图子集及其交互项。
3. 高阶提升大多把“是否提升、提升到多大尺度”当作预处理，而不是样本条件决策。
4. 多粒度损失经常并列相加，缺少跨粒度的信息守恒、冲突定位和因果方向。
5. 噪声学习多把整个样本判为干净或污染，不能表达“某个样本只在某个视图/尺度上脏”。
6. 多正样本通常分别进入分母/分子或汇聚成类原型，缺少保持互补性的实例级综合正样本。
7. 假正/假负校正常是单向筛选，缺少候选关系与可靠关系之间有防坍塌约束的双向教学。
8. 可靠性通常是最终输出的置信度，未形成节点—边—环—图的可靠性张量。
9. 高维几何方法与图结构方法往往并行使用，缺少由图拓扑调制内在度量、再反向调整图传播的闭环。
10. 掩码策略大多选择“遮什么”，很少同时学习“遮多少、向谁遮、何时恢复、用哪个尺度验证恢复”。
