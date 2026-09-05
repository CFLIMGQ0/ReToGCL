# ReToGCL 全模型单点优化实验

## 实验原则

- 统一基线为 `D7-I10 + D7-I01 + D10-I04` 原始 ReToGCL。
- 每个候选只改变一个组件或一条连接，候选之间不累计。
- 编码维度、层数、优化器、学习率、训练轮数、随机种子和五折探针全部固定。
- 六个数据集为 ADHD200、BACE、BBBP、Tox21、AIDS、BZR。

## 十个候选

| 编号 | 唯一修改点 | 修改内容 |
|---|---|---|
| O01 | GIN 连接 | 每层加入等权恒等残差 |
| O02 | 编码器归一化 | BatchNorm 替换为 GraphNorm |
| O03 | 图读出 | 逐层求和替换为逐层均值 |
| O04 | 图读出 | 求和结果除以节点数平方根 |
| O05 | 辅助目标 | 删除 D7-I01 洁净辅助项，保留其表示变换 |
| O06 | 拓扑投影 | D10-I04 表示残差增益从 4 限制为 1 |
| O07 | 困难负样本 | 通过与锚点相同的 D10-I04 度量映射 |
| O08 | 对比目标 | 单向目标替换为双向平均 |
| O09 | 批负样本 | 每个锚点剔除一个最相似非对角负样本 |
| O10 | 对比温度 | 根据当前正对齐置信度逐样本调整温度 |

## 研究依据

- [GraphNorm（ICML 2021）](https://proceedings.mlr.press/v139/cai21e.html) 说明逐图归一化可降低图批次噪声并改善图分类优化。
- [Jumping Knowledge（ICML 2018）](https://proceedings.mlr.press/v80/xu18c.html) 强调不同传播深度信息的保留；本实验保留全部层表示，并单独检查残差与读出尺度。
- [ProGCL（ICML 2022）](https://proceedings.mlr.press/v162/xia22b.html) 指出图对比学习中最相似困难负样本常是假负样本。
- [Debiased Contrastive Learning（NeurIPS 2020）](https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html) 说明无标签批负样本存在采样偏差。
- [Model-Aware Contrastive Learning（ICML 2023）](https://proceedings.mlr.press/v202/huang23c.html) 指出固定温度会造成 uniformity–tolerance 冲突，并使用对齐信息调节温度。

以上候选是针对 ReToGCL 当前实现的最小诊断改动，不宣称复现上述论文的完整模型。
