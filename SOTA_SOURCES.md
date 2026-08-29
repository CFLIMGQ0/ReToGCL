# 新增 SOTA 论文与源码审计

## 论文和官方仓库

2026-08-10 新增的五份 PDF 与官方源码对应关系如下。第三方仓库原样保存在 `src/sota_sources/`，实验适配代码位于 `src/models/sota_graph_models.py`，不会直接修改第三方仓库。

| 方法 | 论文 | 官方仓库 | 本地目录 | 审计提交 |
| --- | --- | --- | --- | --- |
| HISTOGRAPH | Learning from Historical Activations in Graph Neural Networks | [YanivDorGalron/HISTOGRAPH](https://github.com/YanivDorGalron/HISTOGRAPH) | `src/sota_sources/HISTOGRAPH/` | `e6e1d01cb0ffbf145bb06631c8d115f517e9db22` |
| UniImb | One for Two: A Unified Framework for Imbalanced Graph Classification via Dynamic Balanced Prototype | [ALWAYS1815/iclr_UniImb](https://github.com/ALWAYS1815/iclr_UniImb) | `src/sota_sources/UniImb/` | `fa36c7bde799d03dae66af5a3ec70dac85c0c642` |
| DiffLift | Differentiable Lifting for Topological Neural Networks | [JorgeLuizFranco/difflifting](https://github.com/JorgeLuizFranco/difflifting) | `src/sota_sources/DiffLift/` | `899e74a4f857b4fa24d13341547dc8de4b2fcd95` |
| BalanceGCL | Graph Contrastive Learning with Balanced Hard Negatives and Fine-grained Semantic-aware Positives | [YeLiu-Lab/BalanceGCL](https://github.com/YeLiu-Lab/BalanceGCL) | `src/sota_sources/BalanceGCL/` | `6d68ce70eac1b8507fa5d87c3acfc4e4b2c1d68d` |
| Khan-GCL | Khan-GCL: Kolmogorov–Arnold Network Based Graph Contrastive Learning with Hard Negatives | [zihuwang97/KhanGCL](https://github.com/zihuwang97/KhanGCL) | `src/sota_sources/KhanGCL/` | `249b7b18878af7ecfb7461285715d4a5109e388f` |

## 论文方法摘要

- HISTOGRAPH 使用最终层节点表示查询全部历史层激活，先完成逐层注意力，再执行一次节点自注意力；它是监督式图池化/读出模块。
- UniImb 以随机游走和拉普拉斯谱编码描述局部及全局拓扑，通过个性化边/特征扰动扩充样本，并利用动态平衡原型缓解类别及拓扑不平衡。
- DiffLift 根据 GNN 节点嵌入学习候选高阶单元的尺寸与接收概率，再把提升后的超图或复形交给拓扑神经网络端到端训练。
- BalanceGCL 为每个替代类别生成平衡的反事实硬负样本，并把语义区域细分为正贡献、负贡献和无关区域后构造正样本。
- Khan-GCL 用三阶 B 样条 KAN 替换 GCL 编码器中的 MLP，并从 KAN 系数识别独立、可判别维度，在这些关键维度上生成潜空间硬负样本。

## 官方源码可运行性审计

- HISTOGRAPH 的图分类入口固定为十折、350/500 轮并强依赖 W&B；项目适配移除了在线服务依赖并改为统一五折。
- UniImb 的原入口使用绝对日志路径和论文专用不平衡划分；项目适配保留核心模块，但改用与现有基线一致的分层五折。
- DiffLift 的 Docker 环境依赖 TopoBench、TopoModelX、TopoNetX 等完整拓扑学习栈；项目适配采用论文明确给出的可微超图提升实例。
- BalanceGCL 仓库没有依赖文件，入口需要仓库中未发布的预训练权重，且部分模块包含无扩展名源码和缺失导入；无法原样端到端运行。
- Khan-GCL 入口含占位数据路径，并无条件导入多个仓库中不存在的 KAN 层文件；项目适配使用论文指定的三阶 B 样条 KAN 与两类 CKFI 分数。

因此，`outputs/sota/` 中的结果统一标记为 `project_pyg_adapter`，表示“依据论文和官方核心源码完成的当前 PyG 兼容复现”，不表示官方仓库零修改复跑。

## 共同任务和评测协议

五个模型的原生任务交集是图分类，因此统一比较 NCI1、PROTEINS、COLLAB、MUTAG、COLORS-3：

- HISTOGRAPH、UniImb、DiffLift：每折独立端到端训练；外层测试折不参与模型选择，训练折内部再划分分层验证集。
- BalanceGCL、Khan-GCL：在无标签图上预训练编码器，冻结表示后使用项目统一的分层五折线性探针。
- 主指标：Accuracy（%），报告五折均值和样本标准差。
- Cora、CiteSeer、PubMed 不纳入五模型横向表，因为 UniImb、BalanceGCL、Khan-GCL 没有原生节点分类模型。

## 硬件状态

调度器支持 `--devices 0 1 2 3` 四卡并发。当前运行环境实际只检测到两张 RTX 4090（GPU 0、1），因此本机实验使用两卡并发；如果四张卡均可见，可直接改用四卡命令，无需修改训练代码。
