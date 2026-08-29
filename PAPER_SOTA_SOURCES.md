# 十篇论文、源码与实验适配说明

## 论文和源码对应关系

| 实验名 | 论文核心方法 | 官方源码目录 | 审计提交 |
| --- | --- | --- | --- |
| `maxcutpool` | 特征感知、可微 MaxCut 图池化 | `src/sota_sources/MaxCutPool/` | `109b50b929bcd42c8b7a4bf4045d32fe28e7bf7d` |
| `simplicial_mp` | 将图提升为关系结构后进行胞腔消息传递 | `src/sota_sources/Simplicial-Oversquashing/` | `28a9ca589db43b35193906527d10ce02ff921886` |
| `del` | 用多种图布局形成的边距离分布增强消息传递 | `src/sota_sources/DEL_Implementation/` | `bc53614708784f9d7f6fc0617c4dbcba1db1aa45` |
| `cellclat` | 胞腔对比学习与可学习冗余裁剪 | `src/sota_sources/CellCLAT/` | `960b14d7aeb4328c70c2dbd24149f14e369d8adc` |
| `plstm` | DAG 上的 Source、Transition、Mark 门控传播 | `src/sota_sources/pLSTM_Experiments/`、`pLSTM_Core/` | `ac539b6…`、`613fa27…` |
| `abstaingnn` | 预测函数与图级拒识函数的两阶段训练 | `src/sota_sources/AbstainGNN/` | `7c533b3099f7a0e0678843c3d677708d438f4778` |
| `edgeprompt` | 冻结预训练 GNN，通过边提示适配下游任务 | `src/sota_sources/EdgePrompt/` | `fa3d7f4cf42054c4ed88067cb49ae64b31f4ce8e` |
| `dualprism` | 从谱域视角构造图数据增强 | `src/sota_sources/DualPrism/` | `01b26b3b7dcfd92fd33158287d6ad7671b66b845` |
| `invgnn` | 可逆 1×1 变换与图传播交替堆叠 | `src/sota_sources/InvGNN/` | `c33cb6f39ab2ac71e017d61b109cacdac23e7140` |
| `genvsexp` | 用任务相关 cycle-basis 编码平衡表达力和泛化 | `src/sota_sources/GenVsExp/` | `15b95864803183b102fd358e33fdaaf77db8dffe` |

附件给出的 `LOGO-CUHKSZ/DEL` 仓库仅包含一条指向实际实现的说明，因此同时保留入口仓库
`src/sota_sources/DEL/` 和作者指向的实现仓库 `src/sota_sources/DEL_Implementation/`。

## 统一实验协议

- 数据集固定为 NCI1、PROTEINS、COLLAB、MUTAG 和 COLORS-3。
- 监督方法采用外层分层五折，训练折内部再划分验证集并早停。
- CellCLAT 在无标签图上进行胞腔对比预训练，再用同一组分层五折线性探针评测。
- EdgePrompt 先进行 GraphCL 预训练，再冻结编码器，每折只训练边提示和分类头。
- DualPrism 只增强训练批次，验证折与测试折保持原图，防止增强数据泄漏。
- AbstainGNN 按论文先训练预测函数，再启用拒识函数校准；四个常规分类指标按 100% coverage 的预测计算。
- GenVsExp 采用论文实验中的 MPNN 加 3、4 环 cycle-basis 计数，结构编码按图归一化后缓存。
- 每个测试折输出 Accuracy、NMI、ARI、Macro-F1，汇总时报告五折均值与样本标准差。

## 适配边界

十个官方仓库的依赖、数据划分和任务接口不统一，部分仓库依赖旧版 PyG 扩展、固定绝对路径、
W&B、未发布预训练权重或论文限定数据集。官方源码保持原样；统一实现位于
`src/models/paper_graph_models.py`，运行入口为 `src/scripts/run_paper_five_fold.py`。
因此结果标记为 `project_pyg_adapter`，表示依据论文和官方核心源码完成的兼容复现，不能等同于
官方仓库在其原始环境中的零修改成绩。

## 运行方式

单个组合：

```bash
python src/scripts/run_paper_five_fold.py \
  --methods maxcutpool --datasets MUTAG --device cuda:0
```

当前机器的全部可见 GPU：

```bash
python src/scripts/run_paper_parallel.py --devices 0 1 --wait-for-free
```

四张 GPU 均可见时：

```bash
python src/scripts/run_paper_parallel.py --devices 0 1 2 3 --wait-for-free
```
