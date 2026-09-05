# 审计、机制诊断与证据边界

更新：2026-09-05T19:37:13.494460+00:00。正式完成 76/200 次。Git commit：`7ca35b3eac359cb11968f5b22340db7baea3ea9a`。

## 代码与原始设置

- 原训练入口：`src/scripts/run_modular_idea_triple_trial.py`；新入口：`src/scripts/run_aret_ablation.py`，调度入口：`src/scripts/launch_aret_ablation.py`。
- 原评估入口：`src/baselines/evaluation.py::linear_probe_five_fold_metrics`，本次直接调用，原文件未修改。
- ACVG：`src/models/modular_research_ideas.py::prepare_views` 的 family=11、variant=1。node_drop 是随机节点掩码；subgraph 是带随机扰动的度优先节点保留，不是真正随机游走子图。节点不从Data中删除，而是置零并去掉关联边。
- RDTG：`src/models/research_ideas.py::_family2` 的 variant=9。d_i是两个归一化cycle视图的均方差，按批次总体标准差（ddof=0，最小1e-5）标准化，g_i=sigmoid(-z_i)。cycle是节点与均值邻居差的可微代理，不是精确环拓扑。
- LTCP：`src/models/research_ideas.py::_bundle_from_history` 与 `_family10` 的 variant=10。实际GIN history有4个状态，选索引0、2、3；坐标为s0−2s2+s3，经过128→32→128的Tanh瓶颈。几何损失比较投影内部距离矩阵与已detach的node/cycle交叉距离矩阵。
- 三模块共享及残差混合位置：`src/models/modular_triple_research_ideas.py`、`src/models/modular_research_ideas.py::_module_transform`。M2增益1.5、M7增益17/9；并非直接把模块输出替换原表示。温度0.2。
- 历史五数据集配置确认seed42、40轮、Adam、lr0.001、weight_decay=1e-5、hidden32、3层、表示128、batch128（代码对COLLAB实际64）、5折线性探针200轮。辅助权重RDTG=0.1875、LTCP=1.25。历史结果数值未复用。
- 无法从仓库确认：该组合原始多种子集合、历史环境完整版本、历史逐样本划分文件。原代码明确没有早停或独立验证集。通用train.yaml不是该入口的实际训练设置。
- 本次新增统一设置：训练种子0至4、所有种子固定split_seed42、严格确定性计算、独立DataLoader随机生成器及线程数1。Full也从头重跑，无旧checkpoint初始化。
- 额外验证诊断：每个外折单独用60%训练、20%验证、20%测试的索引训练固定轮数诊断探针，仅记录训练/验证损失；随后调用原函数在80%上重训固定200轮并测试20%。没有根据验证结果选模型，不改变原测试评估流程。
- 自监督预训练使用全部无标签图，包含测试图；这是原仓库传导式协议。NMI/ARI基于线性分类预测，不是另行运行的无监督聚类。

## 最小反事实及开关验证

- 新模型继承原模型。全部开关开启直接调用原方法；原模型、评估及增强文件的SHA256与运行前快照一致。`smoke/switch_verification.json` 保存增强、损失、诊断、梯度、更新后权重的逐位比较。
- w/o ACVG：仅将第二支subgraph改为已有node_drop；保留5%和2%节点预算、随机数消耗及逐图保留数量。原BalanceGCL的edge_drop/attr_mask也使用不同扰动类型且不能匹配节点预算，因此这里选已有node_drop作为对称基准；这个取舍是本次消融定义。
- w/o RDTG：去掉gate×cycle残差和专属辅助项，保留归一化graph head与外层混合（1−a）G+a×graph_head(G)。直接返回sum-pool会将占主导的原−0.5G变为+G，带来尺度及符号混杂。当前基准不等同于未经适配的BalanceGCL。
- w/o LTCP：三个原scale head都读取最后一层，用三个归一化终点坐标的平均值替代轨迹二阶差分，再送入相同瓶颈；保持三个head全部活跃、输出128、参数容量一致、原外层残差系数、M7困难负样本路径和其他接口不变。移除专属几何损失。
- Backbone同时使用上述三个基准，没有将输出张量置零。辅助项为零只是表示该专属目标不参与总损失。
- 全部八配置在MUTAG、seed0运行2轮预训练、5轮验证诊断/测试探针，完成五折；正式为40/200轮。smoke与正式产物目录独立。
- 每批检查损失和梯度有限、关闭模块的专属损失为零；检查最终参数、表示、指标有限。逐任务记录开关、初始化哈希、代码哈希、划分哈希、数据哈希和完整命令。

## 机制诊断

- `diagnostics/*.json` 记录每次训练第一批实际增强图的边集合Jaccard和边数量；训练结束后在该固定批次eval模式记录d_i、标准化分歧、g_i的逐样本值及分位数、门控前后范数和终点/轨迹表示差异。训练目标和BN状态不受诊断影响。
- `logs/*.epochs.jsonl` 保存每轮真实训练目标、两个专属辅助损失及加权辅助总和。
- 诊断分布仅代表固定第一批的训练后eval状态，不代表整个训练过程或数据集。关闭模块时同公式算出的诊断是对实际编码状态的反事实计算，不是生效门控；下文仅统计Full的机制量。
- 未报告节点保留重叠度：存在原始全零节点特征，不能从增强后x可靠还原节点掩码；不生成代理数据。

| 数据集 | Full诊断批次数 | 边Jaccard均值 | 门控/未门控图范数比均值 | gate均值 |
|---|---:|---:|---:|---:|
| NCI1 | 3 | 0.8460 | 0.9808 | 0.5247 |
| Mutagenicity | 3 | 0.8518 | 0.9877 | 0.5233 |
| COLLAB | 3 | 0.8851 | 0.9919 | 0.5139 |
| MUTAG | 3 | 0.8170 | 0.9888 | 0.5173 |
| PTC_MR | 3 | 0.7879 | 0.9739 | 0.5218 |

## 真实结果与研究问题

尚无完整五种子配对，暂不回答模块增益或论文主张。

## 异常、失败和限制

- 初始Python导入sklearn时缺少系统CXXABI_1.3.15。实验进程使用当前Conda环境lib目录的LD_LIBRARY_PATH，导入验证成功；未改系统库。torch_scatter/torch_sparse未安装，当前代码使用PyG/PyTorch路径，smoke检查兼容性。
- 两张4090已有其他用户任务，本调度器每卡最多一个任务，持续保存GPU状态；没有停止其他进程。环境实际版本与GPU信息见run_manifest.json。
- 正式/冒烟已记录的失败尝试总数：0。所有错误日志、部分产物及重跑记录保留；无不利种子删除。
- 仅5个训练种子；固定数据划分的区间只反映训练随机性，不衡量重新划分、数据采样或外部泛化不确定性。Student-t区间未做多重比较校正。
- 既有模块残差系数很大，原始图读出可以占主导；LTCP差异受这一实际执行路径约束，不将论文期望替代代码证据。
- 本实验不加入其他数据集、模块或历史候选得分，不重新调参，不修改论文正文、现有LaTeX或PDF。
- 初始未跟踪文件AReT-GCL_ICLR2027_story.pdf保留。完整代码前后快照位于audit/source_before与audit/source_used。
