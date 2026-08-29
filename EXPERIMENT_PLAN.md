# 实验方案

## 研究定位

具体研究问题、任务定义、创新点和适用范围待补充。

## 数据划分

训练集、验证集和测试集的比例及构建方法待确定。最终方案应记录随机种子、分层或分组规则，并明确数据泄漏防范方式。

## 数据预处理

待确定输入格式、标准化或增强方法、缺失值处理方式及训练与推理阶段的一致性约束。

## Baseline

当前采用 GraphCL、JOAOv2、RGCL 和 SimGRACE 作为图对比学习基线。实现位于 `src/baselines/`，统一运行入口为 `src/scripts/run_gcl_baselines.py`，配置为 `src/configs/gcl_baselines.yaml`。

现阶段在 NCI1、PROTEINS、COLLAB、MUTAG、COLORS-3 上采用固定随机种子 42 的分层五折线性评测。完整结果记录在 `BASELINE_RESULTS.md`。

## 新增 SOTA

新增 HISTOGRAPH、UniImb、DiffLift、BalanceGCL 和 Khan-GCL。五者的共同原生任务为图分类，统一在 NCI1、PROTEINS、COLLAB、MUTAG、COLORS-3 上比较。源码地址、精确提交、论文方法和官方仓库可运行性审计见 `SOTA_SOURCES.md`。

- 监督式方法 HISTOGRAPH、UniImb、DiffLift 采用外层分层五折，每折训练集内部再划分验证集进行早停，禁止依据测试折选择训练轮次。
- 自监督方法 BalanceGCL、Khan-GCL 先预训练图编码器，再采用固定随机种子 42 的同一组分层五折线性探针。
- 所有批量预处理和训练过程显示进度条，日志与 checkpoint 写入 `outputs/sota/`。
- Cora、CiteSeer、PubMed 已按项目范围调整移除，不再纳入运行入口的默认实验。

## 主模型

主模型架构、输入输出、损失函数、训练策略和预训练权重方案待确定。正式模型实现统一放入 `src/models/`。

## 评价指标

待根据任务类型确定主指标、辅助指标、统计检验方法和结果报告格式。

## 实验设计

计划至少覆盖以下实验类型，具体内容根据研究问题调整：

1. Baseline 对比实验。
2. 主模型性能实验。
3. 消融实验。
4. 超参数或敏感性分析。
5. 鲁棒性与泛化能力分析。
6. 典型样本及失败案例分析。

## 可复现性约定

- 训练参数统一维护在 `src/configs/train.yaml` 或同目录下的扩展配置中。
- 数据版本、划分方式和统计信息记录在 `DATASETS.md`。
- checkpoint、日志、图表和汇总结果统一写入 `outputs/`。
- 每项正式实验应记录配置、随机种子、代码版本和运行环境。
