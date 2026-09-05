# CITA-GCL

## 项目目标

CITA-GCL（Cleanliness-Informed Topological Alignment for Graph Contrastive Learning，洁净度引导的拓扑对齐图对比学习）是本项目改进模型的正式名称，对应历史实验配置 `PL069`。模型结合互补图增强、共享 GIN 编码、洁净度引导的语义路由、拓扑条件映射、困难负样本对齐和锚点表征旁路，并采用五折 Linear Probe 评估 Accuracy、NMI、ARI 和 Macro-F1。

原始 ReToGCL 继续作为独立对照保留，不与 CITA-GCL 混用。模型名称与实验编号的对应关系见 [CITA-GCL.md](CITA-GCL.md)，最新实验结果见 [table.md](table.md)。本次更名不改变模型结构、参数、训练协议或结果数值；代码接口、实验编号和历史路径保持兼容，GitHub 仓库地址也保持不变。

原始 ReToGCL 的英文论文草稿、公式、训练伪代码及历史 18 数据集结果矩阵位于 `paper_retogcl/`；该草稿不因本次更名自动成为 CITA-GCL 的论文或结果。

## 仓库内容说明

本仓库保留核心代码、配置、实验说明、结果汇总和论文文件。数据集、模型权重、训练缓存、运行日志以及第三方源码镜像不纳入 Git；请根据 `DATASETS.md` 和相应下载脚本在本地准备数据。

## 整体流程

1. 运行数据下载脚本，下载 Planetoid 节点分类数据集或 TU 图分类数据集。
2. 在 `DATASETS.md` 中记录数据来源、标签定义、清洗规则和划分策略。
3. 在 `src/configs/train.yaml` 中维护数据路径、训练参数和输出路径。
4. 将正式模型放入 `src/models/`，将对照模型放入 `src/baselines/`。
5. 将训练、评估、数据处理和结果分析入口放入 `src/scripts/`。
6. 将日志、checkpoint、图表和分析结果统一写入 `outputs/`。

## 目录说明

```text
Project-xmlg/
├── AGENTS.md
├── PROJECT_INIT.md
├── README.md
├── DATASETS.md
├── EXPERIMENT_PLAN.md
├── datasets/
│   └── main_data/
├── outputs/
├── pre_weights/
└── src/
    ├── models/
    ├── baselines/
    ├── configs/
    │   └── train.yaml
    └── scripts/
```

## 运行方式

下载 NCI1、PROTEINS、COLLAB、MUTAG 和 COLORS-3：

```bash
python3 src/scripts/download_tu_datasets.py
```

默认约定如下：

- 主数据路径：`datasets/main_data/`
- 预训练权重路径：`pre_weights/`
- 训练配置：`src/configs/train.yaml`
- 实验输出路径：`outputs/`

## 图对比学习 Baseline

项目已实现 GraphCL、JOAOv2、RGCL 和 SimGRACE，并在五个图分类数据集上执行分层五折评测。运行全部基线：

```bash
python3 src/scripts/run_gcl_baselines.py --device auto
```

实验配置位于 `src/configs/gcl_baselines.yaml`，汇总及逐折结果见 `BASELINE_RESULTS.md`，机器可读结果位于 `outputs/baselines/five_fold_results.json`。

## 配置文件

`src/configs/train.yaml` 提供通用训练配置骨架。字段可随任务扩展，但已有英文键名应保持稳定，避免脚本接口不一致。

## 新增 SOTA 五折实验

五篇新增论文、官方源码、审计提交和适配边界记录在 `SOTA_SOURCES.md`。五模型覆盖 NCI1、PROTEINS、COLLAB、MUTAG、COLORS-3 五个图分类数据集，正式结果见 `SOTA_RESULTS.md`。

## 多空间门控相似度研究

多空间相似度诊断、边界感知门控模块和九种消融配置记录在 `MSGS_RESEARCH.md`。诊断脚本为 `src/scripts/diagnose_multispace_similarity.py`，五折消融入口为 `src/scripts/run_multispace_ablation.py`。

单卡或指定单组运行：

```bash
python3 src/scripts/run_sota_five_fold.py \
  --methods histograph \
  --datasets MUTAG \
  --device cuda:0
```

四卡并行运行全部 25 个模型/数据集组合：

```bash
python3 src/scripts/run_sota_parallel.py --devices 0 1 2 3
```

当前机器只暴露 GPU 0、1 时使用：

```bash
python3 src/scripts/run_sota_parallel.py --devices 0 1
```

配置约定见 `src/configs/sota_five_fold.yaml`，单任务日志、checkpoint 和五折汇总统一写入 `outputs/sota/`。

## 实验输出

所有运行产物应保存至 `outputs/`，建议按实验名称和时间组织子目录。不要将日志、模型权重或分析结果写入 `src/`。
