# 当前项目情况

快照时间：2026-09-05 19:37 UTC。本文根据本地代码、归档结果和调度状态核验；正在运行的任务可能继续推进，GitHub 不是实时监控页面。

## 1. 模型与命名

| 模型 | 实验标识或组合 | 当前定位 |
| --- | --- | --- |
| CITA-GCL | `PL069`；`D7-I10 + D7-I01 + D10-I04`，含 T077 基础改动及锚点旁路 | 已命名的改进模型，结构与参数见 [CITA-GCL.md](CITA-GCL.md) |
| 原始 ReToGCL | `D7-I10 + D7-I01 + D10-I04` 的原始版本 | 独立对照，保留原始实现及历史结果 |
| AReT-GCL 消融模型 | `D11-I01 + D2-I09 + D10-I10`；ACVG、RDTG、LTCP | 另一组三模块的固定参数消融，不是 CITA-GCL 的别名 |

`model.yaml` 仍是通用 BalanceGCL 模块装配清单，不因模型命名或提交而改变默认参数。旧协议标识、检查点路径及 `PL069` 内部方法键保留兼容。

## 2. 已完成的对照实验

| 实验批次 | 实验范围 | 完成情况 | 结果入口 |
| --- | --- | --- | --- |
| 现有 18 数据集对照 | 13 个选定 SOTA + 原始 ReToGCL + CITA-GCL；五折 | 18 个数据集均有完整对照表 | [table.md 第 7 节](table.md#7-18个数据集完整对照13个选定-sotaretogcl-与-cita-gcl) |
| 新增 5 个医学分子数据集 | 上述 15 个模型；五折 | 75/75 项 | `table.md` 第 8.1 节 |
| 六数据集经典直接基线 | 9 个模型；五折 | 54/54 项 | `table.md` 第 8.2 节 |
| 六数据集多随机种子复核 | 6 个模型 × 10 个训练种子；每种子固定五折 | 360/360 项 | `table.md` 第 8.3、8.4 节 |

最近三批合计 **489/489 项**，本次逐文件核验其四项指标字段均存在。“项”指模型/配置、数据集及训练种子的组合，不把每一折重复计为一项。

13 个选定 SOTA 为 BalanceGCL、Khan-GCL、CellCLAT、UniImb、DualPrism、DEL、SpectRe、TopER、LEAP、Hourglass、NodeID、GNN+、RS-Pool。多种子批次的六模型为原始 ReToGCL、CITA-GCL、BalanceGCL、Khan-GCL、DEL、SimplicialMP。

多种子均值下，CITA-GCL 相比原始 ReToGCL 在 BACE、BBBP、AIDS、BZR 的四项指标均更高，在 ADHD200、Tox21 的四项指标均更低；这不等于所有配对差值都有显著性，也不等于超过每个 SOTA。具体均值、标准差、95% 置信区间和配对差值保留在完整表中。

## 3. 数据集范围

- 历史 18 个：NCI1、PROTEINS、COLLAB、MUTAG、COLORS-3、PTC_MR、Mutagenicity、ClinTox、BACE、BBBP、ABIDE、ADHD200、SIDER、HIV、AIDS、Tox21、BZR、COX2。
- 新增 5 个：hERG_Karim、CYP2D6_Veith、CYP3A4_Veith、Pgp_Broccatelli、DILI。
- 当前上述对照覆盖合计 23 个数据集；六个正式复核数据集为 ADHD200、BACE、BBBP、Tox21、AIDS、BZR。

这些数据覆盖分子性质、药理/毒理、脑图及其他图分类任务，不能统称为临床患者诊断数据。原始数据和缓存不随 Git 提交。

## 4. 正在运行的 AReT-GCL 消融

- 数据集固定为 NCI1、Mutagenicity、COLLAB、MUTAG、PTC_MR。
- 每个配置使用种子 0–4；每个种子使用同一套 `split_seed=42` 五折索引，40 轮预训练、200 轮线性探测。
- 核心配置：Full、w/o ACVG、w/o RDTG、w/o LTCP、Backbone，共 `5 × 5 × 5 = 125` 项。
- 核心完成后按既有调度脚本运行 ACVG only、RDTG only、LTCP only，共 75 项；合计计划 200 项。
- 本次发布的汇总快照已完成 **76/200 项**，均属于核心批次；核心为 76/125，扩展为 0/75。其余 124 项尚无完成结果，不按失败或零分处理。
- 快照时本机 202 的 GPU 0、1 各有一个消融任务：COLLAB/seed3 的 w/o RDTG 和 Full；核心队列另有 47 项等待，未记录失败尝试。本次上传不停止、重启或调整调度。
- 8 个配置的冒烟运行与开关/协议验证已通过；正式 200 项的最终验收尚未执行。当前没有任何配置/数据集组凑齐五个种子，暂不作最终模块贡献或协同结论。

已从当前正式任务 JSON 重新生成 [阶段结果表](output/ablation/ablation_table.md)、[逐种子 CSV](output/ablation/per_seed_results.csv)、[审计与诊断](output/ablation/diagnostics.md)。完整设置和任务尝试见 [运行清单](output/ablation/run_manifest.json)，实际模块连接与消融边界见 [执行路径审计](output/ablation/audit/execution_path.md)。

服务器上只读查看实时状态与日志：

```bash
cd /home/Lim/Project-xmlg
watch -n 10 'cat output/ablation/live_status.json'
```

```bash
tail -n 50 -f output/ablation/logs/controller.log
```

## 5. 结果解释和复现边界

- 四指标为 Accuracy、NMI、ARI、Macro-F1。这里的 NMI/ARI 根据线性分类器或监督模型的预测计算，不能直接描述为独立无监督聚类结果。
- 当前自监督协议在全部无标签图上预训练，再划分标签进行五折评估，属于传导式设置；不是外折测试图完全不可见的归纳式预训练。
- DEL、SimplicialMP 的十种子实验采用现有监督式实现，不应写成十个自监督预训练种子。
- SIDER、Tox21 分别对 27、12 个任务评测后作任务宏平均。对比数值来自本项目适配实现，不是各论文直接报告的数值。
- AReT-GCL 的 cycle 分支是现有可微代理，ACVG 的 subgraph 不是严格随机游走；真实行为见审计文档，不用论文设想替代实现证据。
- `paper_retogcl/` 是原始 ReToGCL 草稿；[AReT-GCL_ICLR2027_story.pdf](AReT-GCL_ICLR2027_story.pdf) 是本地已有的研究叙事草稿。保留文件不代表其主张已经验证或论文已经发表。

## 6. 本次上传范围

上传当前新增的 AReT-GCL 消融代码、研究草稿、阶段结果、固定折索引、开关检查、代码快照与审计材料，并补充本状态说明及 README 入口。原始 ReToGCL、CITA-GCL 的模型、参数和 `table.md` 数值均未改动。

不上传数据集、检查点、训练日志、缓存、进程锁和实时状态文件；它们仍保留在本地。部分运行记录包含当时的绝对路径，换机器复现时需根据本地目录解析，且需要另行准备数据。
