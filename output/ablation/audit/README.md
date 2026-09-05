# 本次实验的复核入口

全部训练、验证诊断、五折测试及原始结果属于本次固定实验；历史文件仅用于确认设置，没有导入历史分数。

## 环境与执行

仓库根目录：`/home/Lim/Project-xmlg`。使用当前 Conda 环境 Python，为子进程设置 `LD_LIBRARY_PATH=/home/Lim/conda/envs/myenv/lib`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`PYTHONHASHSEED=0`。

可靠会话名为 `aret_ablation_20260905`。主调度器的命令、PID和环境保存在 `../controller.json`；每次子任务的完整命令、环境、开始/结束时间及退出状态都保存在 `../run_manifest.json`。

- 状态：`../live_status.json`。
- 调度日志：`../logs/controller.log`。
- GPU状态：`../logs/gpu_monitor.jsonl`。
- 正式逐任务日志：`../logs/*.attempt*.log`。
- 正式逐轮损失：`../logs/*.epochs.jsonl`。
- 正式结果：`../jobs/*.json`，保留五折分数与五折验证诊断损失。
- 原始逐种子汇总：`../per_seed_results.csv`；未完成组合保留pending，失败组合保留failed。
- 代码版本冻结：`source_used/`及run_manifest中的SHA256；运行前原代码快照：`source_before/`。

调度顺序为八配置MUTAG/seed0 smoke，随后核心125次，核心全部成功后再执行扩展75次。任何失败都会保存日志并停止启动新任务，已有任务正常完成后退出，便于Agent查因后恢复；已成功任务不重复训练。部分输出会移动至failed_attempts保留，不会静默丢弃失败。

## 独立复核

`verify_protocol.py` 验证五数据集五折分割覆盖、训练/验证/测试隔离、全部八配置smoke输出字段，并从头重放Full两轮训练，逐位比较完整模型状态。结果见 `../smoke/protocol_verification.json`。

原模型与新增开关的等价性验证见 `../smoke/switch_verification.json`：检查相同初始化、增强、损失、诊断、梯度以及参数更新。

统计可执行：

```bash
LD_LIBRARY_PATH=/home/Lim/conda/envs/myenv/lib OPENBLAS_NUM_THREADS=1 python src/scripts/summarize_aret_ablation.py
```

完整200次正式训练结束后附加 `--verify` 可进行最终验收。脚本先从本次jobs生成逐种子CSV，再独立读回CSV计算所有统计；CSV保留全精度。

## 统计定义

每个配置/数据集/种子的四项指标均为五折测试分数的等权均值，以百分数保存。跨种子标准差采用ddof=1。

配对差值为同一数据集、同一固定划分、同一种子的Full减消融。胜率为严格正差值的种子比例，持平不计胜。95%区间为配对差值均值±t分位数×配对样本标准差/√n，仅反映固定数据下训练种子变化，未做多重比较校正。

全因子交互以种子为单位计算：二阶差中差对第三模块两个背景取平均；三阶交互为八角点交替和；Full相对于三个单模块增益加和的超额为Full−ACVG only−RDTG only−LTCP only+2×Backbone。不能仅凭Full最高均值断言协同。
