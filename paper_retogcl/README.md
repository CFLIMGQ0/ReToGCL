# ReToGCL 论文草稿

本目录包含英文论文的 LaTeX 源文件和编译产物。

## 文件

- `paper.tex`：完整英文论文源文件，包含模型公式、训练伪代码、完整主表、实验规划和 TikZ 框架图。
- `full_sota_results.tex`：18 个数据集、35 个对比模型及 ReToGCL 的完整四指标附录表。
- `generate_full_sota_appendix.py`：从项目归档 JSON 结果重新生成完整附录表，并检查方法缺失或重复。
- `paper.pdf`：编译后的 PDF；当前完整结果矩阵位于论文末尾。

## 编译

如系统已安装 Tectonic，可在本目录执行：

```bash
python generate_full_sota_appendix.py
tectonic paper.tex
```

也可以使用完整 TeX Live：

```bash
latexmk -pdf paper.tex
```

## 投稿前必须完成的事项

1. 将匿名作者和机构替换为真实信息。
2. 核验并补齐所有 `[CITATION NEEDED]`，不得直接猜测作者、会议、年份或 DOI。
3. 完成正文明确标记为 `Not run` 的消融实验和参数敏感性实验后，再填写结果。
4. 建议增加多随机种子、严格归纳式预训练、分子 scaffold split 和外部验证。
5. 将统一项目的 35 个对比方法适配说明、环境文件和固定折索引随补充材料公开。

主表中的对比数值来自本项目统一 PyG 适配、统一数据处理与五折评估输出，不得描述为各方法原论文直接报告的数值。
