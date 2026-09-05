"""从本次逐种子CSV生成不依赖中文宏包的论文表；中文说明保留在注释及Markdown中。"""
import csv
import math
from pathlib import Path
from statistics import mean,stdev
ROOT=Path(__file__).resolve().parents[1]
CONFIGS=['Full','w/o ACVG','w/o RDTG','w/o LTCP','Backbone','ACVG only','RDTG only','LTCP only']
DATASETS=['NCI1','Mutagenicity','COLLAB','MUTAG','PTC_MR'];METRICS=['ACC','NMI','ARI','Macro-F1']
with (ROOT/'per_seed_results.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==200 and all(r['status']=='completed' for r in rows),'正式结果未全部完成，拒绝生成论文定稿表'
lines=['% AReT-GCL模块消融。全部数值由本次per_seed_results.csv直接计算。',
       '% 单位：百分数；每种子先对五折求平均，然后报告5种子均值与样本标准差。',
       '% 配置、数据集顺序固定；不省略负收益。只需标准LaTeX，不需要额外表格或中文宏包。']
for ds in DATASETS:
 lines+=['\\begin{table}[htbp]','\\centering','\\caption{'+ds.replace('_','\\_')+r' ($n=5$; $\%$, $\mu\pm s$).}',r'\begin{tabular}{lrrrr}',r'\hline',r' & ACC & NMI & ARI & Macro-F1 \\',r'\hline']
 for c in CONFIGS:
  chosen=[r for r in rows if r['configuration']==c and r['dataset']==ds]
  assert sorted(int(r['seed']) for r in chosen)==[0,1,2,3,4]
  cells=[]
  for metric in METRICS:
   vals=[float(r[metric]) for r in chosen];assert all(math.isfinite(v) for v in vals)
   cells.append(f'${mean(vals):.2f} \\pm {stdev(vals):.2f}$')
  lines.append(c+' & '+' & '.join(cells)+r' \\')
 lines += [r'\hline',r'\end{tabular}',r'\end{table}','']
(ROOT/'ablation_table.tex').write_text('\n'.join(lines),encoding='utf8')
print('论文表已从200条逐种子记录重算；只含标准LaTeX和数值，说明注释为中文')
