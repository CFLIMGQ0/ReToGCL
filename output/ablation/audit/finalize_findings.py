"""正式全因子实验完成后，根据真实配对结果回答研究问题并补充审计。"""
import csv,json,math,statistics,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
CONFIGS=['Full','w/o ACVG','w/o RDTG','w/o LTCP','Backbone','ACVG only','RDTG only','LTCP only']
DS=['NCI1','Mutagenicity','COLLAB','MUTAG','PTC_MR'];MET=['ACC','NMI','ARI','Macro-F1']
with (OUT/'per_seed_results.csv').open() as f:raw=list(csv.DictReader(f))
assert len(raw)==200 and all(r['status']=='completed' for r in raw)
with (OUT/'ablation_summary.csv').open() as f:s=list(csv.DictReader(f))
index={(r['configuration'],r['dataset'],r['metric']):r for r in s}
with (OUT/'factorial_interactions.csv').open() as f:inter=list(csv.DictReader(f))
vals={(r['configuration'],r['dataset'],int(r['seed'])):{m:float(r[m]) for m in MET} for r in raw}
lines=['# 根据真实结果回答研究问题','', '本报告只使用本次200个配置×数据集×种子结果。每组5个种子（0、1、2、3、4），使用相同固定五折。差值为Full减对应消融，单位为百分点。',
       '', '## 三个模块分别是否带来收益','']
for config,module in [('w/o ACVG','ACVG'),('w/o RDTG','RDTG'),('w/o LTCP','LTCP')]:
 r=[index[config,d,m] for d in DS for m in MET]
 pos=sum(float(x['mean_full_minus_ablated'])>1e-12 for x in r);neg=sum(float(x['mean_full_minus_ablated']) < -1e-12 for x in r)
 cipos=sum(float(x['paired_ci95_low'])>0 for x in r);cineg=sum(float(x['paired_ci95_high'])<0 for x in r)
 consistent=pos==20
 lines += [f'### {module}','',f'移除{module}后，20个数据集×指标单元中{pos}个均值下降、{neg}个均值上升、{20-pos-neg}个持平。95%配对区间支持正收益的单元为{cipos}个，支持负收益的单元为{cineg}个。'+('均值方向跨全部数据集及指标一致，但仍需检查种子波动。' if consistent else '收益方向或是否出现收益并非跨全部数据集及指标一致，不能声称普遍有效。'),
           '', '| 数据集 | ΔACC | ΔNMI | ΔARI | ΔMacro-F1 |','|---|---:|---:|---:|---:|']
 for d in DS:lines.append('| '+d+' | '+' | '.join(f'{float(index[config,d,m]["mean_full_minus_ablated"]):+.3f}' for m in MET)+' |')
 bad=[f'{x["dataset"]}/{x["metric"]}（{float(x["mean_full_minus_ablated"]):+.3f}）' for x in r if float(x['mean_full_minus_ablated']) < -1e-12]
 lines += ['', '负收益单元：'+('、'.join(bad) if bad else '无均值为负的单元。'), '']
b=[index['Backbone',d,m] for d in DS for m in MET]
positive=sum(float(r['mean_full_minus_ablated'])>1e-12 for r in b)
all_win=sum(float(r['full_win_rate'])==1. for r in b)
ci_positive=sum(float(r['paired_ci95_low'])>0 for r in b)
lines += ['## Full是否稳定优于Backbone','',f'Full在{positive}/20个单元中均值高于Backbone；在{all_win}/20个单元中五个种子全部获胜；在{ci_positive}/20个单元中95%差值区间完全高于零。'+('当前实验内的均值和种子方向均一致。' if positive==20 and all_win==20 else '不能将局部或平均优势表述为所有数据集、指标与种子上的稳定优势。'),
          '', '| 数据集 | ΔACC | ΔNMI | ΔARI | ΔMacro-F1 |','|---|---:|---:|---:|---:|']
for d in DS:lines.append('| '+d+' | '+' | '.join(f'{float(index["Backbone",d,m]["mean_full_minus_ablated"]):+.3f}' for m in MET)+' |')
complement=[(d,m) for d in DS for m in MET if all(float(index[c,d,m]['mean_full_minus_ablated'])>1e-12 for c in CONFIGS[1:5])]
lines += ['', '## 模块依赖、互补性与协同效应','',f'Full同时高于三个leave-one-out和Backbone的单元为{len(complement)}/20：'+('、'.join(d+'/'+m for d,m in complement) if complement else '无')+'。这一比较只支持对应单元中联合使用的相对优势，不能单独证明超加和协同。',
          '', '| 模块 | 单独加入Backbone的均值增益 | 加入其他两模块背景的均值增益 | 发生方向反转的单元数 |','|---|---:|---:|---:|']
reversal_notes=[]
for module,only,loo in [('ACVG','ACVG only','w/o ACVG'),('RDTG','RDTG only','w/o RDTG'),('LTCP','LTCP only','w/o LTCP')]:
 solo=[];conditional=[];reversal=[]
 for d in DS:
  for m in MET:
   a=statistics.mean(vals[only,d,k][m]-vals['Backbone',d,k][m] for k in range(5))
   c=float(index[loo,d,m]['mean_full_minus_ablated'])
   solo.append(a);conditional.append(c)
   if a*c < -1e-12:reversal.append(d+'/'+m)
 lines.append(f'| {module} | {statistics.mean(solo):+.3f} | {statistics.mean(conditional):+.3f} | {len(reversal)} |')
 reversal_notes.append(f'{module}方向反转位置：'+('、'.join(reversal) if reversal else '无')+'。')
lines += ['',*reversal_notes,'']
lines += ['上表跨单元均值仅作等权描述，不把不同数据集、指标当作独立样本，不据此计算显著性。方向反转提示模块效应依赖其他模块的存在；交互区间用于进一步判断证据强弱。', '', '| 交互项 | 均值为正/负单元数 | 95%区间全正/全负单元数 |','|---|---:|---:|']
for effect in dict.fromkeys(r['effect'] for r in inter):
 rs=[r for r in inter if r['effect']==effect]
 lines.append(f'| {effect} | {sum(float(r["mean"])>0 for r in rs)} / {sum(float(r["mean"])<0 for r in rs)} | {sum(float(r["ci95_low"])>0 for r in rs)} / {sum(float(r["ci95_high"])<0 for r in rs)} |')
synergy=[r for r in inter if r['effect']=='Full相对单模块加和的超额' and float(r['ci95_low'])>0]
antagonism=[r for r in inter if r['effect']=='Full相对单模块加和的超额' and float(r['ci95_high'])<0]
lines += ['', '超加和协同获得正区间支持的位置：'+('、'.join(r['dataset']+'/'+r['metric'] for r in synergy) if synergy else '无')+'。',
          '', '超加和指标获得负区间支持的位置：'+('、'.join(r['dataset']+'/'+r['metric'] for r in antagonism) if antagonism else '无')+'。',
          '', '## 论文表述的证据边界','',
          '- 模块带来增益：可限定为上述正差值单元；差值区间跨零时应写为观察到平均改善，不能夸大为稳定提升。全部负收益已列出。',
          '- 模块具有必要性：未得到普遍必要性证据。本实验是固定设置下的相对贡献检验，不能证明删除模块后任务不可完成。',
          f'- 三个模块具有互补性：{len(complement)}/20个单元符合Full同时超过三个leave-one-out与Backbone；只支持这些单元的联合相对优势。不能据此声称所有数据集和指标均互补。',
          f'- 三个模块存在协同效应：{len(synergy)}/20个单元的Full超加和指标95%区间高于零；{len(antagonism)}/20个单元低于零。应按具体单元报告，不作无条件的协同宣称。',
          '- 只有5个训练种子；Student-t配对区间未做多重比较校正。固定划分条件下的区间不包含重新抽样/重新划分的不确定性，交互分析属于探索性证据。',
          '- 每个种子在全部无标签图上预训练，包含外折测试图；线性评估才划分标签。NMI/ARI由监督线性预测计算。本实验结论不能直接改写为独立归纳式测试或无监督聚类性能。',
          '- 原始模型中存在额外残差混合及未接入最终正样本的默认构造输出；全部配置保留相同行为。实际连接详见audit/execution_path.md。', '']
(OUT/'research_findings.md').write_text('\n'.join(lines),encoding='utf8')
# 更正初版自动说明中过强的推断：本次没有保存显式节点掩码，不声称数据一定存在零特征。
p=OUT/'diagnostics.md';d=p.read_text()
d=d.replace('存在原始全零节点特征，不能从增强后x可靠还原节点掩码；不生成代理数据。','本次未保存显式节点保留掩码，仅记录直接可验证的边交并比；不将增强后零特征无条件等同于被删除节点，也不生成代理数据。')
d+='\n## 补充审计与逐项研究结论\n\n实际残差符号、未接入最终正样本的默认构造输出及参数活跃数量定义见 [执行路径审计](audit/execution_path.md)。八个研究问题的具体结论、所有负收益位置、模块依赖与论文表述边界见 [真实结果研究结论](research_findings.md)。\n'
p.write_text(d,encoding='utf8')
subprocess.run([sys.executable,str(OUT/'audit/render_publication_table.py')],check=True)
m=json.loads((OUT/'run_manifest.json').read_text());m['final_git_status_short']=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
m['postprocessing']={'completed_at':datetime.now(timezone.utc).isoformat(),'commands':[[sys.executable,str(Path(__file__).resolve())],[sys.executable,str(OUT/'audit/render_publication_table.py')]],'note':'只从正式逐种子CSV生成具体研究结论和可直接插入论文的标准LaTeX表；修正未保存节点掩码的说明，不修改训练代码或结果数值。'}
(OUT/'run_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
print('真实研究结论、论文表及补充审计已生成')
