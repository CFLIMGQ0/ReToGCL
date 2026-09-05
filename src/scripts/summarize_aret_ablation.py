"""仅从本次逐种子结果重算消融表、配对统计及全因子交互；保留失败/待运行记录。"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import math
import numpy as np
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.scripts.run_aret_ablation import OUT,CONFIGS,DATASETS,SEEDS,METRICS,SETTINGS,job_id,sha,save,now


def write_csv(path, rows, fields=None):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def stats(values):
    x=np.asarray(values,dtype=float);n=len(x)
    if not n:return dict(n=0,mean='',std='',ci95_low='',ci95_high='')
    mean=float(x.mean());sd=float(x.std(ddof=1)) if n>1 else 0.
    half=float(student_t.ppf(.975,n-1))*sd/math.sqrt(n) if n>1 else None
    return dict(n=n,mean=mean,std=sd,ci95_low=mean-half if half is not None else '',ci95_high=mean+half if half is not None else '')


def gather(manifest):
    rows=[];jobs=[]
    for c in CONFIGS:
        for ds in DATASETS:
            for seed in SEEDS:
                path=OUT/'jobs'/f'{job_id(c,ds,seed)}.json'
                failures=[a for a in manifest['attempts'] if a['phase']=='formal' and (a['configuration'],a['dataset'],a['seed'])==(c,ds,seed) and a['status']=='failed']
                row=dict(configuration=c,dataset=ds,seed=seed,status='failed' if failures else 'pending',
                         **dict(zip(['use_acvg','use_rdtg','use_ltcp'],[int(x) for x in CONFIGS[c]])),
                         **{k:'' for k in METRICS},split_sha256=manifest['datasets'][ds]['split_sha256'],failed_attempts=len(failures),duration_seconds='',result_path=str(path))
                if path.exists():
                    r=json.loads(path.read_text());assert r['phase']=='formal' and r['status']=='completed'
                    assert (r['configuration'],r['dataset'],r['seed'])==(c,ds,seed)
                    assert r['split_sha256']==row['split_sha256']
                    assert r['data_sha256']==manifest['datasets'][ds]['preprocessed_sha256']
                    assert r['code_sha256']==manifest['code_sha256']
                    assert r['settings']=={**SETTINGS,'effective_batch_size':64 if ds=='COLLAB' else 128}
                    assert all(r['metrics'][k]==float(np.mean(r['fold_metrics'][k])) for k in METRICS)
                    row.update(status='completed',**r['metrics'],duration_seconds=r['duration_seconds']);jobs.append(r)
                rows.append(row)
    write_csv(OUT/'per_seed_results.csv',rows)
    return jobs


def recompute():
    with (OUT/'per_seed_results.csv').open() as f:rows=list(csv.DictReader(f))
    values={(r['configuration'],r['dataset'],int(r['seed'])):{k:float(r[k]) for k in METRICS} for r in rows if r['status']=='completed'}
    summaries=[];paired=[]
    for c in CONFIGS:
        for ds in DATASETS:
            for metric in METRICS:
                vals=[values[c,ds,s][metric] for s in SEEDS if (c,ds,s) in values]
                st=stats(vals);pairs=[];wins=ties=0
                if c!='Full':
                    for seed in SEEDS:
                        if ('Full',ds,seed) not in values or (c,ds,seed) not in values:continue
                        full=values['Full',ds,seed][metric];ab=values[c,ds,seed][metric];delta=full-ab
                        pairs.append(delta);wins+=delta>1e-12;ties+=abs(delta)<=1e-12
                        paired.append(dict(configuration=c,dataset=ds,seed=seed,metric=metric,full=full,ablated=ab,full_minus_ablated=delta,full_win=int(delta>1e-12),tie=int(abs(delta)<=1e-12)))
                ps=stats(pairs)
                summaries.append(dict(configuration=c,dataset=ds,metric=metric,n_seeds=st['n'],mean=st['mean'],std=st['std'],
                    n_pairs=ps['n'],mean_full_minus_ablated=ps['mean'],paired_std=ps['std'],full_win_rate=wins/ps['n'] if ps['n'] else '',
                    ties=ties,paired_ci95_low=ps['ci95_low'],paired_ci95_high=ps['ci95_high'],paired_differences=json.dumps(pairs)))
    return rows,values,summaries,paired


def interactions(values):
    result=[]
    by_flags={v:k for k,v in CONFIGS.items()}
    for ds in DATASETS:
        for metric in METRICS:
            by_effect={}
            for seed in SEEDS:
                if not all((c,ds,seed) in values for c in CONFIGS):continue
                v={flags:values[c,ds,seed][metric] for flags,c in by_flags.items()}
                for i,j,n in [(0,1,'ACVG×RDTG'),(0,2,'ACVG×LTCP'),(1,2,'RDTG×LTCP')]:
                    other=3-i-j;diffs=[]
                    for level in [False,True]:
                        f=[False]*3;f[other]=level
                        f00=tuple(f);f[i]=True;f10=tuple(f);f[i]=False;f[j]=True;f01=tuple(f);f[i]=True;f11=tuple(f)
                        diffs.append(v[f11]-v[f10]-v[f01]+v[f00])
                    by_effect.setdefault(n,[]).append(float(np.mean(diffs)))
                triple=v[True,True,True]-v[True,True,False]-v[True,False,True]-v[False,True,True]+v[True,False,False]+v[False,True,False]+v[False,False,True]-v[False,False,False]
                excess=v[True,True,True]-v[True,False,False]-v[False,True,False]-v[False,False,True]+2*v[False,False,False]
                by_effect.setdefault('三阶交互',[]).append(triple);by_effect.setdefault('Full相对单模块加和的超额',[]).append(excess)
            for name,x in by_effect.items():result.append(dict(dataset=ds,metric=metric,effect=name,**stats(x),per_seed_differences=json.dumps(x)))
    return result


def tables(summary, complete_count):
    index={(r['configuration'],r['dataset'],r['metric']):r for r in summary}
    md=['# AReT-GCL 模块消融结果','',f'实际完成 {complete_count}/200 次正式训练；每组计划5种子（0、1、2、3、4），每种子固定五折。核心实验计划125次，扩展75次。',
        '', '单位为百分数；均值 ± 跨种子样本标准差（ddof=1）。每个种子的数值是五折测试指标的等权均值。未完成时明确标记，smoke结果不进入本表。', '']
    tex=['% 本文件由本次逐种子CSV生成；单位为百分数，标准差跨种子计算。','% 只输出独立实验表，不修改现有论文。']
    for ds in DATASETS:
        md += [f'## {ds}','','| 配置 | 种子数 | ACC | NMI | ARI | Macro-F1 |','|---|---:|---:|---:|---:|---:|']
        tex += ['\\begin{table}[ht]','\\centering', '\\caption{'+ds.replace('_','\\_')+' 模块消融：百分数，均值 $\\pm$ 标准差，5个种子。}', '\\begin{tabular}{lrrrr}', '\\hline', '配置 & ACC & NMI & ARI & Macro-F1 \\\\', '\\hline']
        for c in CONFIGS:
            cells=[];tcells=[]
            for metric in METRICS:
                r=index[c,ds,metric]
                if r['n_seeds']==5:cells.append(f'{r["mean"]:.2f} ± {r["std"]:.2f}');tcells.append(f'${r["mean"]:.2f} \\pm {r["std"]:.2f}$')
                elif r['n_seeds']:cells.append(f'{r["mean"]:.2f} ± {r["std"]:.2f}（未完成）');tcells.append('未完成')
                else:cells.append('待运行');tcells.append('待运行')
            md.append('| '+c+' | '+str(index[c,ds,'ACC']['n_seeds'])+' | '+' | '.join(cells)+' |')
            tex.append(c+' & '+' & '.join(tcells)+' \\\\')
        md.append('');tex+=['\\hline','\\end{tabular}','\\end{table}','']
    md += ['## 配对差值（Full − 消融，百分点）','','| 配置 | 数据集 | 指标 | 配对种子数 | 平均差值 | Full胜率 | 差值95%置信区间 |','|---|---|---|---:|---:|---:|---|']
    for r in summary:
        if r['configuration']=='Full' or not r['n_pairs']:continue
        lo,hi=r['paired_ci95_low'],r['paired_ci95_high']
        ci=f'[{lo:.3f}, {hi:.3f}]' if lo!='' else '不足2种子'
        md.append(f'| {r["configuration"]} | {r["dataset"]} | {r["metric"]} | {r["n_pairs"]} | {r["mean_full_minus_ablated"]:+.3f} | {100*r["full_win_rate"]:.0f}% | {ci} |')
    md += ['', '置信区间采用种子配对差值的 Student-t 区间；只有5个种子，正态近似有限，未进行多重比较校正。正值表示Full更好，零差值不计胜。不能把多个数据集和指标单元格当作独立重复。', '']
    (OUT/'ablation_table.md').write_text('\n'.join(md),encoding='utf8');(OUT/'ablation_table.tex').write_text('\n'.join(tex),encoding='utf8')


def diagnostics(m,jobs,summary,inter):
    good=[r for r in summary if r['n_pairs']==5]
    complete=len(jobs)
    lines=['# 审计、机制诊断与证据边界','',f'更新：{now()}。正式完成 {complete}/200 次。Git commit：`{m["git_commit"]}`。',
           '', '## 代码与原始设置','',
           '- 原训练入口：`src/scripts/run_modular_idea_triple_trial.py`；新入口：`src/scripts/run_aret_ablation.py`，调度入口：`src/scripts/launch_aret_ablation.py`。',
           '- 原评估入口：`src/baselines/evaluation.py::linear_probe_five_fold_metrics`，本次直接调用，原文件未修改。',
           '- ACVG：`src/models/modular_research_ideas.py::prepare_views` 的 family=11、variant=1。node_drop 是随机节点掩码；subgraph 是带随机扰动的度优先节点保留，不是真正随机游走子图。节点不从Data中删除，而是置零并去掉关联边。',
           '- RDTG：`src/models/research_ideas.py::_family2` 的 variant=9。d_i是两个归一化cycle视图的均方差，按批次总体标准差（ddof=0，最小1e-5）标准化，g_i=sigmoid(-z_i)。cycle是节点与均值邻居差的可微代理，不是精确环拓扑。',
           '- LTCP：`src/models/research_ideas.py::_bundle_from_history` 与 `_family10` 的 variant=10。实际GIN history有4个状态，选索引0、2、3；坐标为s0−2s2+s3，经过128→32→128的Tanh瓶颈。几何损失比较投影内部距离矩阵与已detach的node/cycle交叉距离矩阵。',
           '- 三模块共享及残差混合位置：`src/models/modular_triple_research_ideas.py`、`src/models/modular_research_ideas.py::_module_transform`。M2增益1.5、M7增益17/9；并非直接把模块输出替换原表示。温度0.2。',
           '- 历史五数据集配置确认seed42、40轮、Adam、lr0.001、weight_decay=1e-5、hidden32、3层、表示128、batch128（代码对COLLAB实际64）、5折线性探针200轮。辅助权重RDTG=0.1875、LTCP=1.25。历史结果数值未复用。',
           '- 无法从仓库确认：该组合原始多种子集合、历史环境完整版本、历史逐样本划分文件。原代码明确没有早停或独立验证集。通用train.yaml不是该入口的实际训练设置。',
           '- 本次新增统一设置：训练种子0至4、所有种子固定split_seed42、严格确定性计算、独立DataLoader随机生成器及线程数1。Full也从头重跑，无旧checkpoint初始化。',
           '- 额外验证诊断：每个外折单独用60%训练、20%验证、20%测试的索引训练固定轮数诊断探针，仅记录训练/验证损失；随后调用原函数在80%上重训固定200轮并测试20%。没有根据验证结果选模型，不改变原测试评估流程。',
           '- 自监督预训练使用全部无标签图，包含测试图；这是原仓库传导式协议。NMI/ARI基于线性分类预测，不是另行运行的无监督聚类。',
           '', '## 最小反事实及开关验证','',
           '- 新模型继承原模型。全部开关开启直接调用原方法；原模型、评估及增强文件的SHA256与运行前快照一致。`smoke/switch_verification.json` 保存增强、损失、诊断、梯度、更新后权重的逐位比较。',
           '- w/o ACVG：仅将第二支subgraph改为已有node_drop；保留5%和2%节点预算、随机数消耗及逐图保留数量。原BalanceGCL的edge_drop/attr_mask也使用不同扰动类型且不能匹配节点预算，因此这里选已有node_drop作为对称基准；这个取舍是本次消融定义。',
           '- w/o RDTG：去掉gate×cycle残差和专属辅助项，保留归一化graph head与外层混合（1−a）G+a×graph_head(G)。直接返回sum-pool会将占主导的原−0.5G变为+G，带来尺度及符号混杂。当前基准不等同于未经适配的BalanceGCL。',
           '- w/o LTCP：三个原scale head都读取最后一层，用三个归一化终点坐标的平均值替代轨迹二阶差分，再送入相同瓶颈；保持三个head全部活跃、输出128、参数容量一致、原外层残差系数、M7困难负样本路径和其他接口不变。移除专属几何损失。',
           '- Backbone同时使用上述三个基准，没有将输出张量置零。辅助项为零只是表示该专属目标不参与总损失。',
           '- 全部八配置在MUTAG、seed0运行2轮预训练、5轮验证诊断/测试探针，完成五折；正式为40/200轮。smoke与正式产物目录独立。',
           '- 每批检查损失和梯度有限、关闭模块的专属损失为零；检查最终参数、表示、指标有限。逐任务记录开关、初始化哈希、代码哈希、划分哈希、数据哈希和完整命令。',
           '', '## 机制诊断','',
           '- `diagnostics/*.json` 记录每次训练第一批实际增强图的边集合Jaccard和边数量；训练结束后在该固定批次eval模式记录d_i、标准化分歧、g_i的逐样本值及分位数、门控前后范数和终点/轨迹表示差异。训练目标和BN状态不受诊断影响。',
           '- `logs/*.epochs.jsonl` 保存每轮真实训练目标、两个专属辅助损失及加权辅助总和。',
           '- 诊断分布仅代表固定第一批的训练后eval状态，不代表整个训练过程或数据集。关闭模块时同公式算出的诊断是对实际编码状态的反事实计算，不是生效门控；下文仅统计Full的机制量。',
           '- 未报告节点保留重叠度：存在原始全零节点特征，不能从增强后x可靠还原节点掩码；不生成代理数据。', '']
    full_jobs=[j for j in jobs if j['configuration']=='Full']
    if full_jobs:
        lines += ['| 数据集 | Full诊断批次数 | 边Jaccard均值 | 门控/未门控图范数比均值 | gate均值 |','|---|---:|---:|---:|---:|']
        for ds in DATASETS:
            dsdi=[json.loads(Path(j['diagnostic_path']).read_text()) for j in full_jobs if j['dataset']==ds]
            if dsdi:lines.append(f'| {ds} | {len(dsdi)} | {np.mean([d["view_edge_jaccard"] for d in dsdi]):.4f} | {np.mean([d["gated_to_ungated_norm_ratio"]["mean"] for d in dsdi]):.4f} | {np.mean([d["g_i"]["mean"] for d in dsdi]):.4f} |')
    lines += ['', '## 真实结果与研究问题','']
    if good:
        for c,name in [('w/o ACVG','ACVG'),('w/o RDTG','RDTG'),('w/o LTCP','LTCP'),('Backbone','三个模块整体')]:
            rs=[r for r in good if r['configuration']==c]
            if not rs:continue
            pos=sum(r['mean_full_minus_ablated']>1e-12 for r in rs);neg=sum(r['mean_full_minus_ablated'] < -1e-12 for r in rs)
            ci_pos=sum(r['paired_ci95_low']>0 for r in rs);ci_neg=sum(r['paired_ci95_high']<0 for r in rs)
            lines.append(f'- {name}：Full−{c}在{len(rs)}个已完整配对的数据集×指标单元中，{pos}个为正、{neg}个为负、{len(rs)-pos-neg}个持平；95%区间全正{ci_pos}个、全负{ci_neg}个。这些单元并非独立重复。')
            for ds in DATASETS:
                dr=[r for r in rs if r['dataset']==ds]
                if dr:lines.append('  - '+ds+'：'+ '，'.join(f'{r["metric"]} {r["mean_full_minus_ablated"]:+.3f}' for r in dr)+'（百分点）。')
        lines += ['', '- “模块带来增益”：只能限定在配对均值为正的数据集、指标和当前固定设置，区间跨零时证据较弱；完整表同时保留所有负收益。',
                  '- “模块具有必要性”：本实验只能检查相对收益，不能证明任何模块具有理论或普遍必要性；消融仍能训练且得到非零表现。',
                  '- “三个模块具有互补性”：需要结合单模块、leave-one-out和Backbone比较；不能仅凭Full优于Backbone概括全部数据集。',
                  '- “三个模块存在协同效应”：使用全因子差中差及Full相对单模块加和的超额检验；正超额且区间高于零才提供对应单元的证据，不能从单个Full最高分推断。']
    else:lines.append('尚无完整五种子配对，暂不回答模块增益或论文主张。')
    if inter:
        lines += ['', '## 全因子交互','', '二阶交互是在第三模块关闭/开启两种背景下差中差的平均；三阶交互为八角点交替和；超额=Full−ACVG only−RDTG only−LTCP only+2×Backbone。逐种子值、均值与95%区间见 `factorial_interactions.csv`。', '', '| 交互项 | 完整单元数 | 均值正/负 | 95%区间全正/全负 |','|---|---:|---:|---:|']
        for effect in dict.fromkeys(r['effect'] for r in inter):
            es=[r for r in inter if r['effect']==effect and r['n']==5]
            if es:lines.append(f'| {effect} | {len(es)} | {sum(r["mean"]>0 for r in es)} / {sum(r["mean"]<0 for r in es)} | {sum(r["ci95_low"]>0 for r in es)} / {sum(r["ci95_high"]<0 for r in es)} |')
    fails=[a for a in m['attempts'] if a['status']=='failed']
    lines += ['', '## 异常、失败和限制','',
              '- 初始Python导入sklearn时缺少系统CXXABI_1.3.15。实验进程使用当前Conda环境lib目录的LD_LIBRARY_PATH，导入验证成功；未改系统库。torch_scatter/torch_sparse未安装，当前代码使用PyG/PyTorch路径，smoke检查兼容性。',
              '- 两张4090已有其他用户任务，本调度器每卡最多一个任务，持续保存GPU状态；没有停止其他进程。环境实际版本与GPU信息见run_manifest.json。',
              f'- 正式/冒烟已记录的失败尝试总数：{len(fails)}。所有错误日志、部分产物及重跑记录保留；无不利种子删除。']
    for f in fails:lines.append(f'- 失败：{f["phase"]} / {f["configuration"]} / {f["dataset"]} / seed{f["seed"]} / attempt{f["attempt"]}；日志 `{f["log"]}`。')
    lines += ['- 仅5个训练种子；固定数据划分的区间只反映训练随机性，不衡量重新划分、数据采样或外部泛化不确定性。Student-t区间未做多重比较校正。',
              '- 既有模块残差系数很大，原始图读出可以占主导；LTCP差异受这一实际执行路径约束，不将论文期望替代代码证据。',
              '- 本实验不加入其他数据集、模块或历史候选得分，不重新调参，不修改论文正文、现有LaTeX或PDF。',
              '- 初始未跟踪文件AReT-GCL_ICLR2027_story.pdf保留。完整代码前后快照位于audit/source_before与audit/source_used。', '']
    (OUT/'diagnostics.md').write_text('\n'.join(lines),encoding='utf8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    manifest=json.loads((OUT/'run_manifest.json').read_text());jobs=gather(manifest)
    rows,values,summary,paired=recompute()
    write_csv(OUT/'ablation_summary.csv',summary)
    write_csv(OUT/'paired_differences.csv',paired,['configuration','dataset','seed','metric','full','ablated','full_minus_ablated','full_win','tie'])
    inter=interactions(values)
    write_csv(OUT/'factorial_interactions.csv',inter,['dataset','metric','effect','n','mean','std','ci95_low','ci95_high','per_seed_differences'])
    tables(summary,len(jobs));diagnostics(manifest,jobs,summary,inter)
    if args.verify:
        assert len(jobs)==200 and len(rows)==200 and all(r['status']=='completed' for r in rows)
        assert len({(r['configuration'],r['dataset'],r['seed']) for r in rows})==200
        assert len(summary)==160 and all(r['n_seeds']==5 for r in summary)
        assert len(paired)==700 and len(inter)==100 and all(r['n']==5 for r in inter)
        for ds in DATASETS:
            for seed in SEEDS:
                group=[j for j in jobs if j['dataset']==ds and j['seed']==seed]
                assert len({j['initial_encoder_sha256'] for j in group})==1
        before=manifest['original_source_sha256']
        assert all(sha(ROOT/f)==h for f,h in before.items())
        assert (OUT/'smoke/switch_verification.json').exists()
        assert len(list((OUT/'smoke/jobs').glob('*.json')))==8
        # 独立读回CSV，再次计算并逐字段比较，确保表格完全来自原始CSV。
        _,_,again,again_pairs=recompute();assert again==summary and again_pairs==paired
        for f in ['run_manifest.json','per_seed_results.csv','ablation_summary.csv','ablation_table.md','ablation_table.tex','diagnostics.md']:
            assert (OUT/f).read_text(encoding='utf8')
        save(OUT/'acceptance.json',dict(status='passed',checked_at=now(),formal_runs=200,core_runs=125,extension_runs=75,smoke_runs=8,seed_count=5,summary_rows=160,paired_rows=700,factorial_rows=100,csv_recomputation=True,identical_splits_initialization_and_source=True,all_original_sources_unchanged=True))
    print(f'汇总完成：{len(jobs)}/200 正式训练，逐种子记录200行',flush=True)

if __name__=='__main__':main()
