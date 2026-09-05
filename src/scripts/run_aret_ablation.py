"""AReT-GCL 固定五数据集模块消融：审计、单次训练与完整性验证。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from src.baselines.gcl_baselines import ensure_features, set_seed
from src.baselines.evaluation import stratified_folds, linear_probe_five_fold_metrics
from src.models.aret_gcl_ablation import AReTGCLAblation, CONFIGS, IDEA_IDS
from src.models.modular_triple_research_ideas import build_modular_triple_research_idea_model

OUT = ROOT / 'output/ablation'
DATASETS = ['NCI1', 'Mutagenicity', 'COLLAB', 'MUTAG', 'PTC_MR']
METRICS = {'ACC': 'accuracy', 'NMI': 'nmi', 'ARI': 'ari', 'Macro-F1': 'macro_f1'}
SEEDS = list(range(5))
SETTINGS = dict(hidden_dim=32, layers=3, projection_dim=128, epochs=40, probe_epochs=200,
                batch_size=128, collab_batch_size=64, folds=5, split_seed=42,
                optimizer='Adam', learning_rate=0.001, weight_decay=0.00001,
                clip_grad_norm=5., early_stopping=False, probe_learning_rate=0.05,
                probe_weight_decay=0.0001, num_workers=0, torch_threads=1,
                deterministic_algorithms=True, auxiliary_weight=0.25)


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug(config):
    return config.replace('w/o ', 'without_').replace(' ', '_')


def job_id(config, dataset, seed):
    return f'{slug(config)}__{dataset}__seed{seed}'


def parameters():
    return tuple(json.loads((OUT/'audit/original_settings.json').read_text())[0]['idea_parameters'])


def runtime():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_data(name):
    assert name in DATASETS
    dataset = TUDataset(root=str(ROOT/'datasets/main_data'), name=name, cleaned=False, use_node_attr=False)
    graphs = []
    digest = hashlib.sha256()
    for i in tqdm(range(len(dataset)), desc=f'{name} 数据校验', unit='图', leave=False):
        g = ensure_features(dataset[i].clone())
        g.y = g.y.long().view(1)
        for x in (g.x, g.edge_index, g.y):
            digest.update(str((str(x.dtype), list(x.shape))).encode())
            digest.update(x.contiguous().numpy().tobytes())
        assert torch.isfinite(g.x).all()
        graphs.append(g)
    return graphs, max(1, int(dataset.num_node_features)), int(dataset.num_classes), digest.hexdigest()


def split_record(labels):
    splits = stratified_folds(labels, SETTINGS['folds'], SETTINGS['split_seed'])
    folds = []
    for i, test in enumerate(splits):
        val = splits[(i+1) % len(splits)]
        diagnostic_train = torch.cat([s for j,s in enumerate(splits) if j not in (i,(i+1)%len(splits))]).sort().values
        train = torch.cat([s for j,s in enumerate(splits) if j != i]).sort().values
        folds.append(dict(fold=i+1, train=train.tolist(), test=test.tolist(), diagnostic_train=diagnostic_train.tolist(), diagnostic_validation=val.tolist()))
    return dict(split_seed=42, folds=folds, protocol='全图无标签预训练；原始80/20五折线性测试；另行60/20/20固定轮数验证诊断，不参与模型选择')


def code_hashes():
    files = json.loads((OUT/'audit/initial_audit.json').read_text())['original_source_sha256']
    files.update({str(p.relative_to(ROOT)): '' for p in [ROOT/'src/models/aret_gcl_ablation.py', ROOT/'src/scripts/run_aret_ablation.py', ROOT/'src/scripts/launch_aret_ablation.py', ROOT/'src/scripts/summarize_aret_ablation.py']})
    return {f:sha(ROOT/f) for f in files}


def setup():
    audit = json.loads((OUT/'audit/initial_audit.json').read_text())
    env = dict(python=sys.version, executable=sys.executable, platform=platform.platform(), cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), gpu_count=torch.cuda.device_count(), LD_LIBRARY_PATH=os.getenv('LD_LIBRARY_PATH'),
               CUBLAS_WORKSPACE_CONFIG=os.getenv('CUBLAS_WORKSPACE_CONFIG'), OMP_NUM_THREADS=os.getenv('OMP_NUM_THREADS'))
    for name in ['torch', 'torch_geometric', 'numpy', 'scipy', 'sklearn', 'yaml', 'tqdm', 'torch_scatter', 'torch_sparse']:
        try: env[name] = importlib.import_module(name).__version__
        except Exception as e: env[name] = repr(e)
    datasets = {}
    for name in DATASETS:
        graphs, dim, classes, digest = load_data(name)
        labels = torch.cat([g.y for g in graphs])
        sp = OUT/'splits'/f'{name}.json'
        save(sp, split_record(labels))
        datasets[name] = dict(graph_count=len(graphs), in_dim=dim, classes=classes,
                              label_counts=torch.bincount(labels).tolist(), preprocessed_sha256=digest,
                              split_path=str(sp), split_sha256=sha(sp), preprocessing='现有TUDataset，cleaned=False，use_node_attr=False；无特征时常数1，其余float；不丢弃图',
                              effective_batch_size=64 if name=='COLLAB' else 128)
        del graphs
    m = {**audit, 'status':'prepared', 'environment':env, 'settings':SETTINGS,
         'seeds':SEEDS, 'seed_count':len(SEEDS), 'seed_selection_reason':'该三模块在五个指定数据集仅确认历史种子42，无法确认原始多种子集合；两张4090均被共享，固定使用0至4共五种子，不按结果筛选',
         'datasets':datasets, 'metrics':list(METRICS), 'idea_ids':IDEA_IDS, 'idea_parameters':parameters(),
         'configurations':{k:dict(zip(['use_acvg','use_rdtg','use_ltcp'],v)) for k,v in CONFIGS.items()},
         'core_configurations':list(CONFIGS)[:5], 'extension_configurations':list(CONFIGS)[5:],
         'expected_core_runs':125, 'expected_extension_runs':75, 'extension_status':'核心完成后依据实测资源决定',
         'commands':[], 'attempts':[], 'ended_at':None,
         'protocol_notes':['原始五折测试没有独立验证集；保留原有80/20评估函数，新增固定200轮60/20验证诊断，验证诊断不选超参数、不早停，不进入结果统计。',
                           '每种子先对五折测试分数取等权平均，再跨种子计算样本标准差；种子是配对统计单位，折不是独立种子。',
                           'NMI/ARI由监督线性分类器预测计算，属于分类预测与标签的一致性，不能表述为无监督聚类结果。',
                           '自监督编码器在所有无标签图上预训练，包含最终测试图；属于仓库原有传导式协议。',
                           'RDTG关闭保留归一化graph head和原残差系数，只去除gate*cycle及专属损失；LTCP关闭保留三个head和瓶颈，三个head都读最终状态并平均。',
                           '确定性模式、固定split_seed=42、五种子集合和额外验证诊断是本次统一设置，不冒称历史原始设置。']}
    save(OUT/'run_manifest.json', m)
    print('审计与固定划分保存完成', flush=True)


def model(config, in_dim, classes):
    return AReTGCLAblation(in_dim,32,3,classes,parameters(),**dict(zip(['use_acvg','use_rdtg','use_ltcp'],CONFIGS[config])))


def finite_model(m):
    for n,p in m.named_parameters():
        if not torch.isfinite(p).all(): raise RuntimeError(f'参数非有限：{n}')
        if p.grad is not None and not torch.isfinite(p.grad).all(): raise RuntimeError(f'梯度非有限：{n}')


def verify_switches(device):
    graphs, dim, classes, _ = load_data('MUTAG')
    batch = Batch.from_data_list(graphs[:16]).to(device)
    set_seed(0); legacy = build_modular_triple_research_idea_model(IDEA_IDS,dim,32,3,classes,parameters()).to(device)
    set_seed(0); full = model('Full',dim,classes).to(device)
    assert list(legacy.state_dict()) == list(full.state_dict())
    assert all(torch.equal(v,full.state_dict()[k]) for k,v in legacy.state_dict().items())
    checks = []
    for epoch in range(2):
        outputs=[]
        for m in [legacy,full]:
            set_seed(100+epoch);m.train();m.zero_grad(set_to_none=True)
            a,b=m.prepare_views(batch);loss,d=m.idea_loss(a,b);loss.backward()
            outputs.append((a,b,loss.detach(),d,{n:p.grad.clone() if p.grad is not None else None for n,p in m.named_parameters()}))
        x,y=outputs
        for i in [0,1]:
            for field in ['x','edge_index','batch','y']:assert torch.equal(getattr(x[i],field),getattr(y[i],field))
        assert torch.equal(x[2],y[2]) and x[3]==y[3]
        for k,v in x[4].items():assert (v is None and y[4][k] is None) or (v is not None and y[4][k] is not None and torch.equal(v,y[4][k]))
        for m in [legacy,full]:
            torch.optim.Adam(m.parameters(),lr=.001,weight_decay=1e-5).step()
        assert all(torch.equal(v,full.state_dict()[k]) for k,v in legacy.state_dict().items())
        checks.append(dict(step=epoch+1, exact_views_loss_diagnostics_gradients_and_updated_parameters=True))
    records=[]
    for config in CONFIGS:
        set_seed(0);m=model(config,dim,classes).to(device);m.train()
        set_seed(777);a,b=m.prepare_views(batch);loss,d=m.idea_loss(a,b);loss.backward();finite_model(m)
        assert torch.isfinite(loss)
        assert m.output_dim==128
        if not m.use_rdtg:assert d['second_auxiliary']==0.
        if not m.use_ltcp:assert d['third_auxiliary']==0.
        with torch.no_grad():
            embedding=m.encode(batch)
            assert embedding.shape==(16,128) and torch.isfinite(embedding).all()
        active={n:p.numel() for n,p in m.named_parameters() if p.grad is not None}
        for j in range(3): assert f'third.scale_heads.{j}.weight' in active
        records.append(dict(configuration=config, switches=CONFIGS[config], loss=loss.item(),diagnostics=d,
                            parameter_count=sum(p.numel() for p in m.parameters()), active_parameter_count=sum(active.values()),
                            active_parameter_names=list(active), embedding_shape=list(embedding.shape)))
    assert len({r['parameter_count'] for r in records})==1
    assert len({round(r['loss'],8) for r in records})==8
    save(OUT/'smoke/switch_verification.json',dict(status='passed',device=str(device),full_exact_equivalence=checks,configurations=records))
    print('八配置开关验证通过；Full增强、损失、梯度、更新后权重逐位相同',flush=True)


def validate_probe(embeddings, labels, device, seed, epochs, split):
    # 训练/验证诊断不会接触该折测试图标签，不用于模型/轮数/超参数选择。
    records=[]
    for fold in tqdm(split['folds'],desc='固定轮数验证诊断',unit='折',leave=False):
        train=torch.tensor(fold['diagnostic_train']);val=torch.tensor(fold['diagnostic_validation'])
        x,y=embeddings[train],embeddings[val]
        mean=x.mean(0,keepdim=True);std=x.std(0,keepdim=True).clamp_min(1e-6)
        x,y=((x-mean)/std).to(device),((y-mean)/std).to(device)
        ty,vy=labels[train].to(device),labels[val].to(device)
        torch.manual_seed(seed+fold['fold'])
        c=torch.nn.Linear(embeddings.size(1),int(labels.max())+1).to(device)
        opt=torch.optim.Adam(c.parameters(),lr=.05,weight_decay=1e-4)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(c(x),ty)
            if not torch.isfinite(loss):raise RuntimeError('验证诊断探针损失非有限')
            loss.backward();opt.step()
        with torch.no_grad():vl=F.cross_entropy(c(y),vy).item()
        assert np.isfinite(vl)
        records.append(dict(fold=fold['fold'],training_loss=loss.item(),validation_loss=vl,epochs=epochs))
    return records


def run_one(args):
    start=now();begin=time.monotonic();device=torch.device(args.device)
    phase_dir=OUT if args.phase=='formal' else OUT/'smoke'
    ident=job_id(args.configuration,args.dataset,args.seed)
    result_path=phase_dir/'jobs'/f'{ident}.json'
    if result_path.exists():raise RuntimeError(f'拒绝覆盖已有结果：{result_path}')
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    actual_hashes=code_hashes()
    if manifest.get('code_sha256') and actual_hashes!=manifest['code_sha256']:raise RuntimeError('代码快照不匹配，拒绝混用实验版本')
    graphs,dim,classes,digest=load_data(args.dataset)
    assert digest==manifest['datasets'][args.dataset]['preprocessed_sha256']
    splitpath=OUT/'splits'/f'{args.dataset}.json'
    assert sha(splitpath)==manifest['datasets'][args.dataset]['split_sha256']
    split=json.loads(splitpath.read_text())
    labels=torch.cat([g.y for g in graphs])
    assert split_record(labels)==split
    set_seed(args.seed);m=model(args.configuration,dim,classes).to(device)
    initial_encoder_sha=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in m.encoder.parameters())).hexdigest()
    batch_size=64 if args.dataset=='COLLAB' else 128
    epochs=40 if args.phase=='formal' else 2
    probe_epochs=200 if args.phase=='formal' else 5
    opt=torch.optim.Adam(m.parameters(),lr=.001,weight_decay=1e-5)
    loader=DataLoader(graphs,batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed),num_workers=0)
    print(json.dumps(dict(configuration=args.configuration,dataset=args.dataset,seed=args.seed,switches=CONFIGS[args.configuration],epochs=epochs,phase=args.phase,split_sha256=sha(splitpath)),ensure_ascii=False),flush=True)
    epoch_path=phase_dir/'logs'/f'{ident}.epochs.jsonl';epoch_path.parent.mkdir(parents=True,exist_ok=True)
    diagnostic_input=None
    with epoch_path.open('x') as stream:
        for epoch in tqdm(range(1,epochs+1),desc=ident,unit='轮'):
            m.train();sums={};count=0
            for batch in loader:
                batch=batch.to(device);a,b=m.prepare_views(batch)
                if diagnostic_input is None:diagnostic_input=(a.clone(),b.clone())
                opt.zero_grad(set_to_none=True);loss,d=m.idea_loss(a,b)
                if not torch.isfinite(loss):raise RuntimeError(f'第{epoch}轮损失非有限')
                if not m.use_rdtg:assert d['second_auxiliary']==0.
                if not m.use_ltcp:assert d['third_auxiliary']==0.
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.)
                if not torch.isfinite(norm):raise RuntimeError(f'第{epoch}轮梯度非有限')
                opt.step();n=batch.num_graphs;count+=n
                for k,v in {'loss':loss.item(),**d}.items():sums[k]=sums.get(k,0.)+float(v)*n
            record=dict(epoch=epoch,**{k:v/count for k,v in sums.items()})
            if not all(np.isfinite(v) for v in record.values()):raise RuntimeError('诊断含非有限值')
            stream.write(json.dumps(record)+'\n');stream.flush()
    finite_model(m)
    m.eval();emb=[]
    with torch.no_grad():
        for batch in tqdm(DataLoader(graphs,batch_size=batch_size,shuffle=False),desc='提取最终表示',unit='批',leave=False):
            v=m.encode(batch.to(device))
            assert torch.isfinite(v).all();emb.append(v.cpu())
    embeddings=torch.cat(emb)
    val=validate_probe(embeddings,labels,device,args.seed,probe_epochs,split)
    folds=linear_probe_five_fold_metrics(embeddings,labels,device=device,seed=args.seed,split_seed=42,folds=5,epochs=probe_epochs)
    raw={k:[float(v)*100 for v in folds[src]] for k,src in METRICS.items()}
    assert all(len(v)==5 and all(np.isfinite(x) for x in v) for v in raw.values())
    diag=m.mechanism_diagnostics(*diagnostic_input)
    a,b=diagnostic_input
    # 节点保留依据真实增强后的非零特征；原本全零特征不能可靠识别，因此同时记录边集合交并比。
    ea=set(map(tuple,a.edge_index.t().cpu().tolist()));eb=set(map(tuple,b.edge_index.t().cpu().tolist()))
    diag['view_edge_jaccard']=len(ea&eb)/max(1,len(ea|eb))
    diag['view_edge_counts']=[len(ea),len(eb)]
    diag['recording_scope']='固定第一训练批的两个真实增强图；训练结束后eval模式诊断；不声称覆盖全数据训练态分布'
    dp=phase_dir/'diagnostics'/f'{ident}.json';save(dp,diag)
    cp=phase_dir/'checkpoints'/f'{ident}.pt';cp.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),embeddings=embeddings,labels=labels,
                    configuration=args.configuration,seed=args.seed,dataset=args.dataset,code_sha256=actual_hashes),cp)
    result=dict(status='completed',phase=args.phase,configuration=args.configuration,dataset=args.dataset,seed=args.seed,
                switches=dict(zip(['use_acvg','use_rdtg','use_ltcp'],CONFIGS[args.configuration])),started_at=start,ended_at=now(),duration_seconds=time.monotonic()-begin,
                git_commit=manifest['git_commit'],code_sha256=actual_hashes,split_sha256=sha(splitpath),data_sha256=digest,
                initial_encoder_sha256=initial_encoder_sha,settings={**SETTINGS,'epochs':epochs,'probe_epochs':probe_epochs,'effective_batch_size':batch_size},
                metrics={k:float(np.mean(v)) for k,v in raw.items()},fold_metrics=raw,validation_diagnostics=val,
                diagnostic_path=str(dp),checkpoint_path=str(cp),parameter_count=sum(p.numel() for p in m.parameters()),
                device=str(device),cuda_peak_memory_mib=torch.cuda.max_memory_allocated(device)/1024**2,command=sys.argv)
    save(result_path,result);print('完成 '+ident+' '+json.dumps(result['metrics']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['setup','verify','run','freeze'])
    p.add_argument('--configuration',choices=list(CONFIGS),default='Full');p.add_argument('--dataset',choices=DATASETS,default='MUTAG')
    p.add_argument('--seed',type=int,choices=SEEDS,default=0);p.add_argument('--device',default='cuda:0');p.add_argument('--phase',choices=['smoke','formal'],default='smoke')
    args=p.parse_args();runtime()
    if args.action=='setup':setup()
    elif args.action=='verify':verify_switches(torch.device(args.device))
    elif args.action=='freeze':
        m=json.loads((OUT/'run_manifest.json').read_text());m['code_sha256']=code_hashes();m['code_frozen_at']=now();save(OUT/'run_manifest.json',m)
        for f in m['code_sha256']:
            target=OUT/'audit/source_used'/f;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/f).read_bytes())
        print('实验代码快照已冻结')
    else:run_one(args)

if __name__=='__main__':main()
