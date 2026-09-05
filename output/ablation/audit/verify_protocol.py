"""独立检查固定划分、结果字段及 smoke 重放的确定性。"""
import json,hashlib,sys
from pathlib import Path
import torch
ROOT=Path('/home/Lim/Project-xmlg');sys.path.insert(0,str(ROOT))
from src.scripts.run_aret_ablation import OUT,DATASETS,CONFIGS,parameters,load_data,runtime,model,save
from src.baselines.gcl_baselines import set_seed
from torch_geometric.loader import DataLoader
runtime()
manifest=json.loads((OUT/'run_manifest.json').read_text())
for name in DATASETS:
 s=json.loads((OUT/'splits'/f'{name}.json').read_text());n=manifest['datasets'][name]['graph_count'];seen=[]
 for f in s['folds']:
  tr,te,dt,dv=(set(f[k]) for k in ['train','test','diagnostic_train','diagnostic_validation'])
  assert not tr&te and len(tr|te)==n
  assert not dt&dv and not dt&te and not dv&te and dt|dv==tr
  seen+=f['test']
 assert sorted(seen)==list(range(n))
for config in CONFIGS:
 slug=config.replace('w/o ','without_').replace(' ','_')
 r=json.loads((OUT/'smoke/jobs'/f'{slug}__MUTAG__seed0.json').read_text())
 assert r['phase']=='smoke' and r['settings']['epochs']==2 and len(r['validation_diagnostics'])==5
 assert set(r['metrics'])=={'ACC','NMI','ARI','Macro-F1'}
 assert r['split_sha256']==manifest['datasets']['MUTAG']['split_sha256']
# 在CPU验证划分；重放训练使用与首个Full smoke相同GPU，完整比较模型状态。
graphs,dim,classes,_=load_data('MUTAG');set_seed(0);m=model('Full',dim,classes).to('cuda:0')
opt=torch.optim.Adam(m.parameters(),lr=.001,weight_decay=1e-5)
loader=DataLoader(graphs,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(0),num_workers=0)
for epoch in range(2):
 m.train()
 for batch in loader:
  a,b=m.prepare_views(batch.to('cuda:0'));opt.zero_grad(set_to_none=True);loss,_=m.idea_loss(a,b);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
stored=torch.load(OUT/'smoke/checkpoints/Full__MUTAG__seed0.pt',map_location='cuda:0',weights_only=True)
assert all(torch.equal(v,stored['model'][k]) for k,v in m.state_dict().items())
save(OUT/'smoke/protocol_verification.json',dict(status='passed',five_dataset_splits_disjoint=True,test_coverage_once=True,eight_smoke_validation_and_test_complete=True,independent_full_smoke_training_replay_exact=True))
print('五数据集划分、八配置字段、Full独立训练重放全部通过')
