"""持续监控用单次快照；只读状态、真实轮次和GPU信息。"""
import json,subprocess,time
from pathlib import Path
out=Path('/home/Lim/Project-xmlg/output/ablation')
s=json.loads((out/'live_status.json').read_text())
m=json.loads((out/'run_manifest.json').read_text())
parts=[]
for r in s.get('running',[]):
 slug=r['configuration'].replace('w/o ','without_').replace(' ','_')
 p=out/('smoke' if s['phase']=='smoke' else '')/'logs'/f'{slug}__{r["dataset"]}__seed{r["seed"]}.epochs.jsonl'
 epoch=0
 if p.exists():
  lines=p.read_text().splitlines()
  if lines:epoch=json.loads(lines[-1])['epoch']
 parts.append(f'{r["dataset"]}/{r["configuration"]}/种子{r["seed"]}：{epoch}/40轮')
formal=len(list((out/'jobs').glob('*.json')))
failed=sum(a['status']=='failed' for a in m['attempts'])
message=f'正式实验已完成 {formal}/200（核心目标125次）；失败尝试{failed}次。'+('；'.join(parts) if parts else '当前无活动训练。')
print(json.dumps(dict(message=message,status=m['status'],formal_completed=formal,stage_completed=s['completed'],stage_total=s['total'],stale_seconds=time.time()-(out/'live_status.json').stat().st_mtime,acceptance=(out/'acceptance.json').exists()),ensure_ascii=False))
