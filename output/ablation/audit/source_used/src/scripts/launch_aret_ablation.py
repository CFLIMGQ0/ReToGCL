"""可靠会话中的两卡调度：先八配置 smoke，再核心125次，最后扩展75次。"""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.scripts.run_aret_ablation import OUT, CONFIGS, DATASETS, SEEDS, job_id, save, now


def environment():
    env=os.environ.copy()
    lib=str(Path(sys.executable).resolve().parent.parent/'lib')
    env.update(LD_LIBRARY_PATH=lib+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''),
               CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',PYTHONUNBUFFERED='1')
    return env


def gpu_state():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(r.split(',')[0]):[int(v.strip()) for v in r.split(',')[1:]] for r in s.strip().splitlines()}


def run_stage(configs, datasets, seeds, phase, env):
    base=OUT if phase=='formal' else OUT/'smoke'
    # 大数据集优先启动以减少末尾等待；配置展示顺序始终固定。
    ordered=[d for d in ['COLLAB','NCI1','Mutagenicity','MUTAG','PTC_MR'] if d in datasets]
    tasks=[(c,d,s) for s in seeds for d in ordered for c in configs]
    queue=[t for t in tasks if not (base/'jobs'/f'{job_id(*t)}.json').exists()]
    running={};failed=[];manifest=json.loads((OUT/'run_manifest.json').read_text())
    attempt_counts={}
    for a in manifest['attempts']:
        key=(a['phase'],a['configuration'],a['dataset'],a['seed'])
        attempt_counts[key]=max(attempt_counts.get(key,0),a['attempt'])
    while queue or running:
        state=gpu_state()
        with (OUT/'logs/gpu_monitor.jsonl').open('a') as f:f.write(json.dumps(dict(time=now(),gpus=state,phase=phase))+'\n')
        for device in [0,1]:
            if device in running or not queue or state[device][0]<9500:continue
            t=queue.pop(0);config,dataset,seed=t;ident=job_id(*t)
            key=(phase,*t);attempt=attempt_counts.get(key,0)+1;attempt_counts[key]=attempt
            cmd=[sys.executable,str(ROOT/'src/scripts/run_aret_ablation.py'),'run','--phase',phase,'--configuration',config,'--dataset',dataset,'--seed',str(seed),'--device',f'cuda:{device}']
            lp=base/'logs'/f'{ident}.attempt{attempt}.log';lp.parent.mkdir(parents=True,exist_ok=True)
            stream=lp.open('x')
            record=dict(phase=phase,configuration=config,dataset=dataset,seed=seed,attempt=attempt,
                        started_at=now(),ended_at=None,command=cmd,environment={k:env[k] for k in ['LD_LIBRARY_PATH','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONHASHSEED']},log=str(lp),status='running')
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            record['pid']=proc.pid;manifest['commands'].append(cmd);manifest['attempts'].append(record)
            running[device]=(proc,t,stream,record);print(f'启动 {phase} {ident} GPU{device} 第{attempt}次',flush=True)
        for device,(proc,t,stream,record) in list(running.items()):
            code=proc.poll()
            if code is None:continue
            stream.close();del running[device];record['exit_code']=code;record['ended_at']=now()
            path=base/'jobs'/f'{job_id(*t)}.json'
            record['status']='completed' if code==0 and path.exists() else 'failed'
            if record['status']=='failed':
                # 原错误日志永久保留；部分输出移至失败目录，使同配置同种子可重跑。
                archive=base/'failed_attempts'/f'{job_id(*t)}.attempt{record["attempt"]}'
                for sub,ext in [('logs','.epochs.jsonl'),('jobs','.json'),('checkpoints','.pt'),('diagnostics','.json')]:
                    partial=base/sub/f'{job_id(*t)}{ext}'
                    if partial.exists():archive.mkdir(parents=True,exist_ok=True);partial.rename(archive/partial.name)
                failed.append(dict(record));print(f'失败 {job_id(*t)}，日志 {record["log"]}，暂停后续启动以便查因',flush=True)
                # 不自动盲重跑；让监控 Agent 审查并修复后显式恢复。
                queue.clear()
            else:
                r=json.loads(path.read_text());print(f'完成 {phase} {job_id(*t)} 用时{r["duration_seconds"]:.1f}秒',flush=True)
        manifest['status']='running_'+phase
        save(OUT/'run_manifest.json',manifest)
        completed=sum((base/'jobs'/f'{job_id(*t)}.json').exists() for t in tasks)
        save(OUT/'live_status.json',dict(updated_at=now(),phase=phase,total=len(tasks),completed=completed,queued=len(queue),running=[dict(configuration=t[1][0],dataset=t[1][1],seed=t[1][2],pid=t[0].pid,device=d) for d,t in running.items()],failures=failed))
        if queue or running:time.sleep(5)
    if failed:raise RuntimeError(f'{len(failed)} 次失败，错误日志已保存；需查因后重启同一调度器')
    if phase=='formal':subprocess.run([sys.executable,str(ROOT/'src/scripts/summarize_aret_ablation.py')],cwd=ROOT,env=env,check=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=environment()
    save(OUT/'controller.json',dict(pid=os.getpid(),started_at=now(),command=sys.argv,session=os.getenv('TMUX'),environment={k:env.get(k) for k in ['LD_LIBRARY_PATH','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONHASHSEED']}))
    try:
        if not (OUT/'smoke/switch_verification.json').exists():
            with (OUT/'smoke/verification.log').open('a') as f:
                subprocess.run([sys.executable,str(ROOT/'src/scripts/run_aret_ablation.py'),'verify','--device','cuda:0'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        run_stage(list(CONFIGS),['MUTAG'],[0],'smoke',env)
        m=json.loads((OUT/'run_manifest.json').read_text());m['smoke_passed_at']=now();save(OUT/'run_manifest.json',m)
        run_stage(list(CONFIGS)[:5],DATASETS,SEEDS,'formal',env)
        m=json.loads((OUT/'run_manifest.json').read_text());m['core_completed_at']=now();m['extension_status']='核心125次全部成功；按同一设置继续75次扩展';save(OUT/'run_manifest.json',m)
        run_stage(list(CONFIGS)[5:],DATASETS,SEEDS,'formal',env)
        m=json.loads((OUT/'run_manifest.json').read_text());m.update(status='completed',extension_status='completed',ended_at=now());save(OUT/'run_manifest.json',m)
        subprocess.run([sys.executable,str(ROOT/'src/scripts/summarize_aret_ablation.py'),'--verify'],cwd=ROOT,env=env,check=True)
    except Exception:
        with (OUT/'logs/controller_errors.log').open('a') as f:f.write(now()+'\n'+traceback.format_exc())
        m=json.loads((OUT/'run_manifest.json').read_text());m['status']='failed_requires_review';save(OUT/'run_manifest.json',m)
        raise

if __name__=='__main__':main()
