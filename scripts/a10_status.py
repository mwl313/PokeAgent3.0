#!/usr/bin/env python3
"""Read-only, low-overhead A10 production status inspection."""
from __future__ import annotations
import argparse, json, os, re, subprocess, time
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'runs/a10_production_20261013_v2'
ITER_RE = re.compile(r'\[(rank(\d+))\] iteration (\d+)/(\d+): collected (\d+) games / (\d+) rows')
COMPLETE_RE = re.compile(r'\[(rank(\d+))\] iteration (\d+) complete')
UPDATE_RE = re.compile(r'\[(rank(\d+))\] update done in ([0-9.]+)s .*global matches (\d+)')
RECOMP_RE = re.compile(r'\[(rank(\d+))\] recompute gate: .*max\|diff\|=([0-9.eE+-]+).*within=(True|False)')

def _proc_cmd(pid: int):
    try: return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
    except OSError: return ''

def find_processes(root: Path):
    token = str(root)
    launchers=[]; workers=[]
    for p in Path('/proc').glob('[0-9]*'):
        try: pid=int(p.name)
        except ValueError: continue
        cmd=_proc_cmd(pid)
        if token not in cmd or 'run_ddp_ppo.py' not in cmd: continue
        (workers if '--worker' in cmd else launchers).append({'pid':pid,'cmd':cmd})
    return launchers, workers

def parse_logs(root: Path):
    ranks={}; last_ts=0.0; current_matches=0; last_complete=0
    for rank in (0,1):
        path=root/f'report.json.rank{rank}.log'
        data={"rank":rank,"iteration":0,"complete_iteration":0,"matches":0,"last_progress":None,"last_update_s":None,"recompute_max_diff":None,"recompute_within":None,"error_lines":[]}
        if path.exists():
            st=path.stat(); last_ts=max(last_ts, st.st_mtime)
            try: lines=path.read_text(errors='replace').splitlines()
            except OSError: lines=[]
            for line in lines[-500:]:
                m=ITER_RE.search(line)
                if m: data['iteration']=int(m.group(3)); data['last_progress']=line
                m=COMPLETE_RE.search(line)
                if m: data['complete_iteration']=max(data['complete_iteration'],int(m.group(3))); data['last_progress']=line
                m=UPDATE_RE.search(line)
                if m: data['last_update_s']=float(m.group(3)); data['matches']=max(data['matches'],int(m.group(4)))
                m=RECOMP_RE.search(line)
                if m: data['recompute_max_diff']=float(m.group(2)); data['recompute_within']=(m.group(4)=='True')
                if re.search(r'\b(?:Traceback|RuntimeError|FloatingPointError|NCCL error|NaN|Inf|failed|ERROR)\b',line,re.I): data['error_lines'].append(line[-300:])
        current_matches=max(current_matches,data['matches']); last_complete=max(last_complete,data['complete_iteration']); ranks[rank]=data
    checkpoint=None; latest=root/'rolling/LATEST'
    if latest.exists():
        name=latest.read_text(errors='replace').strip(); p=root/'rolling'/name
        if name and (p/'manifest.json').exists():
            checkpoint={'generation':name,'manifest':str(p/'manifest.json'),'mtime':p.stat().st_mtime}
    return {'ranks':ranks,'committed_matches':current_matches,'last_complete_iteration':last_complete,'last_progress_epoch':last_ts,'latest_checkpoint':checkpoint}

def disk(root: Path):
    v=os.statvfs(root); return {'free_bytes':v.f_bavail*v.f_frsize,'total_bytes':v.f_blocks*v.f_frsize}

def gpu():
    try:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,power.limit,power.draw,temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=2)
        rows=[]
        for line in out.splitlines():
            x=[z.strip() for z in line.split(',')];
            if len(x)>=7: rows.append({'index':x[0],'name':x[1],'power_limit_w':float(x[2]),'power_draw_w':float(x[3]),'temperature_c':float(x[4]),'utilization_pct':float(x[5]),'memory_used_mib':float(x[6])})
        return rows
    except Exception as e: return {'error':str(e)}

def snapshot(root=DEFAULT_ROOT):
    launchers,workers=find_processes(root); data=parse_logs(root); data.update({'root':str(root),'timestamp':time.time(),'launchers':launchers,'workers':workers,'disk':disk(root),'gpus':gpu(),'duplicate_launcher':len(launchers)!=1}); return data

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=DEFAULT_ROOT); ap.add_argument('--json',action='store_true'); a=ap.parse_args(); s=snapshot(a.root)
    if a.json: print(json.dumps(s,sort_keys=True,indent=2)); return
    print(f"A10 root={s['root']} launchers={len(s['launchers'])} workers={len(s['workers'])} matches={s['committed_matches']} iteration={s['last_complete_iteration']}")
    print(f"checkpoint={s['latest_checkpoint']} disk_free_gib={s['disk']['free_bytes']/2**30:.2f}")
    for r in s['ranks'].values(): print(f"rank{r['rank']}: iter={r['iteration']} complete={r['complete_iteration']} update_s={r['last_update_s']} recompute={r['recompute_within']} errors={len(r['error_lines'])}")
    print(json.dumps(s['gpus']))
if __name__=='__main__': main()
