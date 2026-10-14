"""Read-only snapshot of completed paired PPO telemetry; no learner imports."""
import argparse,csv,datetime,hashlib,json,statistics,subprocess
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gpu-samples');a=p.parse_args();run=Path(a.run_root).resolve();out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    now=datetime.datetime.now(datetime.timezone.utc);source_files=[];rows={}
    for path in sorted((run/'reports').glob('*.ppo.jsonl')):
        raw=path.read_bytes();complete=[line for line in raw.splitlines(keepends=True) if line.endswith(b'\n')]
        saved=b''.join(complete);(out/path.name).write_bytes(saved)
        source_files.append(dict(path=str(path),read_bytes=len(raw),complete_bytes=len(saved),sha256=hashlib.sha256(saved).hexdigest()))
        for line in complete:
            value=json.loads(line);key=(value['identity']['absolute_iteration'],value['identity']['rank'])
            if key in rows:raise ValueError('duplicate iteration/rank telemetry')
            rows[key]=value
    first=json.loads((run/'reports/first.json').read_text());(out/'first.json').write_text(json.dumps(first,indent=2)+'\n')
    pairs=[]
    for iteration in sorted({i for i,r in rows}):
        if not all((iteration,r) in rows for r in (0,1)):continue
        ranks=[rows[iteration,r] for r in (0,1)]
        clocks={r['identity']['committed_matches'] for r in ranks}
        games={r['performance']['committed_matches_this_iteration'] for r in ranks}
        if len(clocks)!=1 or len(games)!=1:raise ValueError('rank clock mismatch')
        for field in ['model_digest','optimizer_digest','source_sha']:
            if len({r['safety']['rank_verification'][field] for r in ranks})!=1:raise ValueError('rank digest mismatch')
        critical=max(ranks,key=lambda r:r['performance']['iteration_wall_s'])
        wall=critical['performance']['iteration_wall_s'];timing=critical['performance']['timings']
        budget={'collection_and_clock_wait':timing['collect_wall_s']+timing['lr_clock_wall_s'],
                'ppo_update':timing['update_wall_s'],'prepare':timing['prepare_wall_s'],
                'other_and_unaccounted':wall-timing['collect_wall_s']-timing['lr_clock_wall_s']-timing['update_wall_s']-timing['prepare_wall_s']}
        if budget['other_and_unaccounted']<0:raise ValueError('invalid exclusive budget')
        pairs.append(dict(iteration=iteration,clock=next(iter(clocks)),games=next(iter(games)),wall_s=wall,
            games_per_s=next(iter(games))/wall,critical_rank=critical['identity']['rank'],exclusive_budget_s=budget,
            max_rank_stage_s={key:max(r['performance']['timings'][key] for r in ranks) for key in timing},
            learner_rows=sum(r['ppo']['rows'] for r in ranks),epochs=[r['ppo']['epochs_run'] for r in ranks],
            optimizer_steps=ranks[0]['ppo']['optimizer_steps'],skipped_steps=[r['ppo']['optimizer_steps_skipped'] for r in ranks],
            errors=[r['collector']['operational_errors'] for r in ranks],
            recompute_max=max(r['safety']['recompute']['max_abs_diff'] for r in ranks),
            source_sha=ranks[0]['safety']['rank_verification']['source_sha']))
    if not pairs:raise ValueError('no paired completed iterations')
    for previous,current in zip(pairs,pairs[1:]):
        if current['clock']-previous['clock']!=current['games']:raise ValueError('snapshot has a clock gap')
    def aggregate(selected):
        games=sum(x['games'] for x in selected);wall=sum(x['wall_s'] for x in selected)
        return dict(iterations=len(selected),games=games,wall_s=wall,weighted_games_per_s=games/wall,
            min_games_per_s=min(x['games_per_s'] for x in selected),max_games_per_s=max(x['games_per_s'] for x in selected),
            mean_wall_s=wall/len(selected),
            mean_exclusive_budget_s={k:sum(x['exclusive_budget_s'][k] for x in selected)/len(selected) for k in selected[0]['exclusive_budget_s']})
    result=dict(captured_utc=now.isoformat(),captured_kst=now.astimezone(ZoneInfo('Asia/Seoul')).isoformat(),run_root=str(run),
        source_files=source_files,paired_iterations=pairs,aggregate=aggregate(pairs),
        continuation_aggregate=aggregate(pairs[1:]) if len(pairs)>1 else None,
        first_launch_all_in_games_per_s=first['all_in_committed_games_per_s'],
        first_profiles=[{key:r['iterations'][0][key] for key in ['rank','collector_stats','profile','prepare_profile','gpu_peak_reserved_gib']} for r in first['ranks']],
        limitations=['Snapshot of complete paired lines only; evolving policy and short sample, not a fixed-policy benchmark',
          'CPU perf_counter substage labels are inclusive host wall and may include CUDA synchronization; not pure kernel timings',
          'Stage maxima across ranks are not additive; exclusive budget uses one critical rank per iteration',
          'No profiler attachment, new PPO experiment, restart, or runtime setting change'])
    manifest=json.loads((run/'release/manifest.json').read_text());(out/'launch_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    latest=(run/'rolling/LATEST').read_text().strip();generation=json.loads((run/'rolling/generations'/latest/'manifest.json').read_text())
    result['durable_matches']=generation['committed_matches'];result['latest_durable_generation']=latest
    result['last_verified_in_memory_matches']=pairs[-1]['clock']
    units={}
    for unit in [manifest['systemd_unit'],manifest['evaluation_timer'],manifest['evaluation_service']]:
        r=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','MainPID','-p','Result'],capture_output=True,text=True)
        units[unit]={'returncode':r.returncode,'status':r.stdout.strip()}
    result['units']=units
    if a.gpu_samples:
        samples=json.loads(Path(a.gpu_samples).read_text());(out/'gpu_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
        gpu={}
        for sample in samples:
            for row in sample['gpus']:
                i,u,m,t,w,limit=[float(x.strip()) for x in next(csv.reader([row]))]
                gpu.setdefault(str(int(i)),[]).append((u,m,t,w,limit))
        result['gpu_sampling']={i:dict(samples=len(v),mean_util_percent=statistics.mean(x[0] for x in v),max_util_percent=max(x[0] for x in v),
            max_temperature_c=max(x[2] for x in v),max_power_w=max(x[3] for x in v),power_limit_w=v[0][4]) for i,v in gpu.items()}
        result['gpu_sampling_note']='15 samples about 2s apart; phase-limited window, not full-iteration utilization or a thermal/power-throttling diagnosis'
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['captured_kst','aggregate','continuation_aggregate','durable_matches','last_verified_in_memory_matches']},indent=2))


if __name__=='__main__':main()
