import collections,json,pathlib,statistics
ROOT=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0')
OUT=ROOT/'runs/perf/cooling_20261011'
ORDER=['iteration_setup_wall_s','collect_wall_s','recompute_wall_s','lr_clock_wall_s',
       'prepare_wall_s','update_wall_s','digest_wall_s','checkpoint_wall_s','cleanup_wall_s',
       'iteration_barrier_wall_s','unaccounted_wall_s']

def stats(values):
    v=list(values)
    return {'n':len(v),'mean':statistics.mean(v),'median':statistics.median(v),
            'min':min(v),'max':max(v),'stdev':statistics.stdev(v) if len(v)>1 else 0}

def analyze():
    alignment_path=OUT/'phase_alignment.json'
    alignment=json.loads(alignment_path.read_text()) if alignment_path.exists() else {}
    runs={name:json.loads((OUT/f'{name}.json').read_text()) for name in
          ['cold1','cold2','cold3','persistent1','persistent2','persistent3']}
    steady=[]; critical=[]; telemetry=collections.defaultdict(list)
    for name,r in runs.items():
        samples=[json.loads(line) for line in (OUT/f'{name}_telemetry.jsonl').read_text().splitlines()]
        for rank in r['ranks']:
            gpu=rank['rank']
            key=f'{name}:rank{gpu}'
            end=alignment[key] if key in alignment else (OUT/f'{name}.json.rank{gpu}.json').stat().st_mtime
            alignment[key]=end
            start=end-rank['worker_wall_s']+rank['startup_wall_s']
            windows=[]
            for it in rank['iterations']:
                cursor=start
                for key in ORDER[:-1]:
                    duration=it[key]
                    windows.append((cursor,cursor+duration,key,it['iteration']))
                    cursor+=duration
                start+=it['iteration_wall_s']
            for row in samples:
                when=row['utc_s']
                phase=next(((key,index) for a,b,key,index in windows if a<=when<=b),('startup_shutdown',0))
                v=row['gpu'][gpu]
                record={k:float(v[k]) for k in ['temperature.gpu','power.draw','power.limit','clocks.sm',
                    'clocks.mem','utilization.gpu','utilization.memory','memory.used']}
                record.update({k:v[k]=='Active' for k in ['clocks_event_reasons.sw_power_cap',
                    'clocks_event_reasons.hw_thermal_slowdown','clocks_event_reasons.sw_thermal_slowdown',
                    'clocks_event_reasons.hw_power_brake_slowdown']})
                record.update(case=name,phase=phase[0],iteration=phase[1],utc_s=when)
                telemetry[gpu].append(record)
        if name.startswith('persistent'):
            for index in (1,2):
                pair=[rank['iterations'][index] for rank in r['ranks']]
                steady.append(pair)
                critical.append(max(pair,key=lambda row:row['iteration_wall_s']))
    s={'runs':{name:{'all_in_games_s':r['all_in_committed_games_per_s'],
                    'launcher_wall_s':r['launcher_wall_s'],'steady':r['steady_state']} for name,r in runs.items()},
       'cold_games_s':stats(runs[f'cold{i}']['all_in_committed_games_per_s'] for i in (1,2,3)),
       'steady_run_games_s':stats(runs[f'persistent{i}']['steady_state']['committed_games_per_s'] for i in (1,2,3)),
       'steady_games_s_weighted':sum(sum(row['games'] for row in pair) for pair in steady)/sum(max(row['iteration_wall_s'] for row in pair) for pair in steady),
       'steady_iteration_wall':stats(row['iteration_wall_s'] for row in critical),
       'steady_total_games':sum(sum(row['games'] for row in pair) for pair in steady),
       'steady_total_rows':sum(sum(row['rows'] for row in pair) for pair in steady),
       'critical_rank_stages':{key:stats(row[key] for row in critical) for key in ORDER},
       'ranks':{},'gpu':{},'correctness':{},'host':{}}
    for gpu in (0,1):
        rows=[pair[gpu] for pair in steady]
        s['ranks'][gpu]={'stages':{key:stats(row[key] for row in rows) for key in ORDER},
            'collector':{key:stats(row['collector_stats'][key] for row in rows)
                         for key in rows[0]['collector_stats'] if key.endswith('_seconds') or key in ['rounds','games','decisions','candidate_calls','candidates_returned','cpu_fraction_of_one_core']},
            'prepare':{key:stats(row['prepare_profile'][key] for row in rows) for key in rows[0]['prepare_profile']},
            'learner_host_timers':{key:stats(row['profile'][key] for row in rows) for key in rows[0]['profile']},
            'rows':stats(row['rows'] for row in rows),
            'optimizer_steps':stats(row['optimizer_steps_this_iteration'] for row in rows),
            'peak_reserved_gib':max(row['gpu_peak_reserved_gib'] for row in rows),
            'peak_rss_gib':max(row['rss_peak_gib'] for row in rows)}
        s['gpu'][gpu]={}
        for phase in ['all','collect_wall_s','prepare_wall_s','update_wall_s']:
            rr=telemetry[gpu] if phase=='all' else [x for x in telemetry[gpu] if x['phase']==phase and x['case'].startswith('persistent') and x['iteration']>1]
            s['gpu'][gpu][phase]={key:stats(row[key] for row in rr) for key in rr[0] if isinstance(rr[0][key],(int,float)) and key not in ['utc_s','iteration']}
    all_iterations=[row for r in runs.values() for rank in r['ranks'] for row in rank['iterations']]
    s['correctness']={'unprofiled_iterations':sum(len(r['iterations']) for r in runs.values()),
        'total_games':sum(r['total_games'] for r in runs.values()),
        'operational_errors':sum(row['operational_errors'] for row in all_iterations),
        'skipped_steps':sum(row['report']['optimizer_steps_skipped'] for row in all_iterations),
        'all_epochs_four':all(row['report']['epochs_run']==4 for row in all_iterations),
        'all_digests_equal':all(row['digests_equal'] for row in all_iterations),
        'all_recompute_pass':all(row['recompute']['within_gate'] for row in all_iterations),
        'recompute_max_diff':max(row['recompute']['max_abs_diff'] for row in all_iterations),
        'all_clean':all(not r['provenance']['git_dirty'] for r in runs.values()),
        'sha':sorted({r['provenance']['git_sha'] for r in runs.values()})}
    host_samples=[json.loads(line) for name in runs for line in (OUT/f'{name}_telemetry.jsonl').read_text().splitlines()]
    s['host']={'min_available_gib':min(row['mem_available_kib'] for row in host_samples)/(1<<20),
        'swap_in_pages_delta':max(row['vm']['pswpin'] for row in host_samples)-min(row['vm']['pswpin'] for row in host_samples),
        'swap_out_pages_delta':max(row['vm']['pswpout'] for row in host_samples)-min(row['vm']['pswpout'] for row in host_samples),
        'system_major_faults_delta':max(row['vm']['pgmajfault'] for row in host_samples)-min(row['vm']['pgmajfault'] for row in host_samples)}
    (OUT/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    (OUT/'aligned_gpu_telemetry.json').write_text(json.dumps(telemetry)+'\n')
    alignment_path.write_text(json.dumps(alignment,indent=2)+'\n')
    print(json.dumps({k:s[k] for k in ['cold_games_s','steady_games_s_weighted','steady_run_games_s','correctness','host']},indent=2))
    return s

if __name__=='__main__': analyze()
