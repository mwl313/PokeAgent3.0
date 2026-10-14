"""Recalculate paired telemetry and independent CPU parity evidence."""
import argparse,csv,hashlib,json,pathlib,statistics

def cpu_compare(baseline,candidate,output):
    a=json.loads(pathlib.Path(baseline).read_text());b=json.loads(pathlib.Path(candidate).read_text())
    result={'representation_equal':a['representation']==b['representation'],
            'loss_equal':a['losses']==b['losses'],
            'gradient_equal':a['gradient_digest']==b['gradient_digest'],
            'adam_equal':a['adam_digest']==b['adam_digest'] and a['updated_model_digest']==b['updated_model_digest'],
            'model_config_equal':a['model_config']==b['model_config'],
            'baseline_artifact_sha256':hashlib.sha256(pathlib.Path(baseline).read_bytes()).hexdigest(),
            'candidate_artifact_sha256':hashlib.sha256(pathlib.Path(candidate).read_bytes()).hexdigest(),
            'runtime_file_hashes':b.get('runtime_file_hashes'), 'stages':{}}
    for key in a['times'][0]:
        av=[x[key] for x in a['times']];bv=[x[key] for x in b['times']]
        result['stages'][key]={'baseline_median_s':statistics.median(av),'candidate_median_s':statistics.median(bv),
            'baseline_range_s':[min(av),max(av)],'candidate_range_s':[min(bv),max(bv)],
            'local_speedup':statistics.median(av)/statistics.median(bv)}
    pathlib.Path(output).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

def telemetry(paths,output):
    ranks={}
    for path in paths:
        for line in pathlib.Path(path).read_text().splitlines():
            value=json.loads(line);identity=value['identity'];key=(identity['absolute_iteration'],identity['rank'])
            if key in ranks:raise ValueError('duplicate paired telemetry')
            ranks[key]=value
    result=[];previous=0
    for i in sorted({i for i,r in ranks}):
        if not all((i,r) in ranks for r in (0,1)):continue
        pair=[ranks[i,r] for r in (0,1)];critical=max(pair,key=lambda r:r['performance']['iteration_wall_s'])
        wall=critical['performance']['iteration_wall_s'];clock=critical['identity']['committed_matches']
        if any(r['identity']['committed_matches']!=clock for r in pair):raise ValueError('rank clocks differ')
        for field in ['source_sha','model_digest','optimizer_digest']:
            if len({r['safety']['rank_verification'][field] for r in pair})!=1:raise ValueError('rank digests differ')
        rows=sum(r['ppo']['rows'] for r in pair);games=critical['performance']['committed_matches_this_iteration']
        total_steps=critical['ppo']['optimizer_steps'];steps=total_steps-previous;previous=total_steps
        t=critical['performance']['timings']
        result.append(dict(iteration=i,clock=clock,games=games,rows=rows,wall_s=wall,games_s=games/wall,
          rows_s=rows/wall,rows_game=rows/games,optimizer_steps=steps,steps_s=steps/wall,
          collection_s=t['collect_wall_s'],clock_wait_s=t['lr_clock_wall_s'],prepare_s=t['prepare_wall_s'],update_s=t['update_wall_s'],
          collection_imbalance_s=abs(pair[0]['performance']['timings']['collect_wall_s']-pair[1]['performance']['timings']['collect_wall_s']),
          epochs=[r['ppo']['epochs_run'] for r in pair],recompute=max(r['safety']['recompute']['max_abs_diff'] for r in pair),
          errors=[r['collector']['operational_errors'] for r in pair],skips=[r['ppo']['optimizer_steps_skipped'] for r in pair],
          candidate_records_s=None,gpu_peak=None,cpu_rss=None,
          model_digest=critical['safety']['rank_verification']['model_digest'],optimizer_digest=critical['safety']['rank_verification']['optimizer_digest']))
    out={'iterations':result,'weighted_games_s':sum(x['games'] for x in result)/sum(x['wall_s'] for x in result),
         'note':'candidate records/GPU peak/RSS unavailable in per-iteration production JSONL; first step delta assumes initial scratch stream when included'}
    pathlib.Path(output).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='iterations'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline');p.add_argument('--candidate');p.add_argument('--telemetry',nargs='*');p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.telemetry:telemetry(a.telemetry,a.output)
    else:cpu_compare(a.baseline,a.candidate,a.output)
