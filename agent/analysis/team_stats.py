"""Checkpoint-bound natural-match telemetry. Team identities never enter model inputs.

Rank receipts are immutable content-addressed JSON. Only receipts named in a
verified generation manifest are eligible for durable reports. Staging and
replayed attempts are deliberately invisible to those reports.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import uuid
from pathlib import Path

SCHEMA = 'pa3-ppo-team-stats-v1'


def run_identity(root):
    path=Path(root).resolve()
    return path.name+'-'+hashlib.sha256(str(path).encode()).hexdigest()[:16]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    temp=path.with_name('.'+path.name+f'.{uuid.uuid4().hex}.tmp')
    with temp.open('xb') as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def counter():
    return dict(games=0,wins=0,losses=0,draws=0,length_sum=0)


def increment(row,winner,seat,length):
    row['games']+=1;row['length_sum']+=length
    row['draws' if winner is None else 'wins' if winner==seat else 'losses']+=1


class TeamAccumulator:
    def __init__(self,team_count):
        self.team_count=int(team_count);self.matches=0;self.draws=0
        self.teams={};self.matchups={};self.policies={};self.seen=set()

    def record(self,local_match_id,teams,policies,winner,length,*,natural=True):
        if not natural:return
        if local_match_id in self.seen:raise ValueError('duplicate natural match')
        if winner not in (None,0,1) or length<0 or len(teams)!=2 or len(policies)!=2:
            raise ValueError('invalid natural match metadata')
        if any(not 0<=int(t)<self.team_count for t in teams):raise ValueError('unknown team index')
        self.seen.add(local_match_id);self.matches+=1;self.draws+=winner is None
        for seat in (0,1):
            team=str(int(teams[seat]));opponent=str(int(teams[1-seat]))
            row=self.teams.setdefault(team,dict(total=counter(),seat0=counter(),seat1=counter()))
            increment(row['total'],winner,seat,length);increment(row[f'seat{seat}'],winner,seat,length)
            key=json.dumps([int(team),int(opponent),seat],separators=(',',':'))
            increment(self.matchups.setdefault(key,counter()),winner,seat,length)
            key=json.dumps([int(team),seat,str(policies[seat]),str(policies[1-seat])],separators=(',',':'))
            increment(self.policies.setdefault(key,counter()),winner,seat,length)

    def receipt(self,*,run_id,rank,iteration,attempt_id,source_sha,dataset_manifest_sha256,team_file_sha256,start_matches,end_matches):
        value=dict(schema_version=SCHEMA,run_id=run_id,rank=int(rank),iteration_id=int(iteration),
            attempt_id=attempt_id,source_sha=source_sha,dataset_manifest_sha256=dataset_manifest_sha256,
            team_file_sha256=team_file_sha256,team_count=self.team_count,start_matches=int(start_matches),
            end_matches=int(end_matches),natural_matches=self.matches,drawn_matches=self.draws,
            team_counts=self.teams,matchups=self.matchups,policy_splits=self.policies)
        validate(value);value['receipt_sha256']=digest(value)
        return value


def validate(value):
    if value.get('schema_version')!=SCHEMA:raise ValueError('wrong statistics schema')
    if 'receipt_sha256' in value and digest({k:v for k,v in value.items() if k!='receipt_sha256'})!=value['receipt_sha256']:
        raise ValueError('statistics receipt digest mismatch')
    n,d=int(value['natural_matches']),int(value['drawn_matches'])
    if not 0<=d<=n or not 0<=int(value['rank'])<2:raise ValueError('invalid rank/match counts')
    sums=counter()
    for team,row in value['team_counts'].items():
        if not 0<=int(team)<value['team_count']:raise ValueError('invalid team identity')
        for group in ['total','seat0','seat1']:
            c=row[group]
            if any(not isinstance(x,int) or x<0 for x in c.values()) or c['games']!=c['wins']+c['losses']+c['draws']:
                raise ValueError('invalid WLD counter')
        if any(row['total'][k]!=row['seat0'][k]+row['seat1'][k] for k in sums):raise ValueError('seat sum mismatch')
        for k in sums:sums[k]+=row['total'][k]
    if (sums['games'],sums['wins'],sums['losses'],sums['draws'])!=(2*n,n-d,n-d,2*d):
        raise ValueError('two appearances / WLD invariant failed')
    for family in ['matchups','policy_splits']:
        other={k:sum(row[k] for row in value[family].values()) for k in sums}
        if other!=sums:raise ValueError('split counters do not reconcile')


def stage(root,value):
    validate(value);sha=value['receipt_sha256'];path=Path(root).resolve()/'receipts'/f'{sha}.json'
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('immutable receipt collision')
    else:atomic_json(path,value)
    return {'path':str(path),'sha256':sha}


def load(reference):
    value=json.loads(Path(reference['path']).read_text());validate(value)
    if value.get('receipt_sha256')!=reference['sha256']:raise ValueError('wrong receipt reference')
    return value


def reconcile(references,*,committed_matches,run_id=None):
    """Verify complete two-rank attempts and exact contiguous clock; reject duplicates."""
    values=[load(ref) for ref in references];seen=set();iterations={};identity=None
    for value in values:
        key=(value['run_id'],value['iteration_id'],value['rank'])
        if key in seen:raise ValueError('duplicate rank/iteration receipt')
        seen.add(key)
        contract=tuple(value[k] for k in ['run_id','dataset_manifest_sha256','team_file_sha256','team_count'])
        if identity is None:identity=contract
        if contract!=identity or run_id is not None and value['run_id']!=run_id:raise ValueError('statistics contract mismatch')
        iterations.setdefault(value['iteration_id'],{})[value['rank']]=value
    clock=0
    for expected_iteration,(iteration,ranks) in enumerate(sorted(iterations.items()),1):
        if iteration != expected_iteration:raise ValueError('statistics iteration gap')
        if set(ranks)!={0,1}:raise ValueError('missing statistics rank')
        a,b=ranks[0],ranks[1]
        for key in ['start_matches','end_matches','source_sha','attempt_id']:
            if a[key]!=b[key]:raise ValueError('statistics rank boundary mismatch')
        if a['start_matches']!=clock or a['end_matches']-clock!=a['natural_matches']+b['natural_matches']:
            raise ValueError('statistics committed clock gap')
        clock=a['end_matches']
    if clock!=committed_matches:raise ValueError('statistics/checkpoint watermark mismatch')
    return values


def wilson(wins,games):
    if not games:return None,None
    z=1.959963984540054;p=wins/games;den=1+z*z/games
    mid=(p+z*z/(2*games))/den
    half=z*math.sqrt(p*(1-p)/games+z*z/(4*games*games))/den
    return mid-half,mid+half


def summarize(references,teams,committed_matches):
    values=reconcile(references,committed_matches=committed_matches)
    totals={str(i):dict(total=counter(),seat0=counter(),seat1=counter()) for i in range(len(teams))}
    matchups={};windows=[];policies={}
    for value in values:
        if value['team_count']!=len(teams):raise ValueError('team coverage mismatch')
        windows.append(value)
        for team,groups in value['team_counts'].items():
            for group,c in groups.items():
                for k,v in c.items():totals[team][group][k]+=v
        for name,target in [('matchups',matchups),('policy_splits',policies)]:
            for key,c in value[name].items():
                out=target.setdefault(key,counter())
                for k,v in c.items():out[k]+=v
    rows=[];usage={}
    for i,team in enumerate(teams):
        c=totals[str(i)];t=c['total'];n=t['games'];lo,hi=wilson(t['wins'],n)
        row=dict(team_index=i,team_id=team['id'],**t,win_rate=t['wins']/n if n else None,
            score_rate=(t['wins']+.5*t['draws'])/n if n else None,confidence_low=lo,confidence_high=hi,
            mean_length=t['length_sum']/n if n else None)
        for seat in (0,1):row.update({f'seat{seat}_{k}':v for k,v in c[f'seat{seat}'].items()})
        for j,member in enumerate(team['members']):row[f'pokemon_{j+1}']=member['species']
        for species in set(m['species'] for m in team['members']):
            u=usage.setdefault(str(species),counter())
            for k,v in t.items():u[k]+=v
        rows.append(row)
    return dict(team_totals=rows,team_matchups=matchups,team_windows=windows,opponent_policy_splits=policies,
        pokemon_usage=usage,integrity=dict(verified=True,committed_natural_matches=committed_matches,
        team_rows=len(rows),team_appearances=sum(r['games'] for r in rows),
        confidence_method='Wilson 95% binary win probability; draws count as non-wins; score_rate has no CI',
        caveat='Policy-conditioned training outcomes; neither causal team strength nor unseen-roster generalization'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generation',required=True)
    parser.add_argument('--teams',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--min-games',type=int,default=100)
    parser.add_argument('--top',type=int,default=20)
    parser.add_argument('--team',type=int)
    parser.add_argument('view',nargs='?',choices=['summary','leaderboard','matchup','trends','integrity'],default='summary')
    args=parser.parse_args();generation=Path(args.generation)
    from agent.ppo.storage_manager import CheckpointGenerationStore
    if generation not in CheckpointGenerationStore(generation.parent.parent).generations_ready():
        raise ValueError('statistics require a fully verified checkpoint generation')
    manifest=json.loads((generation/'manifest.json').read_text())
    teams_path=Path(args.teams);team_sha=hashlib.sha256(teams_path.read_bytes()).hexdigest()
    refs=manifest['team_stats_receipts'];values=reconcile(refs,committed_matches=manifest['committed_matches'])
    if any(v['team_file_sha256']!=team_sha for v in values):raise ValueError('team file mismatch')
    result=summarize(refs,json.loads(teams_path.read_text()),manifest['committed_matches'])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'team_statistics.json',result);atomic_json(out/'integrity.json',result['integrity'])
    rows=result['team_totals']
    with (out/'team_totals.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name in ['team_matchups','opponent_policy_splits','pokemon_usage']:
        entries=[dict(key=k,**v) for k,v in result[name].items()]
        with (out/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['key',*counter()]);writer.writeheader();writer.writerows(entries)
    window_rows=[]
    for value in result['team_windows']:
        for team,groups in value['team_counts'].items():
            window_rows.append(dict(iteration=value['iteration_id'],rank=value['rank'],
                start_matches=value['start_matches'],end_matches=value['end_matches'],
                team_index=int(team),**groups['total']))
    with (out/'team_windows.csv').open('w') as f:
        fields=['iteration','rank','start_matches','end_matches','team_index',*counter()]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(window_rows)
    qualified=sorted((r for r in rows if r['games']>=max(1,args.min_games)),key=lambda r:r['score_rate'],reverse=True)
    lines=['# Committed training-team statistics','',result['integrity']['caveat'],
        '',f"Natural matches: {manifest['committed_matches']}; appearances: {result['integrity']['team_appearances']}; all {len(rows)} teams present.",
        '',result['integrity']['confidence_method'],
        '',f'Top and bottom up to {args.top} teams with at least {args.min_games} appearances.']
    for label,selected in [('Top',qualified[:args.top]),('Bottom',list(reversed(qualified[-args.top:])) if qualified else [])]:
        lines+=['',label,'','| Team | Games | W/L/D | Score | Win CI |','|---|---:|---|---:|---|']
        for r in selected:lines.append(f"| {r['team_id']} | {r['games']} | {r['wins']}/{r['losses']}/{r['draws']} | {r['score_rate']:.3f} | {r['confidence_low']:.3f}–{r['confidence_high']:.3f} |")
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    if args.view=='leaderboard':print(json.dumps(qualified[:args.top],indent=2))
    elif args.view=='matchup':
        if args.team is None:parser.error('matchup requires --team')
        print(json.dumps({k:v for k,v in result['team_matchups'].items() if json.loads(k)[0]==args.team},indent=2))
    elif args.view=='trends':print(json.dumps(window_rows,indent=2))
    print(json.dumps(result['integrity'],indent=2))


if __name__=='__main__':main()
