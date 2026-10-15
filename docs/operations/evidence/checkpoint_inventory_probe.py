import pathlib,json,hashlib,torch
from agent.ppo.experimental_checkpoint import load_envelope
r=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2-integration');p=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2-throughput/runs/perf/v2_recovery/reconstruction_contract/reconstructed/rolling/generations/iter-000000057344-1791986684864852699')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((p/'manifest.json').read_text());result={'checkpoint':str(p),'manifest':m,'manifest_sha256':sha(p/'manifest.json'),'ranks':{},'historical_failures':{}}
for rank in (0,1):
 path=p/f'rank{rank}.pt';d=load_envelope(str(path));expected=m['ranks'][str(rank)]['sha256'];assert sha(path)==expected
 result['ranks'][str(rank)]={'sha256':expected,'source_sha':d.get('source_sha'),'iteration':d.get('iteration'),'clock':d.get('committed_match_clock'),'experiment':d.get('experiment'),'collector':d.get('collector'),'rank_digests':d.get('rank_digests'),'keys':list(d),'learner_keys':list(d['learner']),'learner_config':d['learner'].get('config'),'rng_keys':list(d.get('rng',{})),'dataset_manifest_hash':d.get('dataset_manifest_hash'),'action_schema_hash':d.get('action_schema_hash'),'opponent_manifest':d.get('opponent_manifest')}
 q=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z/reports')/f'continuation.json.rank{rank}.iteration19.recompute.pt';e=load_envelope(str(q));result['historical_failures'][str(rank)]={'path':str(q),'sha256':sha(q),'max_abs_diff':e['result']['max_abs_diff'],'actor_groups':len(e['actor_groups']),'status':'BLOCKED_MISSING_ACTOR_PROVENANCE'}
(r/'docs/operations/evidence/checkpoint_audit.json').write_text(json.dumps(result,indent=2,default=str)+'\n');print('Checkpoint artifact hashes verified; historical groups absent on both ranks')
