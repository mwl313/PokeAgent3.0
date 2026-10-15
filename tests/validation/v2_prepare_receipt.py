"""Prepare final-SHA migration with preserved legacy and fresh precision proofs."""
import argparse,json,hashlib
from pathlib import Path
from agent.ppo.v2_release_migration import BASE,AUDITED,blobs,git,sha,prepare
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=Path.cwd();out=Path(a.output).resolve();target=git(root,'rev-parse','HEAD')
reference=out/'precision_reference.json';candidate=out/'precision_candidate.json'
l=json.loads(reference.read_text());r=json.loads(candidate.read_text())
assert l['source_sha']==AUDITED and r['source_sha']==target
for key in ('captured_artifact_sha256','probe_sha256','losses','gradient_digest','adam_digest','updated_model_digest','model_config'):assert l[key]==r[key],key
proof={'schema':'pa3-effects-fp32-independent-v1','target_sha':target,'runtime_blobs':blobs(root,target),'reference_artifact':str(reference),'reference_sha256':sha(reference),'candidate_artifact':str(candidate),'candidate_sha256':sha(candidate)}
(out/'precision_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
legacy=Path('/home/minidc/Desktop/PokeAgent3.0-V2-integration/runs/v2_final_release/parity.json');parity=json.loads(legacy.read_text())
for key in ('baseline','candidate','gpu_baseline','gpu_candidate'):
 path=Path(parity[key+'_artifact']);expected=parity[key+'_sha256']
 if not path.exists() or sha(path)!=expected:
  choices=list((legacy.parent/'proof_archive').rglob('*'))
  path=next(p for p in choices if p.is_file() and sha(p)==expected)
 parity[key+'_artifact']=str(path)
parity.update(legacy_target_runtime_blobs=parity['target_runtime_blobs'],target_runtime_blobs=blobs(root,target),precision_transition_report=str(out/'precision_proof.json'),precision_transition_sha256=sha(out/'precision_proof.json'))
(out/'parity.json').write_text(json.dumps(parity,indent=2)+'\n')
checkpoint='/home/minidc/Desktop/PokeAgent3.0-V2-throughput/runs/perf/v2_recovery/reconstruction_contract/reconstructed/rolling/generations/iter-000000057344-1791986684864852699'
receipt=prepare(root,checkpoint,out/'parity.json',out/'smoke/team_stats',out/'migration.json')
print(json.dumps({'target':target,'receipt_sha256':sha(out/'migration.json'),'checkpoint_manifest':receipt['checkpoint_manifest_sha256'],'rank_hashes':receipt['rank_artifact_hashes']},indent=2))
