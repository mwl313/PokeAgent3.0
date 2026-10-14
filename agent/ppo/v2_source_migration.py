"""Exact V2 data-path migration for bounded experiments; no production approval."""
import hashlib,json,subprocess
from pathlib import Path

BASE='5ccc51b0cd4a49efb93018258b9750d5e804ae32'
ALLOWED={
 'agent/types/action_mapping.py','agent/types/ragged.py',
 'agent/buffer/rollout_buffer.py','agent/train/native_collector.py',
 'agent/ppo/v2_source_migration.py','agent/ppo/source_migration.py',
 'scripts/v2_throughput_ab.py','scripts/v2_data_path_probe.py',
 'scripts/v2_recovery_analysis.py','scripts/v2_gpu_profile.py',
 'scripts/analyze_v2_live_bottlenecks.py','scripts/run_ddp_ppo.py',
 'scripts/v2_continuation_probe.py',
 'scripts/v2_reconstruct_boundary.py',
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()

def runtime_delta(root,target):
    changed=git(root,'diff','--name-only',BASE,target,'--','agent','engine','configs','data','scripts').splitlines()
    if not set(changed)<=ALLOWED:raise ValueError('V2 migration changed protected runtime')
    return {p:git(root,'rev-parse',f'{target}:{p}') for p in changed}

def validate_v2_migration(receipt,root,source,target):
    if receipt.get('schema')!='pa3-v2-data-path-migration-v1' or source!=BASE:
        raise ValueError('unsupported V2 migration source')
    if (receipt.get('source_sha'),receipt.get('target_sha'))!=(source,target):
        raise ValueError('V2 migration source/target mismatch')
    if receipt.get('purpose')!='bounded-validation-only':raise ValueError('V2 migration has no production authorization')
    if receipt.get('execution_environment')!={'CUBLAS_WORKSPACE_CONFIG':':4096:8'}:
        raise ValueError('V2 migration must preserve original CUDA workspace contract')
    if git(root,'rev-parse','HEAD')!=target or git(root,'status','--porcelain','--untracked-files=no'):
        raise ValueError('V2 migration requires exact clean committed runtime')
    if runtime_delta(root,target)!=receipt.get('runtime_blobs'):raise ValueError('V2 runtime blob mismatch')
    proof=Path(receipt['parity_report'])
    if sha(proof)!=receipt['parity_report_sha256']:raise ValueError('V2 parity receipt checksum mismatch')
    parity=json.loads(proof.read_text())
    if not all(parity.get(k) is True for k in ('representation_equal','loss_equal','gradient_equal','adam_equal')):
        raise ValueError('V2 independent parity evidence incomplete')
    compared=['agent/types/action_mapping.py','agent/types/ragged.py','agent/buffer/rollout_buffer.py','agent/train/native_collector.py']
    if parity.get('runtime_file_hashes')!={name:sha(Path(root)/name) for name in compared}:
        raise ValueError('V2 parity evidence belongs to different data-path code')
    checkpoint=Path(receipt['checkpoint'])
    if sha(checkpoint/'manifest.json')!=receipt['checkpoint_manifest_sha256']:
        raise ValueError('V2 checkpoint manifest mismatch')
    for rank in (0,1):
        if sha(checkpoint/f'rank{rank}.pt')!=receipt['rank_artifact_hashes'][str(rank)]:
            raise ValueError('V2 checkpoint rank checksum mismatch')
    return {'source_sha':source,'target_sha':target,'scope':'v2-bounded-data-path-validation-only'}

def prepare(root,checkpoint,proof,output):
    root=Path(root).resolve();checkpoint=Path(checkpoint).resolve();proof=Path(proof).resolve()
    target=git(root,'rev-parse','HEAD')
    manifest=json.loads((checkpoint/'manifest.json').read_text())
    for rank in (0,1):
        if sha(checkpoint/f'rank{rank}.pt')!=manifest['ranks'][str(rank)]['sha256']:
            raise ValueError('input checkpoint rank hash mismatch')
    receipt={'schema':'pa3-v2-data-path-migration-v1','purpose':'bounded-validation-only',
       'execution_environment':{'CUBLAS_WORKSPACE_CONFIG':':4096:8'},
       'source_sha':BASE,'target_sha':target,'runtime_blobs':runtime_delta(root,target),
       'checkpoint':str(checkpoint),'checkpoint_manifest_sha256':sha(checkpoint/'manifest.json'),
       'rank_artifact_hashes':{str(r):sha(checkpoint/f'rank{r}.pt') for r in (0,1)},
       'parity_report':str(proof),'parity_report_sha256':sha(proof)}
    validate_v2_migration(receipt,root,BASE,target)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--parity-report',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(Path(__file__).resolve().parents[2],a.checkpoint,a.parity_report,a.output),indent=2))
