"""Reviewed V2 full-state migration; exact local artifacts, no source bypass."""
import hashlib,json,subprocess
from pathlib import Path
BASE='5ccc51b0cd4a49efb93018258b9750d5e804ae32'
SCHEMA='pa3-v2-reviewed-release-migration-v1'
MANIFEST='b9539f3a0c426f0878375b61630cb4de9dcd2a5d4a6a59858b4d64f4287fd424'
ENGINE='8f3643a9f7b2febfa567f039b911d7af2f88bbf7bfc912849ada9e6a690d1c62'
# Explicitly reviewed executable surface. Engine/dataset/loss/learner/config
# architecture/semantic contracts are deliberately absent.
REVIEWED=set('''agent/buffer/rollout_buffer.py agent/model/cuda/exact_lookup.cu
agent/model/encoder.py agent/model/embeddings.py agent/model/exact_row_lookup.py agent/model/ragged_embeddings.py
agent/model/scorer.py agent/ppo/recompute_safety.py agent/ppo/source_migration.py
agent/ppo/v2_source_migration.py agent/ppo/v2_release_migration.py agent/runtime/config.py
agent/train/native_collector.py agent/types/action_mapping.py agent/types/observation.py
agent/types/ragged.py agent/types/requests.py configs/pa3_v2_production.yaml
scripts/bench_pa3_end_to_end.py scripts/analyze_v2_live_bottlenecks.py scripts/benchmark_v2_lookup_recovery.py
scripts/run_ddp_ppo.py scripts/v2_continuation_probe.py scripts/v2_data_path_probe.py
scripts/v2_gpu_profile.py scripts/v2_reconstruct_boundary.py scripts/v2_recovery_analysis.py
scripts/v2_throughput_ab.py scripts/validate_v2_source_of_truth.py scripts/v2_release_validation.py'''.split())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
def blobs(root,target):
    files=git(root,'diff','--name-only',BASE,target,'--','agent','engine','configs','data','scripts').splitlines()
    if not set(files)<=REVIEWED:raise ValueError('unreviewed executable source delta: '+str(set(files)-REVIEWED))
    return {name:git(root,'rev-parse',f'{target}:{name}') for name in files}
def validate_contract(receipt,*,source,target,checkpoint_hash,rank_hashes,runtime_blobs,profile,original_profile,engine_hash,parity):
    if receipt.get('schema')!=SCHEMA or (source,target)!=(BASE,receipt.get('target_sha')) or receipt.get('source_sha')!=source:raise ValueError('migration source identity mismatch')
    expected={'checkpoint_manifest_sha256':checkpoint_hash,'rank_artifact_hashes':rank_hashes,
              'runtime_blobs':runtime_blobs,'target_profile':profile,'original_profile':original_profile,'engine_sha256':engine_hash}
    for key,value in expected.items():
        if receipt.get(key)!=value:raise ValueError('migration '+key+' mismatch')
    if profile['training']!=original_profile['training']:raise ValueError('migration PPO hyperparameters changed')
    for section in ('collector','evaluation'):
        if profile[section]!=original_profile[section]:raise ValueError('migration execution contract changed: '+section)
    before=dict(original_profile['precision']);after=dict(profile['precision']);after.pop('recompute_secondary',None);before.pop('recompute_secondary',None)
    after.pop('canonical_encoder_batch_rows',None);before.pop('canonical_encoder_batch_rows',None)
    if original_profile['precision'].get('canonical_encoder_batch_rows',0)!=0 or profile['precision'].get('canonical_encoder_batch_rows')!=1024 or receipt.get('canonical_encoder_transition')!={'historical':0,'future':1024}:raise ValueError('canonical encoder contract mismatch')
    if before!=after or profile['precision'].get('recompute_secondary') is not True:raise ValueError('migration precision incompatible')
    for key in ('manifest_sha256','teams_sha256','train_teams','sampling','id'):
        if profile['dataset'][key]!=original_profile['dataset'][key]:raise ValueError('migration dataset changed')
    if receipt.get('token_lookup_backend')!='gather' or receipt.get('secondary_activation')!={'historical':False,'future':True}:raise ValueError('unauthorized arithmetic backend or safety transition')
    if not all(parity.get(k) is True for k in ('representation_equal','loss_equal','gradient_equal','adam_equal','model_config_equal','gpu_loss_equal','gpu_gradient_equal','gpu_adam_equal','gpu_model_equal')):raise ValueError('migration independent parity proof incomplete')
    if receipt.get('semantic_contract')!='dd723dc687a60b0568302fa0b09cfb110cbe67eac2c7022f2b4ffe4dff373e35':raise ValueError('migration semantic contract mismatch')

def validate(receipt,root,source,target):
    root=Path(root)
    if git(root,'rev-parse','HEAD')!=target or git(root,'status','--porcelain','--untracked-files=no'):raise ValueError('migration requires exact clean source')
    checkpoint=Path(receipt['checkpoint']);manifest=json.loads((checkpoint/'manifest.json').read_text())
    if sha(checkpoint/'manifest.json')!=MANIFEST:raise ValueError('wrong original checkpoint manifest')
    if sha(root/'engine/python/pa3_engine/pa3_engine.so')!=ENGINE:raise ValueError('native binary identity mismatch')
    ranks={str(r):sha(checkpoint/f'rank{r}.pt') for r in (0,1)}
    if manifest.get('source_sha')!=BASE or manifest.get('committed_matches')!=57344 or manifest.get('iteration')!=14 or not manifest.get('complete'):raise ValueError('unsupported original checkpoint boundary')
    if any(ranks[k]!=manifest['ranks'][k]['sha256'] for k in ranks):raise ValueError('checkpoint rank artifact corrupted')
    parity_path=Path(receipt['parity_report'])
    if sha(parity_path)!=receipt['parity_report_sha256']:raise ValueError('parity proof checksum mismatch')
    parity=json.loads(parity_path.read_text())
    if parity.get('baseline_source_sha')!=BASE:raise ValueError('parity baseline source mismatch')
    inputs=[]
    for key in ('baseline','candidate'):
        path=Path(parity[key+'_artifact'])
        if sha(path)!=parity[key+'_sha256']:raise ValueError('independent parity artifact changed')
        inputs.append(json.loads(path.read_text()))
    left,right=inputs
    for key in ('representation','losses','gradient_digest','adam_digest','updated_model_digest','model_config'):
        if left[key]!=right[key]:raise ValueError('independent math/data parity failed: '+key)
    for name,digest in right['runtime_file_hashes'].items():
        if sha(root/name)!=digest:raise ValueError('data-path parity code changed: '+name)
    gpu=[]
    for key in ('gpu_baseline','gpu_candidate'):
        path=Path(parity[key+'_artifact'])
        if sha(path)!=parity[key+'_sha256']:raise ValueError('GPU parity artifact changed')
        gpu.append(json.loads(path.read_text()))
    for artifact in gpu:
        if artifact.get('experimental_row_lookup') or artifact.get('experimental_exact_row_lookup') or artifact.get('rows')!=1024 or artifact.get('precision')!='D/FP16':raise ValueError('unauthorized GPU proof execution')
    if gpu[0]['source_sha']!=BASE:raise ValueError('GPU baseline source mismatch')
    for key in ('losses','gradient_digest','adam_digest','updated_model_digest'):
        if gpu[0][key]!=gpu[1][key]:raise ValueError('GPU same-microbatch parity failed: '+key)
    for name,digest in gpu[1]['runtime_file_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('GPU parity source changed: '+name)
    if parity.get('target_runtime_blobs')!=blobs(root,target):raise ValueError('parity target executable identity mismatch')
    old_path=Path(receipt['original_profile_path'])
    if sha(old_path)!=receipt['original_profile_sha256']:raise ValueError('original profile changed')
    profile=json.loads((root/'configs/pa3_v2_production.yaml').read_text());original=json.loads(old_path.read_text())
    validate_contract(receipt,source=source,target=target,checkpoint_hash=sha(checkpoint/'manifest.json'),rank_hashes=ranks,runtime_blobs=blobs(root,target),profile=profile,original_profile=original,engine_hash=sha(root/'engine/python/pa3_engine/pa3_engine.so'),parity=parity)
    from agent.runtime.config import resolve_profile
    # Relative versioned dataset paths resolve from the exact runtime root.
    resolved=resolve_profile(str(root/'configs/pa3_v2_production.yaml'))
    if resolved['config_hash']!=receipt['target_config_hash']:raise ValueError('target config drift')
    from agent.ppo.experimental_checkpoint import load_envelope,validate_envelope
    from agent.analysis.team_stats import reconcile,run_identity
    states=[]
    for rank in (0,1):
        state=load_envelope(str(checkpoint/f'rank{rank}.pt'));states.append(state)
        validate_envelope(state,dataset_manifest_hash=profile['dataset']['manifest_sha256'],action_schema_hash=receipt['action_schema_hash'],source_sha=BASE,expected_rank=rank,expected_world_size=2,expected_executor='ddp')
        if state['experiment'].get('recompute_secondary') or state['experiment'].get('precision_mode')!='D' or state['experiment'].get('precision_stability')!='fp16-reduction-off':raise ValueError('original checkpoint precision mismatch')
        if state['model_config_hash']!=receipt['model_config_hash'] or state['ppo_config_hash']!=receipt['ppo_config_hash']:raise ValueError('checkpoint model/PPO identity mismatch')
        if state['rank_digests']!=receipt['verified_rank_digests'] or state['accounting']['committed_matches']!=57344:raise ValueError('checkpoint learner identity mismatch')
    refs=[ref for s in states for ref in s['team_stats_receipts']]
    reconcile(refs,committed_matches=57344,run_id=receipt['team_stats_lineage_id'])
    if run_identity(receipt['team_stats_origin_root'])!=receipt['team_stats_lineage_id']:raise ValueError('statistics origin identity mismatch')
    return {'source_sha':source,'target_sha':target,'scope':SCHEMA,'secondary_activation':receipt['secondary_activation'],
            'team_stats_lineage_id':receipt['team_stats_lineage_id'],'team_stats_root':receipt['team_stats_root'],
            'receipt_sha256':hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()}

def prepare(root,checkpoint,parity_report,team_stats_root,output):
    root=Path(root).resolve();checkpoint=Path(checkpoint).resolve();proof=Path(parity_report).resolve()
    from agent.ppo.experimental_checkpoint import load_envelope
    from agent.runtime.config import resolve_profile
    from agent.analysis.team_stats import run_identity
    state=load_envelope(str(checkpoint/'rank0.pt'));target=git(root,'rev-parse','HEAD')
    origin=Path('/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z');old=origin/'release/runtime_profile.json'
    receipt={'schema':SCHEMA,'source_sha':BASE,'target_sha':target,'checkpoint':str(checkpoint),
        'checkpoint_manifest_sha256':sha(checkpoint/'manifest.json'),'rank_artifact_hashes':{str(r):sha(checkpoint/f'rank{r}.pt') for r in (0,1)},
        'runtime_blobs':blobs(root,target),'original_profile_path':str(old),'original_profile_sha256':sha(old),
        'original_profile':json.loads(old.read_text()),'target_profile':json.loads((root/'configs/pa3_v2_production.yaml').read_text()),
        'target_config_hash':resolve_profile(str(root/'configs/pa3_v2_production.yaml'))['config_hash'],
        'semantic_contract':'dd723dc687a60b0568302fa0b09cfb110cbe67eac2c7022f2b4ffe4dff373e35',
        'action_schema_hash':state['action_schema_hash'],'model_config_hash':state['model_config_hash'],'ppo_config_hash':state['ppo_config_hash'],
        'verified_rank_digests':state['rank_digests'],'engine_sha256':sha(root/'engine/python/pa3_engine/pa3_engine.so'),
        'canonical_encoder_transition':{'historical':0,'future':1024},
        'token_lookup_backend':'gather','secondary_activation':{'historical':False,'future':True},
        'team_stats_origin_root':str(origin/'team_stats'),'team_stats_lineage_id':run_identity(origin/'team_stats'),
        'team_stats_root':str(Path(team_stats_root).resolve()),'parity_report':str(proof),'parity_report_sha256':sha(proof)}
    validate(receipt,root,BASE,target);Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(receipt,indent=2)+'\n');return receipt
