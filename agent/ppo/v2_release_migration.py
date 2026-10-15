"""Reviewed V2 full-state migration; exact local artifacts, no source bypass."""
import hashlib,json,subprocess
from pathlib import Path
BASE='5ccc51b0cd4a49efb93018258b9750d5e804ae32'
AUDITED='0b7253cfd915ff61c88248b1979bd8e254285ef0'
ERROR_WRAPPER_AST='a24af9614ac5542e7c0b377ed88ad6266439d02b03e6a6cd323599ceca887a22'
LEGACY_SCHEMA='pa3-v2-reviewed-release-migration-v1'
SCHEMA='pa3-v2-reviewed-release-migration-v2'
REVIEWED_SOURCE='93ebc1b5086b7a2bf247b779eb6442671bbbdd10'
REVIEWED_REPAIRS={'agent/ppo/source_migration.py': '540698ee4f62d7b48125721db264d7082eea02f5', 'scripts/run_ddp_ppo.py': 'f6d7eb197deaefd8edb066bb04be2a8cc84ba988', 'agent/ppo/learner.py': 'e06009a3f2fba15f0e85352bf88848d3717d53d7', 'agent/runtime/config.py': '8c7deb3ba55af5819c42cb5057f369c1f227e22a', 'configs/pa3_v2_production.yaml': '464149253b631548d604be34369972c08004644a', 'agent/ppo/telemetry.py': 'f879957a63b9f2a07e487a738fecf1b572c4db1a', 'agent/train/native_collector.py': '9626a27b564602b0293f3aaf0c0ad2bca459e7df', 'scripts/bench_pa3_end_to_end.py': 'bb51079faeffb79b36742861256aa3f84bf443da'}
VALIDATOR_AST='4d7c0853ebe92b627a311d8f6375e44550cdda054f913d2e08fa0684f50576cd'
MANIFEST='b9539f3a0c426f0878375b61630cb4de9dcd2a5d4a6a59858b4d64f4287fd424'
ENGINE='8f3643a9f7b2febfa567f039b911d7af2f88bbf7bfc912849ada9e6a690d1c62'
# Exact reviewed executable surface; objective, architecture, engine and data
# semantics stay fixed. Learner changes are only bounded collective AMP retry.
REVIEWED=set('''agent/buffer/rollout_buffer.py agent/model/cuda/exact_lookup.cu
agent/model/encoder.py agent/model/embeddings.py agent/model/exact_row_lookup.py agent/model/ragged_embeddings.py
agent/model/scorer.py agent/ppo/recompute_safety.py agent/ppo/source_migration.py
agent/ppo/v2_source_migration.py agent/ppo/v2_release_migration.py agent/runtime/config.py
agent/train/native_collector.py agent/types/action_mapping.py agent/types/observation.py
agent/types/ragged.py agent/types/requests.py configs/pa3_v2_production.yaml
scripts/bench_pa3_end_to_end.py scripts/analyze_v2_live_bottlenecks.py scripts/benchmark_v2_lookup_recovery.py
scripts/run_ddp_ppo.py scripts/v2_continuation_probe.py scripts/v2_data_path_probe.py
scripts/v2_gpu_profile.py scripts/v2_reconstruct_boundary.py scripts/v2_recovery_analysis.py
scripts/v2_throughput_ab.py scripts/validate_v2_source_of_truth.py scripts/v2_release_validation.py
agent/ppo/telemetry.py agent/ppo/learner.py agent/model/pa3_model.py agent/model/precision.py scripts/v2_training_status.py
scripts/v2_evaluation_sidecar.py scripts/v2_recompute_forensics.py'''.split())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
def blobs(root,target):
    files=git(root,'diff','--name-only',BASE,target,'--','agent','engine','configs','data','scripts').splitlines()
    if not set(files)<=REVIEWED:raise ValueError('unreviewed executable source delta: '+str(set(files)-REVIEWED))
    result={name:git(root,'rev-parse',f'{target}:{name}') for name in files}
    # Also inspect reversions to BASE: those disappear from the BASE diff.
    audit_files=set(files)|set(git(root,'diff','--name-only',REVIEWED_SOURCE,target,'--','agent','engine','configs','data','scripts').splitlines())
    if not audit_files<=REVIEWED:raise ValueError('unreviewed executable source delta: '+str(audit_files-REVIEWED))
    for name in audit_files:
        blob=git(root,'rev-parse',f'{target}:{name}')
        if name == 'agent/ppo/v2_release_migration.py':
            # Seal the validator AST with only its digest literal blanked.
            # This avoids a commit hashing itself while rejecting other edits.
            import ast
            tree=ast.parse(git(root,'show',f'{target}:{name}'))
            for node in tree.body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VALIDATOR_AST' for t in node.targets):
                    node.value=ast.Constant('')
            if hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()!=VALIDATOR_AST:
                raise ValueError('unreviewed migration validator')
        elif blob != (REVIEWED_REPAIRS[name] if name in REVIEWED_REPAIRS else git(root,'rev-parse',f'{REVIEWED_SOURCE}:{name}')):
            raise ValueError('runtime differs from independently audited code: '+name)
    return result
def validate_contract(receipt,*,source,target,checkpoint_hash,rank_hashes,runtime_blobs,profile,original_profile,engine_hash,parity):
    if receipt.get('schema')!=SCHEMA or (source,target)!=(BASE,receipt.get('target_sha')) or receipt.get('source_sha')!=source:raise ValueError('migration source identity mismatch')
    expected={'checkpoint_manifest_sha256':checkpoint_hash,'rank_artifact_hashes':rank_hashes,
              'runtime_blobs':runtime_blobs,'target_profile':profile,'original_profile':original_profile,'engine_sha256':engine_hash}
    for key,value in expected.items():
        if receipt.get(key)!=value:raise ValueError('migration '+key+' mismatch')
    if profile['training']!=original_profile['training']:raise ValueError('migration PPO hyperparameters changed')
    for section in ('collector','evaluation'):
        left=dict(profile[section]);right=dict(original_profile[section])
        if section=='collector':
            left.pop('token_lookup_backend',None);right.pop('token_lookup_backend',None)
        if left!=right:raise ValueError('migration execution contract changed: '+section)
    before=dict(original_profile['precision']);after=dict(profile['precision']);after.pop('recompute_secondary',None);before.pop('recompute_secondary',None)
    after.pop('canonical_encoder_batch_rows',None);before.pop('canonical_encoder_batch_rows',None)
    # This is a deliberately narrow numerical migration: Mode D's ragged
    # effects projection is forced to FP32 after the preserved V100 failure
    # replay proved its FP16 GEMM was batch-composition dependent.  The
    # receipt must name the transition; no checkpoint may silently acquire it.
    after.pop('ragged_effects_fp32',None);before.pop('ragged_effects_fp32',None)
    after.pop('amp_overflow_retries',None);before.pop('amp_overflow_retries',None)
    if (original_profile['precision'].get('amp_overflow_retries',0)!=0
            or profile['precision'].get('amp_overflow_retries')!=3
            or receipt.get('amp_overflow_retry_transition')!={'historical':0,'future':3}):
        raise ValueError('AMP retry precision migration mismatch')
    if original_profile['precision'].get('canonical_encoder_batch_rows',0)!=0 or profile['precision'].get('canonical_encoder_batch_rows')!=1024 or receipt.get('canonical_encoder_transition')!={'historical':0,'future':1024}:raise ValueError('canonical encoder contract mismatch')
    if (original_profile['precision'].get('ragged_effects_fp32',False)
            or profile['precision'].get('ragged_effects_fp32') is not True
            or receipt.get('ragged_effects_precision_transition') != {'historical':'fp16','future':'fp32'}):
        raise ValueError('ragged effects precision migration mismatch')
    if before!=after or profile['precision'].get('recompute_secondary') is not True:raise ValueError('migration precision incompatible')
    for key in ('manifest_sha256','teams_sha256','train_teams','sampling','id','frozen'):
        if profile['dataset'][key]!=original_profile['dataset'][key]:raise ValueError('migration dataset changed')
    backend=profile['collector'].get('token_lookup_backend','gather')
    if backend not in ('gather','exact-row-v100-v1') or original_profile['collector'].get('token_lookup_backend','gather')!='gather':
        raise ValueError('unauthorized arithmetic backend')
    if receipt.get('token_lookup_backend')!=backend or receipt.get('token_lookup_transition')!={'historical':'gather','future':backend} or receipt.get('secondary_activation')!={'historical':False,'future':True}:
        raise ValueError('unauthorized arithmetic backend or safety transition')
    if not all(parity.get(k) is True for k in ('representation_equal','loss_equal','gradient_equal','adam_equal','model_config_equal','gpu_loss_equal','gpu_gradient_equal','gpu_adam_equal','gpu_model_equal')):raise ValueError('migration independent parity proof incomplete')
    if receipt.get('semantic_contract')!='dd723dc687a60b0568302fa0b09cfb110cbe67eac2c7022f2b4ffe4dff373e35':raise ValueError('migration semantic contract mismatch')

def validate(receipt,root,source,target):
    root=Path(root)
    if git(root,'rev-parse','HEAD')!=target or git(root,'status','--porcelain','--untracked-files=no') or git(root,'ls-files','--others','--exclude-standard','--','agent','engine','configs','data','scripts'):raise ValueError('migration requires exact clean source')
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
        if hashlib.sha256(subprocess.check_output(['git','-C',str(root),'show',f'{AUDITED}:{name}'])).hexdigest()!=digest:raise ValueError('historical data-path parity code changed: '+name)
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
        if hashlib.sha256(subprocess.check_output(['git','-C',str(root),'show',f'{AUDITED}:{name}'])).hexdigest()!=digest:raise ValueError('historical GPU parity source changed: '+name)
    legacy=parity.get('legacy_target_runtime_blobs',{})
    legacy_files=git(root,'diff','--name-only',BASE,AUDITED,'--','agent','engine','configs','data','scripts').splitlines()
    if legacy!={n:git(root,'rev-parse',f'{AUDITED}:{n}') for n in legacy_files}:
        raise ValueError('historical independently audited executable identity mismatch')
    if parity.get('target_runtime_blobs')!=blobs(root,target):raise ValueError('parity target executable identity mismatch')
    transition=Path(parity['precision_transition_report'])
    if sha(transition)!=parity['precision_transition_sha256']:raise ValueError('precision proof checksum mismatch')
    proof=json.loads(transition.read_text())
    if proof.get('schema')!='pa3-effects-fp32-independent-v1' or proof.get('target_sha')!=target:
        raise ValueError('precision proof target mismatch')
    if proof.get('runtime_blobs')!=blobs(root,target):raise ValueError('precision proof executable mismatch')
    for key in ('reference','candidate'):
        path=Path(proof[key+'_artifact'])
        if sha(path)!=proof[key+'_sha256']:raise ValueError('precision proof artifact changed')
    reference=json.loads(Path(proof['reference_artifact']).read_text())
    candidate=json.loads(Path(proof['candidate_artifact']).read_text())
    if reference.get('source_sha')!=AUDITED or candidate.get('source_sha')!=target:
        raise ValueError('precision proof source mismatch')
    for artifact in (reference,candidate):
        if artifact.get('rows')!=1024 or artifact.get('precision')!='D/FP16-effects-FP32' or artifact.get('workspace')!=':4096:8':
            raise ValueError('precision proof execution mismatch')
    if reference.get('reference') is not True or candidate.get('reference') is not False:
        raise ValueError('precision proof independent execution missing')
    if reference.get('captured_artifact_sha256')!=candidate.get('captured_artifact_sha256'):
        raise ValueError('precision proof input mismatch')
    if reference.get('probe_sha256')!=candidate.get('probe_sha256') or candidate.get('probe_sha256')!=sha(root/'tests/validation/v2_precision_probe.py'):
        raise ValueError('precision proof runner mismatch')
    expected_backend=json.loads((root/'configs/pa3_v2_production.yaml').read_text())['collector'].get('token_lookup_backend','gather')
    if reference.get('token_lookup_backend')!='gather' or candidate.get('token_lookup_backend')!=expected_backend:
        raise ValueError('precision proof token backend mismatch')
    if candidate.get('amp_overflow_retries')!=3 or reference.get('amp_overflow_retries')!=0:
        raise ValueError('precision proof AMP retry execution mismatch')
    for key in ('losses','gradient_digest','adam_digest','updated_model_digest','model_config'):
        if reference[key]!=candidate[key]:raise ValueError('independent precision transition parity failed: '+key)

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
            'token_lookup_backend':receipt['token_lookup_backend'],'token_lookup_transition':receipt['token_lookup_transition'],
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
        'amp_overflow_retry_transition':{'historical':0,'future':3},
        'canonical_encoder_transition':{'historical':0,'future':1024},
        'ragged_effects_precision_transition':{'historical':'fp16','future':'fp32'},
        'token_lookup_backend':json.loads((root/'configs/pa3_v2_production.yaml').read_text())['collector'].get('token_lookup_backend','gather'),
        'token_lookup_transition':{'historical':'gather','future':json.loads((root/'configs/pa3_v2_production.yaml').read_text())['collector'].get('token_lookup_backend','gather')},
        'secondary_activation':{'historical':False,'future':True},
        'team_stats_origin_root':str(origin/'team_stats'),'team_stats_lineage_id':run_identity(origin/'team_stats'),
        'team_stats_root':str(Path(team_stats_root).resolve()),'parity_report':str(proof),'parity_report_sha256':sha(proof)}
    validate(receipt,root,BASE,target);Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(receipt,indent=2)+'\n');return receipt
