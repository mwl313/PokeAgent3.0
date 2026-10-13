"""A10 CPU league contracts. Owns only evaluation artifacts, never PPO state."""
from __future__ import annotations
import hashlib, json, math, os, tempfile, time
from pathlib import Path
from agent.evaluation.identity import canonical_model_digest, sha256_file

SOURCE = '0b5d930743298720e073f65264464f0d953ca7d3'
DATASET = '1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb'
ACTION = 'a838aad854edc4ac019fbb8450ca19f25d37d344e8d9ec6e4d63191493280559'
MODEL = 'f3c4fedd17c1bcfabb2152f59d78ab574e8e92af1e4635cb483ea01089f50f45'
VERSION = 'a10-cpu-league-v2'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def atomic(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.league-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,sort_keys=True); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def read(path, default=None):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError: return default

def validate_payload(p, manifest, rank):
    if p.get('kind')!='ppo_experimental_iteration_boundary': raise ValueError('not a recovery envelope')
    for k,v in [('source_sha',SOURCE),('dataset_manifest_hash',DATASET),('action_schema_hash',ACTION),('model_config_hash',MODEL)]:
        if p.get(k)!=v: raise ValueError(f'checkpoint {k} mismatch')
    a=p['accounting']; d=p['ddp']; e=p['experiment']
    if (a['iteration'],a['committed_matches'])!=(manifest['iteration'],manifest['committed_matches']): raise ValueError('rank clock mismatch')
    if d.get('rank')!=rank or d.get('world_size')!=2 or d.get('communicator')!='recreate': raise ValueError('DDP identity mismatch')
    drain=p.get('collector_drain',{})
    if drain.get('complete') is not True or drain.get('in_flight_battles')!=0: raise ValueError('undrained checkpoint')
    if (e.get('precision_mode'),e.get('precision_stability'),e.get('precision_contract_version'))!=('D','fp16-reduction-off','pa3-execution-contract-v1'): raise ValueError('precision contract mismatch')
    rd=p.get('rank_digests',{})
    if set(rd)!={'0','1'} or any(not x.get('model') or not x.get('optimizer') for x in rd.values()) or rd['0']!=rd['1']: raise ValueError('rank digest mismatch')
    return p

def capture_generation(rolling, name, archive):
    """Pin identity and open both files before rotation; open descriptors survive unlink."""
    import torch
    rolling,archive=Path(rolling),Path(archive)
    if Path(name).name!=name: raise ValueError('invalid generation name')
    generation=rolling/'generations'/name; m=read(generation/'manifest.json')
    if not m or m.get('complete') is not True or m.get('iteration_complete') is not True: raise ValueError('incomplete generation')
    training_source=m.get('source_sha')
    migration=None
    if training_source!=SOURCE:
        # Trusted-local exact migration registration, consumed by the short
        # discovery process. The already-running battle worker keeps its
        # original inference contract and requires no reload or interruption.
        if not isinstance(training_source,str) or len(training_source)!=40 or any(c not in '0123456789abcdef' for c in training_source): raise ValueError('invalid training source')
        migration=read(rolling/'source_migrations'/f'{training_source}.json')
        if not migration or migration.get('source_sha')!=SOURCE or migration.get('target_sha')!=training_source or migration.get('scope')!='telemetry-and-recompute-verification-only': raise ValueError('unregistered training source migration')
    if m.get('world_size')!=2 or m.get('dataset_manifest_hash')!=DATASET: raise ValueError('manifest contract mismatch')
    files=[]
    try:
        for rank in ('0','1'):
            row=m['ranks'][rank]
            if row['path']!=f'rank{rank}.pt': raise ValueError('unexpected rank path')
            files.append((generation/row['path']).open('rb'))
        payloads=[]
        for rank,f in enumerate(files):
            row=m['ranks'][str(rank)]; h=hashlib.sha256()
            for b in iter(lambda:f.read(1<<20),b''): h.update(b)
            if f.tell()!=row['bytes'] or h.hexdigest()!=row['sha256']: raise ValueError('rank checksum mismatch')
            f.seek(0); payload=torch.load(f,map_location='cpu',weights_only=False)
            if migration:
                if payload.get('source_sha')!=training_source or payload.get('experiment',{}).get('source_migration')!=migration: raise ValueError('checkpoint migration receipt mismatch')
                # Validate unchanged inference contract without relabeling the
                # stored recovery envelope or its manifest/hashes.
                validate_payload({**payload,'source_sha':SOURCE},m,rank)
            else: validate_payload(payload,m,rank)
            payloads.append(payload)
        if payloads[0]['rank_digests']!=payloads[1]['rank_digests']: raise ValueError('rank peer mismatch')
        for rank,payload in enumerate(payloads):
            if training_digests(payload)!=payload['rank_digests'][str(rank)]: raise ValueError('actual model/Adam digest mismatch')
        model=payloads[0]['learner']['model']; md=canonical_model_digest(model)
        if canonical_model_digest(payloads[1]['learner']['model'])!=md: raise ValueError('rank actual model mismatch')
        meta={'kind':'a10_inference_snapshot','source_sha':SOURCE,'model_config_hash':MODEL,'action_schema_hash':ACTION,'dataset_manifest_hash':DATASET,'experiment':payloads[0]['experiment'],'accounting':{'committed_matches':m['committed_matches']},'checkpoint_generation':name,'source_checkpoint_sha256':m['ranks']['0']['sha256'],'learner':{'model':model}}
        if migration:
            meta.update(training_source_sha=training_source,training_source_migration=migration,
                        source_sha_semantics='frozen evaluation execution contract; training_source_sha is actual trainer revision')
        return store_snapshot(archive,meta,legacy=False)
    finally:
        for f in files: f.close()

def store_snapshot(archive, payload, legacy=False):
    import torch
    archive=Path(archive); archive.mkdir(parents=True,exist_ok=True)
    md=canonical_model_digest(payload['learner']['model']); clock=payload['accounting']['committed_matches']
    identity=digest({'model':md,'clock':clock,'source':SOURCE,'dataset':DATASET,'precision':payload['experiment']})
    target=archive/f'{identity}.pt'
    if not target.exists():
        fd,tmp=tempfile.mkstemp(prefix='snapshot-',suffix='.pt',dir=archive); os.close(fd)
        try: torch.save(payload,tmp); os.chmod(tmp,0o444); os.replace(tmp,target)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    existing=torch.load(target,map_location='cpu',weights_only=False)
    if canonical_model_digest(existing['learner']['model'])!=md: raise ValueError('existing inference archive corrupted')
    entry={'id':f'a10-{clock}-{identity[:16]}','path':str(target.resolve()),'file_sha256':sha256_file(target),'model_digest':md,'committed_matches':clock,'source_sha':SOURCE,'dataset_manifest_hash':DATASET,'model_config_hash':MODEL,'action_schema_hash':ACTION,'observation_schema':'pa3-entity-observation-v1','action_schema':'atomic-action-v1','precision_mode':'D','precision_stability':'fp16-reduction-off','precision_contract_version':'pa3-execution-contract-v1','generation':payload.get('checkpoint_generation'),'source_checkpoint_sha256':payload.get('source_checkpoint_sha256'),'created_at':time.time(),'legacy_snapshot_revalidated':legacy,'integrity_level':'legacy-metadata-and-current-model-hash' if legacy else 'both-rank-manifest-SHA-and-actual-model-Adam-digests'}
    atomic(archive/f'{identity}.json',entry); return entry

def import_legacy(p, archive):
    import torch
    meta=read(p)
    if not meta or meta.get('source_sha')!=SOURCE or meta.get('dataset_manifest_hash')!=DATASET: raise ValueError('legacy metadata mismatch')
    path=Path(meta['path']); payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('kind')!='a10_inference_snapshot' or payload.get('source_sha')!=SOURCE or payload.get('accounting',{}).get('committed_matches')!=meta['committed_matches'] or payload.get('source_checkpoint_sha256')!=meta.get('source_checkpoint_sha256'): raise ValueError('legacy snapshot identity mismatch')
    e=payload.get('experiment',{})
    if (e.get('precision_mode'),e.get('precision_stability'),e.get('precision_contract_version'))!=('D','fp16-reduction-off','pa3-execution-contract-v1'): raise ValueError('legacy precision mismatch')
    from agent.model import build_model,PA3Config
    model=build_model(PA3Config(),device='cpu'); model.load_state_dict(payload['learner']['model'],strict=True)
    payload.update(dataset_manifest_hash=DATASET,model_config_hash=MODEL,action_schema_hash=ACTION)
    return store_snapshot(archive,payload,legacy=True)

def verify_entry(entry):
    import torch
    if sha256_file(entry['path'])!=entry['file_sha256']: raise ValueError('snapshot file checksum mismatch')
    p=torch.load(entry['path'],map_location='cpu',weights_only=False)
    for k,v in [('source_sha',SOURCE),('dataset_manifest_hash',DATASET),('model_config_hash',MODEL),('action_schema_hash',ACTION)]:
        if p.get(k)!=v or entry.get(k)!=v: raise ValueError(f'snapshot {k} mismatch')
    if canonical_model_digest(p['learner']['model'])!=entry['model_digest']: raise ValueError('model digest mismatch')
    return p

def promotion(outcomes, comparison_index, stage, opponent_index=1):
    """Pair-cluster Hoeffding bound; alpha spending across challengers AND stages."""
    from agent.evaluation.fixed_policy import FixedPolicyEvaluator
    scores=FixedPolicyEvaluator._paired_scores(outcomes)
    if not scores or any(x['outcome']=='error' for x in outcomes): return {'decision':'invalid'}
    alpha=.05/(comparison_index*(comparison_index+1)*opponent_index*(opponent_index+1)*3)
    mean=sum(scores)/len(scores); radius=math.sqrt(math.log(1/alpha)/(2*len(scores)))
    return {'decision':'promote' if mean-radius>.5 else 'retain' if mean+radius<.5 or stage==2 else 'escalate','score':mean,'lower':max(0,mean-radius),'upper':min(1,mean+radius),'alpha':alpha,'pairs':len(scores),'method':'pair-cluster-Hoeffding-global-alpha-spending-v1'}

def training_digests(payload):
    """Recompute original launcher model/Adam digest from serialized state."""
    import torch
    from agent.model import build_model, PA3Config
    state=payload['learner']['model'];h=hashlib.sha256()
    for name,v in state.items():
        v=v.detach().cpu().contiguous();h.update(name.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
    model=build_model(PA3Config(),device='cpu');names=[n for n,p in model.named_parameters()];opt=payload['learner']['optimizer'];ids=[i for g in opt['param_groups'] for i in g['params']];mapping=dict(zip(ids,names));oh=hashlib.sha256()
    for pid,entry in opt['state'].items():
        oh.update(mapping.get(pid,'unknown').encode())
        for k in ('step','exp_avg','exp_avg_sq'):
            if entry.get(k) is None:continue
            oh.update(k.encode());oh.update(torch.as_tensor(entry[k]).detach().cpu().contiguous().to(torch.float64).numpy().tobytes())
    return {'model':h.hexdigest(),'optimizer':oh.hexdigest()}

def retain_archive(own, max_bytes=2*2**30):
    """Explicit ownership-safe retention; never follows files outside league_v2/policies."""
    own=Path(own);reg=read(own/'registry.json');q=read(own/'jobs.json',{'jobs':{}});best=read(own/'best_policy.json',{})
    if not reg:return
    entries=sorted([e for e in reg['policies'].values() if not e.get('retired')],key=lambda e:e['committed_matches']);keep={e['id'] for e in entries[:2]+entries[-8:]};keep.add(best.get('policy_id'))
    for h in best.get('promotions',[])[-4:]:keep.update([h['previous_policy'],h['new_policy']])
    for j in q['jobs'].values():
        if j['status'] in ('pending','running','retry'):keep.update([j['candidate'],j.get('opponent')])
    total=sum(Path(e['path']).stat().st_size for e in entries if Path(e['path']).exists())
    for e in entries:
        if total<=max_bytes:break
        if e['id'] in keep:continue
        path=Path(e['path'])
        if path.parent.resolve()!=(own/'policies').resolve() or path.suffix!='.pt':raise ValueError('refusing cleanup outside owned archive')
        if path.exists():size=path.stat().st_size;path.unlink();total-=size
        e['retired']=True;e['retirement_reason']='bounded evaluation archive; receipts retained'
    atomic(own/'registry.json',reg)
