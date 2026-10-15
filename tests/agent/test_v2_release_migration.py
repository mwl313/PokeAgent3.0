import copy,json
from pathlib import Path
import pytest
from agent.ppo.v2_release_migration import validate_contract,BASE,SCHEMA

def contract():
    root=Path(__file__).resolve().parents[2]
    new=json.loads((root/'configs/pa3_v2_production.yaml').read_text());old=copy.deepcopy(new);old['precision'].pop('recompute_secondary');old['precision'].pop('canonical_encoder_batch_rows');old['precision'].pop('ragged_effects_fp32');old['precision'].pop('amp_overflow_retries')
    values=dict(source=BASE,target='f'*40,checkpoint_hash='a',rank_hashes={'0':'b','1':'c'},runtime_blobs={'agent/model/scorer.py':'d'},profile=new,original_profile=old,engine_hash='e',parity={k:True for k in ['representation_equal','loss_equal','gradient_equal','adam_equal','model_config_equal','gpu_loss_equal','gpu_gradient_equal','gpu_adam_equal','gpu_model_equal']})
    receipt=dict(amp_overflow_retry_transition={'historical':0,'future':3},canonical_encoder_transition={'historical':0,'future':1024},ragged_effects_precision_transition={'historical':'fp16','future':'fp32'},schema=SCHEMA,source_sha=BASE,target_sha=values['target'],checkpoint_manifest_sha256='a',rank_artifact_hashes=values['rank_hashes'],runtime_blobs=values['runtime_blobs'],target_profile=new,original_profile=old,engine_sha256='e',token_lookup_backend='gather',token_lookup_transition={'historical':'gather','future':'gather'},secondary_activation={'historical':False,'future':True},semantic_contract='dd723dc687a60b0568302fa0b09cfb110cbe67eac2c7022f2b4ffe4dff373e35')
    return receipt,values

def test_exact_contract_accepts_future_safety_only():
    receipt,values=contract();validate_contract(receipt,**values)

@pytest.mark.parametrize('key',['checkpoint_manifest_sha256','rank_artifact_hashes','runtime_blobs','target_profile','original_profile','engine_sha256','source_sha','target_sha','semantic_contract','token_lookup_backend'])
def test_wrong_artifact_contract_rejected(key):
    receipt,values=contract();receipt[key]='wrong'
    with pytest.raises(ValueError):validate_contract(receipt,**values)

@pytest.mark.parametrize('section,key,value',[('training','ppo_epochs',4),('training','microbatch',2048),('precision','tolerance',.002),('dataset','train_teams',1137),('collector','world_size',1)])
def test_matched_but_unauthorized_configuration_rejected(section,key,value):
    receipt,values=contract();values['profile'][section][key]=value
    with pytest.raises(ValueError):validate_contract(receipt,**values)

@pytest.mark.parametrize('key',['representation_equal','loss_equal','gradient_equal','adam_equal','model_config_equal','gpu_loss_equal','gpu_gradient_equal','gpu_adam_equal','gpu_model_equal'])
def test_missing_or_failed_independent_proof(key):
    receipt,values=contract();values['parity'][key]=False
    with pytest.raises(ValueError):validate_contract(receipt,**values)


def test_even_allowlisted_file_requires_audited_content(tmp_path,monkeypatch):
    import subprocess
    import agent.ppo.v2_release_migration as migration
    def git(*args):return subprocess.check_output(['git',*args],cwd=tmp_path,text=True).strip()
    git('init','-q');git('config','user.name','Migration Test');git('config','user.email','test@example.invalid')
    path=tmp_path/'agent/model/scorer.py';path.parent.mkdir(parents=True);path.write_text('audited = True\n')
    git('add','.');git('commit','-qm','audited');baseline=git('rev-parse','HEAD')
    monkeypatch.setattr(migration,'BASE',baseline);monkeypatch.setattr(migration,'AUDITED',baseline);monkeypatch.setattr(migration,'REVIEWED_SOURCE',baseline)
    path.write_text('audited = False\n');git('add','.');git('commit','-qm','unreviewed change');target=git('rev-parse','HEAD')
    with pytest.raises(ValueError,match='independently audited'):migration.blobs(tmp_path,target)

@pytest.mark.parametrize('field,value',[
 ('ragged_effects_precision_transition',None),
 ('ragged_effects_precision_transition',{'historical':'fp16','future':'fp16'}),
 ('canonical_encoder_transition',{'historical':0,'future':512})])
def test_precision_transition_receipt_required(field,value):
    receipt,values=contract();receipt[field]=value
    with pytest.raises(ValueError):validate_contract(receipt,**values)

def test_precision_flag_is_mandatory():
    receipt,values=contract();values['profile']['precision']['ragged_effects_fp32']=False
    with pytest.raises(ValueError):validate_contract(receipt,**values)

@pytest.mark.parametrize('name',['agent/model/encoder.py','agent/model/precision.py','agent/ppo/learner.py','agent/unexpected.py'])
def test_source_edits_fail_closed(name,tmp_path,monkeypatch):
    import subprocess
    import agent.ppo.v2_release_migration as migration
    def git(*args):return subprocess.check_output(['git',*args],cwd=tmp_path,text=True).strip()
    git('init','-q');git('config','user.name','Test');git('config','user.email','test@example.invalid')
    path=tmp_path/name;path.parent.mkdir(parents=True);path.write_text('approved=True\n')
    git('add','.');git('commit','-qm','baseline');baseline=git('rev-parse','HEAD')
    monkeypatch.setattr(migration,'BASE',baseline);monkeypatch.setattr(migration,'REVIEWED_SOURCE',baseline)
    path.write_text('approved=False\n');git('add','.');git('commit','-qm','tamper')
    with pytest.raises(ValueError):migration.blobs(tmp_path,git('rev-parse','HEAD'))

def test_reverting_to_historical_blob_cannot_escape_review(tmp_path,monkeypatch):
    import subprocess
    import agent.ppo.v2_release_migration as migration
    def git(*args):return subprocess.check_output(['git',*args],cwd=tmp_path,text=True).strip()
    git('init','-q');git('config','user.name','Test');git('config','user.email','test@example.invalid')
    path=tmp_path/'agent/model/encoder.py';path.parent.mkdir(parents=True);path.write_text('canonical=0\n')
    git('add','.');git('commit','-qm','historical');base=git('rev-parse','HEAD')
    path.write_text('canonical=1024\n');git('add','.');git('commit','-qm','reviewed');review=git('rev-parse','HEAD')
    path.write_text('canonical=0\n');git('add','.');git('commit','-qm','unapproved reversion');target=git('rev-parse','HEAD')
    monkeypatch.setattr(migration,'BASE',base);monkeypatch.setattr(migration,'REVIEWED_SOURCE',review)
    with pytest.raises(ValueError,match='independently audited'):migration.blobs(tmp_path,target)
