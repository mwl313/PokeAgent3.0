import json
import pytest
from agent.ppo import v2_source_migration as migration

def fixture(tmp_path,monkeypatch):
    proof=tmp_path/'parity.json'
    files={}
    for name in ['agent/types/action_mapping.py','agent/types/ragged.py','agent/buffer/rollout_buffer.py','agent/train/native_collector.py']:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
        files[name]=migration.sha(path)
    proof.write_text(json.dumps({**{k:True for k in
        ['representation_equal','loss_equal','gradient_equal','adam_equal']},'runtime_file_hashes':files}))
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    for name in ['manifest.json','rank0.pt','rank1.pt']:(checkpoint/name).write_bytes(name.encode())
    target='f'*40
    def git(root,*args):
        if args[:1]==('status',):return ''
        if args==('rev-parse','HEAD'):return target
        raise AssertionError(args)
    monkeypatch.setattr(migration,'git',git)
    monkeypatch.setattr(migration,'runtime_delta',lambda root,target:{'agent/types/ragged.py':'blob'})
    r={'schema':'pa3-v2-data-path-migration-v1','purpose':'bounded-validation-only',
       'source_sha':migration.BASE,'target_sha':target,'runtime_blobs':{'agent/types/ragged.py':'blob'},
       'checkpoint':str(checkpoint),'checkpoint_manifest_sha256':migration.sha(checkpoint/'manifest.json'),
       'rank_artifact_hashes':{str(i):migration.sha(checkpoint/f'rank{i}.pt') for i in (0,1)},
       'parity_report':str(proof),'parity_report_sha256':migration.sha(proof)}
    return r,target

def test_exact_receipt_passes(tmp_path,monkeypatch):
    receipt,target=fixture(tmp_path,monkeypatch)
    assert migration.validate_v2_migration(receipt,tmp_path,migration.BASE,target)['scope']=='v2-bounded-data-path-validation-only'

@pytest.mark.parametrize('field',['source_sha','target_sha','purpose','runtime_blobs','rank_artifact_hashes','parity_report_sha256','checkpoint_manifest_sha256'])
def test_migration_refuses_tampering(tmp_path,monkeypatch,field):
    receipt,target=fixture(tmp_path,monkeypatch)
    receipt[field]={'0':'bad','1':'bad'} if field=='rank_artifact_hashes' else 'bad'
    with pytest.raises(ValueError):migration.validate_v2_migration(receipt,tmp_path,migration.BASE,target)

def test_protected_runtime_rejected(monkeypatch):
    monkeypatch.setattr(migration,'git',lambda *args:'agent/model/pa3_model.py')
    with pytest.raises(ValueError,match='protected'):migration.runtime_delta('/tmp','target')
