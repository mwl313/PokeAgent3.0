import copy
import json
import pytest
from agent.analysis.team_stats import TeamAccumulator,stage,reconcile,summarize,load


def pair(root,iteration,start,attempt='first'):
    refs=[]
    for rank in (0,1):
        stats=TeamAccumulator(1220)
        stats.record(0,(0,1),('current','old'),0,12)
        stats.record(1,(1,0),('old','current'),1,14)
        stats.record(2,(0,1),('current','current'),None,16)
        stats.record(3,(0,0),('current','current'),0,18)
        stats.record(4,(0,1),('current','current'),None,0,natural=False)
        refs.append(stage(root,stats.receipt(run_id='run',rank=rank,iteration=iteration,
            attempt_id=attempt,source_sha='abc',dataset_manifest_sha256='manifest',team_file_sha256='teams',
            start_matches=start,end_matches=start+8)))
    return refs


def teams():return [{'id':f'team{i}','members':[{'species':i+j} for j in range(6)]} for i in range(1220)]


def test_two_ranks_natural_draw_mirror_seats_and_zero_coverage(tmp_path):
    refs=pair(tmp_path,1,0);result=summarize(refs,teams(),8)
    assert result['integrity']['team_appearances']==16
    assert len(result['team_totals'])==1220
    a,b,*rest=result['team_totals']
    assert (a['games'],a['wins'],a['losses'],a['draws'])==(10,6,2,2)
    assert (b['games'],b['wins'],b['losses'],b['draws'])==(6,0,4,2)
    assert a['seat0_games']+a['seat1_games']==10
    assert all(row['games']==0 for row in rest)
    assert sum(row['wins'] for row in result['team_totals'])==6
    assert '[0,1,0]' in result['team_matchups'] and '[0,1,1]' in result['team_matchups']


def test_crash_orphan_rewind_and_different_attempt_do_not_count(tmp_path):
    committed=pair(tmp_path,1,0)
    pair(tmp_path,2,8,'orphan')
    assert summarize(committed,teams(),8)['integrity']['team_appearances']==16
    resumed=pair(tmp_path,2,8,'resumed')
    assert summarize(committed+resumed,teams(),16)['integrity']['team_appearances']==32
    assert summarize(committed,teams(),8)['integrity']['team_appearances']==16
    with pytest.raises(ValueError,match='duplicate'):reconcile(committed+committed,committed_matches=8)
    with pytest.raises(ValueError,match='watermark'):reconcile(committed,committed_matches=16)
    with pytest.raises(ValueError,match='missing'):reconcile(committed[:1],committed_matches=8)


def test_digest_identity_attempt_and_counts_fail_closed(tmp_path):
    refs=pair(tmp_path,1,0)
    other=pair(tmp_path,1,0,'other')
    with pytest.raises(ValueError,match='boundary'):reconcile([refs[0],other[1]],committed_matches=8)
    with pytest.raises(ValueError,match='contract'):reconcile(refs,committed_matches=8,run_id='wrong')
    ref=refs[0];path=tmp_path/'receipts'/f"{ref['sha256']}.json"
    content=json.loads(path.read_text());content['natural_matches']+=1;path.write_text(json.dumps(content))
    with pytest.raises(ValueError,match='digest'):load(ref)


def test_duplicate_natural_event_rejected():
    a=TeamAccumulator(2);a.record(1,(0,1),('x','x'),0,4)
    with pytest.raises(ValueError,match='duplicate'):a.record(1,(0,1),('x','x'),0,4)


def test_checkpoint_publication_uses_inspected_byte_snapshot(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import importlib.util,io
    from pathlib import Path
    import torch
    script=Path(__file__).resolve().parents[2]/'scripts/run_ddp_ppo.py'
    spec=importlib.util.spec_from_file_location('v2_test_ddp_launcher',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    prefix=tmp_path/'boundary'
    envelope=dict(accounting=dict(iteration=1,committed_matches=8,optimizer_steps=2),
        collector_drain=dict(complete=True,in_flight_battles=0),rank_digests={'0':'x','1':'x'},
        source_sha='source',dataset_manifest_hash='dataset',opponent_manifest={},experiment={'ppo_epochs':2})
    for rank in (0,1):torch.save(envelope,str(prefix)+f'.rank{rank}')
    original=Path.read_bytes;reads=[]
    def read_once(path):
        payload=original(path)
        if path.name.startswith('boundary.rank'):
            reads.append(path.name)
            changed=copy.deepcopy(envelope);changed['accounting']['iteration']=2
            torch.save(changed,path)
        return payload
    monkeypatch.setattr(Path,'read_bytes',read_once)
    class Store:
        def generations_ready(self):return []
        def publish(self,name,payloads,metadata):
            assert metadata['iteration']==1
            assert all(torch.load(io.BytesIO(data),weights_only=False)['accounting']['iteration']==1 for data in payloads.values())
            return metadata
    args=SimpleNamespace(rolling_checkpoint_root=str(tmp_path),experimental_checkpoint=str(prefix),
        team_stats_root='',dataset_manifest='manifest',precision_mode='D',epochs=2)
    result=module._publish_completed_rolling_boundary(args,Store(),set())
    assert result['iteration']==1 and len(reads)==2
