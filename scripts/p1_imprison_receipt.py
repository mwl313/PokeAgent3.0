"""Capture P1 evidence and mechanically update local execution ledgers."""
import csv
import hashlib
import json
import pathlib
import subprocess

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/audit_v2/p1_imprison'
BASE='ec5912736440d5f5456f678c4d76f8fd26fd20aa'
SOURCE='86ffebc09f9c91e765314c79f04d2b8004fc7ac9'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def strip_request_details(body):
    for fixture in body['fixtures']:
        for expected in [fixture.get('initial')]+[s.get('expected') for s in fixture.get('steps',[])]:
            if not expected:
                continue
            for side in expected.get('sides',[]):
                side.pop('request_detail',None)
    return body


def main():
    regen=json.loads((OUT/'regeneration.json').read_text())
    checked=[]
    for entry in regen['files']:
        path='engine/data/'+entry['file']
        before=subprocess.check_output(['git','-C',str(ROOT),'show',f'{BASE}:{path}'])
        after=(ROOT/path).read_bytes()
        assert digest(before)==entry['before_sha256'],path
        assert digest(after)==entry['after_sha256'],path
        old,new=json.loads(before),json.loads(after)
        assert strip_request_details(old)==strip_request_details(new),f'unrelated drift: {path}'
        checked.append(path)
    # Ignore only the transport wall-clock line, not battle turns or RNG.
    def clock_free(value):
        if isinstance(value,dict):return {k:clock_free(v) for k,v in value.items()}
        if isinstance(value,list):return [clock_free(v) for v in value]
        if isinstance(value,str) and value.startswith('|t:|'):return '|t:|<transport-clock>'
        return value
    old=json.loads((OUT/'reference_before.json').read_text())
    fresh=json.loads((OUT/'reference_fresh_build.json').read_text())
    for key in ['oracle_commit','format','fixture','seed','teams','boundaries','hidden_choice']:
        assert clock_free(old[key])==clock_free(fresh[key]),key
    teams=json.loads((ROOT/'engine/data/a10-training-teams.json').read_text())
    tables=json.loads((ROOT/'engine/data/dex.json').read_text())['tables']
    moves={r['id']:r['numeric_id'] for r in tables['moves']}
    holders=[{'team_index':i,'canonical_id':t['id'],'roster':j}
             for i,t in enumerate(teams) for j,m in enumerate(t['members']) if moves['imprison'] in m['moves']]
    reference=ROOT/'vendor/pokemon-showdown-p1'
    assert subprocess.check_output(['git','-C',str(reference),'rev-parse','HEAD'],text=True).strip()==old['oracle_commit']
    assert not subprocess.check_output(['git','-C',str(reference),'status','--porcelain'],text=True).strip()
    receipt={
        'source_sha':SOURCE,'baseline':BASE,'P1_status':'BLOCKED',
        'oracle_discrepancy':'FIXED','remaining_blocker':'native hidden Imprison choice re-request transaction unsupported; fail-closed quarantine, not a complete engine fix',
        'rust_tests':39,'node_adapter_tests':5,'full_oracle_fixtures':1152,'full_oracle_boundaries':25231,
        'all_changed_data_outside_request_details_identical':True,'verified_data_files':checked,
        'fresh_reference_matches_before_probe_except_transport_clock':True,
        'reference_commit':old['oracle_commit'],'reference_clean':True,
        'fresh_built_battle_js_sha256':digest((reference/'dist/sim/battle.js').read_bytes()),
        'fresh_built_pokemon_js_sha256':digest((reference/'dist/sim/pokemon.js').read_bytes()),
        'actual_pool_imprison_holders':holders,
        'team_file_sha256':digest((ROOT/'engine/data/a10-training-teams.json').read_bytes()),
        'changed_source_and_fixture_files':git('diff-tree','--no-commit-id','--name-only','-r',SOURCE).splitlines(),
        'old_evaluation_stop_authorized_by_user':True,
        'old_evaluation_units':subprocess.check_output(['systemctl','--user','show','a10-evaluation-worker.service','a10-evaluation.timer','a10-evaluation.service','-p','Id','-p','ActiveState','-p','MainPID','-p','UnitFileState'],text=True),
        'stop_commands':['systemctl --user disable --now a10-evaluation.timer a10-evaluation-worker.service','systemctl --user stop a10-evaluation.service'],
        'old_evaluation_note':'Worker and discovery timer inactive/disabled. Discovery oneshot retains failed state, MainPID=0; failure not erased. No old results/checkpoints deleted.',
        'production_started':False,
        'artifact_sha256':{p.name:digest(p.read_bytes()) for p in OUT.iterdir() if p.is_file() and p.name!='RECEIPT.json'},
    }
    (OUT/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    path=ROOT/'docs/audit_v2/02_DEFECT_REGISTER.csv'
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        if row['issue_id'] in ['KNOW-03','ENG-01']:
            row.update(local_status='BLOCKED',source_sha=SOURCE,evidence='p1_imprison/RECEIPT.json',
                remaining_gate='Public oracle corpus passes; Imprison re-request unsupported and reachable in 48 frozen teams. No P1 PASS.')
    extra={key:'' for key in fields}
    extra.update(issue_id='ENG-IMPRISON-RETRY',priority='P0',verdict='REPRODUCED',title='Advertised hidden move may be rejected at selection; native retry transaction absent',
        primary_sources='sim/side.ts;engine/src/battle.rs',planned_phase='P1',required_acceptance='Reject/re-request without RNG/turn advance; handle all-hidden Struggle substitution and retry through native binding',
        test_id='T-ENG-IMPRISON-RETRY',local_status='BLOCKED',source_sha=SOURCE,
        evidence='p1_imprison/reference_fresh_build.json;p1_imprison/native_before.json;p1_imprison/native_retry_after.log',
        remaining_gate='Quarantined Unsupported; reference retry transaction and consumer handling remain unimplemented.')
    rows=[r for r in rows if r['issue_id']!=extra['issue_id']]+[extra]
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps({'data_files_verified':len(checked),'imprison_teams':len(holders),'P1_status':'BLOCKED'}))


if __name__=='__main__':main()
