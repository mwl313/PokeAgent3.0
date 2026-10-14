import importlib.util
from pathlib import Path
import pytest


def module():
    spec=importlib.util.spec_from_file_location('v2_sidecar_test',Path(__file__).resolve().parents[2]/'scripts/v2_evaluation_sidecar.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def test_retention_preserves_first_best_recent_receipts_and_aliases(tmp_path):
    own=tmp_path/'evaluation/v2';policies=own/'policies';policies.mkdir(parents=True)
    entries={}
    for i in range(8):
        path=policies/f'{i}.pt';path.write_text(f'checkpoint {i}')
        entries[str(i)]={'id':str(i),'committed_matches':i,'path':str(path)}
    entries['2']['path']=entries['3']['path']
    receipts=own/'receipts.json';receipts.write_text('immutable outcomes')
    registry={'policies':entries}
    module().retain(own,registry,'3')
    assert all(Path(entries[i]['path']).exists() for i in ['0','3','5','6','7'])
    assert all(entries[i]['retired'] for i in ['1','2','4'])
    assert Path(entries['2']['path']).exists()  # alias of retained best
    assert not (policies/'1.pt').exists() and not (policies/'4.pt').exists()
    assert receipts.read_text()=='immutable outcomes'


def test_retention_refuses_foreign_files(tmp_path):
    own=tmp_path/'evaluation/v2';own.mkdir(parents=True)
    entries={str(i):{'id':str(i),'committed_matches':i,'path':str(tmp_path/f'foreign{i}')} for i in range(6)}
    for entry in entries.values():Path(entry['path']).write_text('preserve')
    with pytest.raises(ValueError,match='foreign'):module().retain(own,{'policies':entries},'0')
    assert all(Path(entry['path']).read_text()=='preserve' for entry in entries.values())
