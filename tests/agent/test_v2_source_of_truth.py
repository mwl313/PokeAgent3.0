import importlib.util,json,shutil
from pathlib import Path
import pytest

def test_source_of_truth_parity_and_drift(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[2];monkeypatch.chdir(root)
    spec=importlib.util.spec_from_file_location('sot',root/'scripts/validate_v2_source_of_truth.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.validate(root)['status']=='PASS'
    for name in ['configs/pa3_v2_production.yaml','configs/pa3_v2_contract.json','docs/operations/V2_SOURCE_OF_TRUTH_FACTS.json','docs/TRAINING_SOURCE_OF_TRUTH.md']:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,path)
    path=tmp_path/'docs/operations/V2_SOURCE_OF_TRUTH_FACTS.json';facts=json.loads(path.read_text());facts['training']['ppo_epochs']=4;path.write_text(json.dumps(facts))
    with pytest.raises(ValueError,match='drift'):module.validate(tmp_path)
