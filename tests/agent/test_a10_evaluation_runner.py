import json
import pytest
from scripts.a10_evaluation_runner import run_job

def test_unknown_policy_rejected(tmp_path):
 (tmp_path/'evaluation').mkdir(); (tmp_path/'evaluation/a10_registry.json').write_text(json.dumps({'policies':{}}))
 with pytest.raises(ValueError): run_job(tmp_path,'a','b')

def test_same_policy_rejected(tmp_path):
 (tmp_path/'evaluation').mkdir(); p={'path':'/tmp/x','committed_matches':1}; (tmp_path/'evaluation/a10_registry.json').write_text(json.dumps({'policies':{'a':p}}))
 with pytest.raises(ValueError): run_job(tmp_path,'a','a')
