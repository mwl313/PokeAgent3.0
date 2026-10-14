import importlib.util,json
from pathlib import Path
import pytest


def launcher():
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('v2_operational_launcher',root/'scripts/run_ddp_ppo.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return root,module


def test_v2_profile_keeps_ppo_hyperparameters_and_requires_durable_stats(tmp_path,monkeypatch):
    root,module=launcher();monkeypatch.chdir(root)
    old=json.loads((root/'configs/a10_production.yaml').read_text())
    new=json.loads((root/'configs/pa3_v2_production.yaml').read_text())
    assert old['training']==new['training']
    with pytest.raises(SystemExit):module.parse_args(['--profile','configs/pa3_v2_production.yaml'])
    args=module.parse_args(['--profile','configs/pa3_v2_production.yaml',
        '--team-stats-root',str(tmp_path/'stats'),'--rolling-checkpoint-root',str(tmp_path/'rolling'),
        '--experimental-checkpoint',str(tmp_path/'boundary'),'--stop-file',str(tmp_path/'STOP')])
    assert args.deterministic_diagnostic and args.epochs==2 and args.checkpoint_interval_matches==100000
    assert args.microbatch==1024 and args.minibatch==4096 and args.experimental_lr_scale==.5


def test_cooperative_stop_requires_checkpoint_destination():
    _,module=launcher()
    with pytest.raises(SystemExit):module.parse_args(['--stop-file','STOP'])


def test_deterministic_workspace_is_pinned_and_conflicts_rejected(monkeypatch):
    from types import SimpleNamespace
    _,module=launcher();args=SimpleNamespace(deterministic_diagnostic=True)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    module.pin_deterministic_workspace(args)
    import os
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':16:8')
    with pytest.raises(RuntimeError,match='requires'):module.pin_deterministic_workspace(args)
