import json
import pytest
from agent.ppo import source_migration as migration


def test_recovery_migration_requires_exact_clean_target_and_scope(tmp_path, monkeypatch):
    target = 'f'*40
    changes = ['agent/ppo/telemetry.py', 'scripts/run_ddp_ppo.py']
    receipt = {'schema':'a10-exact-recovery-migration-v1','source_sha':migration.BASE,
               'target_sha':target,'changed_runtime_files':changes}
    path = tmp_path/'receipt.json'; path.write_text(json.dumps(receipt))
    state = {'dirty':'', 'changes':changes}
    def git(root, *args):
        if args[0] == 'rev-parse': return target
        if args[0] == 'status': return state['dirty']
        if '--name-only' in args: return '\n'.join(state['changes'])
        return 'protected change'
    monkeypatch.setattr(migration, 'git', git)
    assert migration.validate_recovery_migration(path,tmp_path,migration.BASE,target)['target_sha']==target
    with pytest.raises(ValueError,match='source/target'):
        migration.validate_recovery_migration(path,tmp_path,migration.BASE,'a'*40)
    state['dirty']=' M agent/ppo/learner.py'
    with pytest.raises(ValueError,match='clean'):
        migration.validate_recovery_migration(path,tmp_path,migration.BASE,target)
    state['dirty']='';state['changes'].append('agent/ppo/learner.py')
    with pytest.raises(ValueError,match='protected'):
        migration.validate_recovery_migration(path,tmp_path,migration.BASE,target)
    with pytest.raises(ValueError,match='unsupported'):
        migration.validate_recovery_migration(path,tmp_path,'other',target)
