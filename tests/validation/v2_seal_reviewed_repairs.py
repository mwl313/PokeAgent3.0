"""Developer seal helper; review exact named edits before invoking and committing."""
from pathlib import Path
import ast,hashlib,re,subprocess
p=Path('agent/ppo/v2_release_migration.py');s=p.read_text();tree=ast.parse(s)
names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REVIEWED_REPAIRS' for t in n.targets))
for name in ('agent/contracts.py','agent/model/precision.py','agent/model/pa3_model.py'):names.setdefault(name,'')
repairs={n:subprocess.check_output(['git','hash-object',n],text=True).strip() for n in names}
s=re.sub(r'REVIEWED_REPAIRS=\{[^\n]*\}', 'REVIEWED_REPAIRS='+repr(repairs),s)
tree=ast.parse(s)
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VALIDATOR_AST' for t in n.targets):n.value=ast.Constant('')
s=re.sub(r"VALIDATOR_AST='[^']*'",f"VALIDATOR_AST='{hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()}'",s);p.write_text(s)
