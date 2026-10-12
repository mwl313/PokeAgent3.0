from __future__ import annotations
import argparse, json
from agent.runtime.config import resolve_profile, launcher_values
p=argparse.ArgumentParser(); p.add_argument('profile'); a=p.parse_args(); r=resolve_profile(a.profile)
print(json.dumps({**r, 'launcher_values': launcher_values(r)}, indent=2, sort_keys=True))
