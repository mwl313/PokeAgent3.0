#!/usr/bin/env python3
"""Local incident sink; optionally creates a bounded Codex request when explicitly enabled."""
from __future__ import annotations
import argparse, json, shutil, subprocess, time
from pathlib import Path

def dispatch(incident, operations: Path, enable_codex=False):
    out=operations/'dispatch'; out.mkdir(parents=True,exist_ok=True)
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()); req=out/f'{stamp}_{incident.get("type","incident")}.json'; req.write_text(json.dumps(incident,indent=2,sort_keys=True)+'\n')
    result={'local_request':str(req),'codex':'NOT_CONFIGURED'}
    if enable_codex and shutil.which('codex'):
        try:
            subprocess.run(['codex','exec','--help'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
            result['codex']='AVAILABLE_NOT_INVOKED'
        except Exception as e: result['codex']=f'UNAVAILABLE:{e}'
    return result
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('incident_json'); ap.add_argument('--operations',type=Path,required=True); ap.add_argument('--enable-codex',action='store_true'); a=ap.parse_args(); print(json.dumps(dispatch(json.loads(Path(a.incident_json).read_text()),a.operations,a.enable_codex),indent=2))
