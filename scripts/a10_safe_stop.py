#!/usr/bin/env python3
"""Request a safe stop; never kills a running learner."""
import argparse,time,json
from pathlib import Path
from a10_status import DEFAULT_ROOT,snapshot
ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=DEFAULT_ROOT); a=ap.parse_args(); s=snapshot(a.root); p=a.root/'operations/stop.requested'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({'requested_at':time.time(),'committed_matches':s['committed_matches'],'iteration':s['last_complete_iteration']},indent=2)+'\n'); print(f'wrote {p}; launcher must honor this at a boundary; no signal was sent')
