"""Replay the captured engine failure without CUDA, PPO or a profiler."""
import json,pathlib,sys
ROOT=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0')
sys.path.insert(0,str(ROOT/'engine/python'))
import pa3_engine
out=ROOT/'runs/perf/cooling_20261011'
engine=pa3_engine.NativeEngine(str(ROOT/'engine/data'),str(ROOT/'engine/data/training-teams.json'),workers=1)
handle=engine.reset_batch([0],[1],[(1,2,3,4)],[(0,1)])[0]
engine.restore(handle[0],handle[1],list((out/'catalogue_bad_slot942_side0.snapshot').read_bytes()))
result={'requires_cuda':False,'requires_profiler':False,'sides':[]}
for side in (0,1):
    try:
        engine.observe_fixed_batch([handle],[side])
        result['sides'].append({'side':side,'reproduced':False})
    except ValueError as e:
        result['sides'].append({'side':side,'reproduced':True,'error':str(e)})
(out/'catalogue_snapshot_reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
