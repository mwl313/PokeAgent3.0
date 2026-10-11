import hashlib,json,pathlib,sys,time,traceback
ROOT=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0')
OUT=ROOT/'runs/perf/cooling_20261011'
sys.path[:0]=[str(ROOT),str(ROOT/'engine/python')]
import torch,pa3_engine
from agent.model import PA3Config,build_model
from agent.train.native_collector import NativeCollector,NativeCollectorConfig

class CaptureEngine:
    def __init__(self,engine): self.engine=engine
    def __getattr__(self,name): return getattr(self.engine,name)
    def observe_fixed_batch(self,handles,sides):
        try: return self.engine.observe_fixed_batch(handles,sides)
        except Exception as error:
            failures=[]
            for h,side in zip(handles,sides):
                try: self.engine.observe_fixed_batch([h],[side])
                except Exception as e:
                    raw=bytes(self.engine.snapshot(*h))
                    dest=OUT/f'catalogue_bad_slot{h[0]}_side{side}.snapshot'
                    dest.write_bytes(raw)
                    failures.append({'handle':h,'side':side,'error':str(e),'snapshot':str(dest),
                                     'snapshot_sha256':hashlib.sha256(raw).hexdigest()})
            (OUT/'catalogue_failure.json').write_text(json.dumps({'error':str(error),'failures':failures},indent=2)+'\n')
            raise

device=torch.device('cuda:1'); torch.cuda.set_device(device)
torch.manual_seed(20261009)
engine=CaptureEngine(pa3_engine.NativeEngine(str(ROOT/'engine/data'),str(ROOT/'engine/data/training-teams.json'),workers=16))
model=build_model(PA3Config(),device=device)
def collector(seed):
    return NativeCollector(engine,model,NativeCollectorConfig(envs=1024,workers=16,seed=seed,device=str(device),
        observation_mode='fixed',candidate_wire='packed',amp=True,inference_mode=True,telemetry=True),device=device)
result={'profiler':False,'checkpoint':'diagnostic_failed_first.pt','seed_first':20261010,'seed_second':20261012}
try:
    first=collector(20261010); a=first.collect(1024)
    result['first_rows']=len(a.rows); del first,a
    state=torch.load(OUT/'diagnostic_failed_first.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'])
    second=collector(20261012); b=second.collect(1024)
    result.update(second_rows=len(b.rows),reproduced=False)
except Exception as e:
    result.update(reproduced=True,error=str(e),traceback=traceback.format_exc())
(OUT/'catalogue_reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
