"""Bounded same-payload comparison of bulk versus legacy offset rescanning."""
import argparse,json,time,statistics
from unittest.mock import patch
import pa3_engine
from pa3_engine import observation as wire
from agent.types.observation import ObservationBatch
from agent.ppo.recompute_safety import compact_hash


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
 engine=pa3_engine.NativeEngine('engine/data','engine/data/a10-training-teams.json',workers=2)
 result={}
 bulk=wire.split_batch_ragged_all
 for n in (32,128,1024):
  handles=engine.reset_batch([0]*n,[1]*n,[(2026,10,14,i+1) for i in range(n)],[(0,1)]*n)
  fixed,ragged=engine.observe_fixed_batch(handles,[0]*n);payload=wire.parse_batch(fixed,ragged,n)
  row={};hashes=[]
  for name,decoder in [('rescan',lambda payload:[wire.split_batch_ragged(payload,i) for i in range(n)]),('bulk',bulk)]:
   times=[]
   with patch.object(wire,'split_batch_ragged_all',decoder):
    for _ in range(3):
     start=time.perf_counter();observation=ObservationBatch.from_native_payload(payload);times.append(time.perf_counter()-start)
   hashes.append(compact_hash(observation.to_compact_numpy()))
   row[name]={'seconds':times,'median_seconds':statistics.median(times)}
  row['identical_model_input_hash']=hashes[0]==hashes[1]
  if not row['identical_model_input_hash']:raise RuntimeError('decode changed model inputs')
  row['speedup']=row['rescan']['median_seconds']/row['bulk']['median_seconds'];result[n]=row
 with open(args.output,'w') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result,indent=2))


if __name__=='__main__':main()
