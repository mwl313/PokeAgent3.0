"""Cold numeric ledger: exact frozen-pool preview ranges and FP16 boundary audit."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pa3_engine
from pa3_engine.observation import parse_batch
from agent.contracts import SPEC,CONTRACT_HASH


def normalize(feature,value):
    if feature['name']=='EventChange':return math.asinh(value)/math.asinh(2147483648)
    return value/float(feature['normalization'].split('/')[1])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];data=root/'engine/data';pool=data/'a10-training-teams.json'
    engine=pa3_engine.NativeEngine(str(data),str(pool),workers=16)
    total=engine.team_count();assert total==1220
    handles=engine.reset_batch(list(range(total)),[(i+1)%total for i in range(total)],[(2026,10,14,i+1) for i in range(total)],[(0,1)]*total)
    fixed,ragged=engine.observe_fixed_batch(handles,[0]*total);payload=parse_batch(fixed,ragged,total)
    rows=[]
    for feature in SPEC['float_features']:
        i=feature['index'];known=payload['float_known'][:,:,i]&payload['token_mask'];values=payload['floats'][:,:,i][known]
        lo,hi=feature['raw_min'],feature['raw_max'];raw=[lo,hi,0]
        raw.extend(v for v in [-6,-1,1,6] if lo<=v<=hi)
        encoded=np.asarray([normalize(feature,v) for v in raw],np.float32);quantized=encoded.astype(np.float16).astype(np.float32)
        if not np.isfinite(quantized).all():raise ValueError(f"FP16 overflow {feature['name']}")
        actual_half=values.astype(np.float16).astype(np.float32)
        row={**feature,'actual_preview_known_values':len(values),
             'actual_preview_normalized_min':float(values.min()) if len(values) else None,
             'actual_preview_normalized_max':float(values.max()) if len(values) else None,
             'actual_preview_unique_fp32':len(np.unique(values)),'actual_preview_unique_fp16':len(np.unique(actual_half)),
             'actual_preview_nonzero_underflows':int(((values!=0)&(actual_half==0)).sum()),
             'boundary_raw':json.dumps(raw),'boundary_fp16':json.dumps(quantized.tolist()),
             'smallest_fp16_subnormal':float(np.nextafter(np.float16(0),np.float16(1))),
             'scope_note':'All 1220 own preview observations; absent event/duration/midbattle ranges are not inferred.'}
        rows.append(row)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='')as f:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary={'semantic_hash':CONTRACT_HASH,'dataset_sha256':hashlib.sha256(pool.read_bytes()).hexdigest(),
             'team_count':total,'fields':len(rows),'preview_underflows':sum(r['actual_preview_nonzero_underflows'] for r in rows),
             'limitations':['Preview ranges are not full dynamic ranges. Full declared domain endpoints and signed small event changes are audited separately.',
                            'FP16 quantizes some large-domain adjacent values; no claim of exact recovery of all i32/u16 values.'],
             'ragged_numeric_audit':{'effects_duration_and_stacks':'i16 /32768: -32768..32767 maps to [-1,1); step 1/32768 survives FP16',
                                    'base_moves_pp':'u8 /255; finite [0,1]',
                                    'move_effects':'chance /100, signed boosts /6, finite heal ratio; explicit known masks',
                                    'categorical_ids':'not FP16 quantized'}}
    output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
