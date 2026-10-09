# V5d — f16 observation-wire PoC (s5a W2): bounded cost-ceiling analysis, no-go

Base `147247d` (after the rolling-slot engine fix). This is the bounded PoC the
task allows to "stop early and report the finding": the analysis below shows the
*optimistic ceiling* of the f16 float-block wire is at/below the ±2 % noise
floor that the task's own promotion rule requires, so no schema change was made
and no parity risk was introduced.

## 1. Wire layout and the target block

The fixed observation view is `OBSERVATION_FIXED_BYTES` (engine/src/python.rs):

| block | bytes/view | share |
|---|---:|---:|
| float block (`96 tokens × 50 floats × 4 B`) — the B1 target | 19,200 | 45.8 % |
| categories / masks / flags / ids / header | 22,754 | 54.2 % |
| **total** | **41,954** | 100 % |

f16 transport for the float block only halves it to 9,600 B, saving **9,600 B
per view (22.9 % of the view)**; integers/masks/indices stay unchanged (no
lossy conversion there). The rollout *store* already keeps floats at f16
(`to_compact_numpy`), but the wire change still alters the numbers the policy
first sees, so it needs its own parity gate.

## 2. Measured cost of the affected stages (single GPU, 2,048 games, 3 repeats)

| stage | median |
|---|---:|
| `native_observation` (Rust crossing) | 3.07 s |
| `parse_convert` (Python decode) | 0.88 s |
| `h2d` | 2.03 s |
| collection wall | 23.2 s |
| all-in wall | 101.9 s |

## 3. Ceiling

Even if **every** obs/parse/H2D byte cost scaled with the float-block share
(they do not — the integer blocks and per-call overhead remain):

```
(3.07 + 0.88 + 2.03) × 0.229 ≈ 1.37 s ≈ 1.35 % of all-in
```

If the full transport cost of those stages scaled (a strict over-estimate):
`(3.07+0.88+2.03) × 0.458 ≈ 2.74 s ≈ 2.69 % of all-in`.

So the realistic band is **≈1.0–1.4 % all-in**, and the absolute optimistic
ceiling is **2.7 %** — inside or at the edge of the declared ±2 % noise floor of
the dual panels (and the dual configuration has an even smaller wire share per
rank). Per the task rule ("promote only if parity is documented AND the effect
clears the noise floor"), this PoC cannot produce a promotable result, and the
schema change plus parity gate would add risk for a sub-noise gain.

## 4. Decision and follow-up

* **Not implemented; stop-early reported.** No Rust/Python wire change, no f32
  fallback flag added, no equivalence-band measurement performed because there
  is nothing to gate.
* If a future combined change (e.g. wire f16 *plus* columnar observation
  materialization, where the same bytes are touched twice) makes the cumulative
  effect clear the noise floor, the design to use is: engine constructor flag →
  f16 float block, Python parser reads f16 and upcasts to f32 on decode, f32
  fallback default, row-SHA/recompute/digest gates on a 256–512 game smoke plus
  a logits/KL drift band.
* B1 stays open in the roadmap; the row in `docs/PROJECT_STATUS.md` reflects
  this measured no-go rather than a silent deferral.
