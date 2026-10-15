# V2 recompute failure analysis at 98,304 matches

## Status

The production service stopped before the PPO update for the attempted next
iteration. The last completed learner update is the 98,304-match boundary.
Both ranks agreed on the model and Adam digests for every completed iteration;
the failed rollout was not applied to the optimizer. The service was not
restarted during this analysis.

Evidence is preserved at
`runs/pa3_v2_scratch_20261014T060525Z/reports/v2-final-8052606/continuation.json.rank{0,1}.iteration25.recompute.pt`.

## What failed

The primary gate sampled 576 rows in six strata. Five rank-0 rows and one
rank-1 row were outliers, all in the same stratum:

`kind1_branches2_actor1` (normal request, two branches, current-policy actor).

The rank-1 maximum was `0.0029621124`, above the unchanged `0.001` tolerance.
The other strata were at or below `2.861e-6`. The rank-0 outliers were
`0.0013530` to `0.0030243`.

The decisive rank-1 row was row `26896` (match `2722`, request `3`, seat `1`,
legal counts `[9, 11]`, selected actions `[5, 5]`):

| path | log probability |
|---|---:|
| stored actor value | -3.0076684952 |
| actor-group replay AMP | -3.0076684952 |
| actor-group repeated AMP | -3.0076684952 |
| sampled batched AMP | -3.0106306076 |
| actor-group FP32 | -3.0096492763 |
| isolated FP32 | -3.0096507072 |
| isolated AMP | -3.0076670642 |

The observation, legal actions, selected actions and stored log probability were
verified identical. Actor replay was repeatable, and the isolated AMP replay
matched the actor value. Therefore this is not an action-mask mutation, RNG
consumption, collector identity error, or DDP disagreement.

## Controlled replay

Loading the preserved model and rows and running the same CUDA paths reproduced
the values above. The target observation's encoder output differed between the
two AMP batches by maximum absolute `0.00201899` and mean absolute `0.00021485`.
The same observation was bitwise identical between the two FP32 batches.

The actor group had 1,320 rows. Its first physical encoder chunk was exactly
1,024 rows. The sampled recompute batch had 576 rows and was internally padded
to 1,024 rows by duplicating the first observation. Thus both paths used a
1,024-row physical chunk, but one was a full actor chunk and the other contained
448 duplicated padding rows. The AMP Transformer/SDPA execution produced
different values for the valid row under those two batch compositions. The
Mode-D scorer and GRU are FP32; the observed difference is already present in
the AMP encoder tokens.

This is a real batch-composition-dependent FP16 numerical discrepancy. The
previous fixed physical-row change reduced most variation, but it does not make
an unpadded 1,024-row actor chunk numerically identical to a padded 1,024-row
recompute chunk for every learned model state and request stratum.

## Why the safety response was correct

The secondary audit accepted neither interpretation: actor AMP was about
`0.00198` from its independent FP32 reference, while the sampled AMP result
was about `0.000981` from that reference. The evidence therefore did not meet
the existing rule that both AMP paths stay within tolerance of stable FP32.
Both ranks rejected the update before any PPO math ran. No tolerance was
relaxed, no failed rollout was counted, and no checkpoint was published for the
unfinished 102,400-match boundary.

## Recommended next experiment

Keep the gate fail-closed. The next bounded experiment should compare, on the
preserved failure artifact, a canonical replay that uses the exact actor-group
partition and padding policy against the current sampled-batch replay, while
recording encoder-token deltas by layer. If the discrepancy remains, the safe
production options are to make the audited policy encoder path FP32 or to make
the recompute use the exact actor-group forward shape. Either option requires a
new parity proof and source-migration receipt. Do not resume from 98,304 as if it
were a complete checkpoint: the approved resumable state remains the verified
57,344-match checkpoint.
