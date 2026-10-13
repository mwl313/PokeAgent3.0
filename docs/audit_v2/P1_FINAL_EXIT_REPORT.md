# P1 exit — PASS for the frozen 1,220-team self-play pool

Source: `548902547a5d5235f532193a47a77a89abfb1cdd`, isolated V2 worktree.
Evidence: `p1_final/RECEIPT.json`. This supersedes earlier P1 BLOCKED reports;
their failures and intermediate receipts remain intact.

The public-information gate now has a protocol-derived differential oracle:
28 fixtures / 765 boundaries covering Soak, Magic Powder, Burn Up, Double
Shock, Protean, Libero, Mimicry, Forecast, Stance Change, Mega and Illusion.
Opponent species/types come from public Showdown switch/drag/replace/forme/type
messages; privileged `getTypes()` is used only for the owning player's types.
Explicit typeless `???` maps to native type ID 0, not unknown. Snapshot objects
are copied per boundary so future type changes cannot rewrite earlier evidence.

The old global closure's 11 unsupported callers are not reachable from this
pool. The new conservative pool-specific identity closure contains 347 moves
(including Struggle), 153 abilities, 137 held items and 273 species/forms.
It includes compatible Mega stones across the pool, species-guarded natural
forms, present-ability copies/swaps, Mummy, Simple Beam, Sleep Talk and reflected
moves. It records reference hook hashes and fails on unreviewed identity-changing
callers. All reachable IDs have native implementations. Transform/Imposter
remain unsupported in the global engine and are explicitly NOT_REACHABLE here.
Changing the dataset or adding external opponents invalidates this conclusion.
This is not exhaustive proof of every executable handler or full Reg M-C.

A real whole-pool smoke exposed an additional retry integration defect:
`StepResult` said a submitted peer was Wait, while request introspection returned
Normal. The exact binding reproducer failed `1 == 3` before the fix. Batch
request APIs now return a waiting request for a committed side, using a borrowed
request on the ordinary hot path. Native, packed and per-view introspection
share this source. Consumer retries no longer resubmit the waiting peer.

Final checks: **44 relevant Rust tests**, **5 Python retry tests**, **3 Node
closure tests**, all PASS. The Rust suite includes the full existing stored
oracle corpus, information isolation, observation completeness, Illusion,
public types, closure and batch determinism. The prior independent emitted
request verifier passed 1152 fixtures / 25231 boundaries; none of those existing
expectations changed here. A model-free, deterministic random-legal smoke used
all 1220 teams in both seats: **2440 natural completions, 6 retries, 0 operational
errors, 0 unfinished**. Its ~2183 games/s is CPU native execution only, **not PPO
throughput**, and is not a strength or exhaustive mechanics measurement.

P0 historical checkpoints and evaluation evidence remain protected. PPO is
inactive; old League worker/timer remain retired by the user's explicit later
authorization. Safe legacy restore generation remains the P0 1,400,832-match
generation, prohibited as a V2 model initializer.

Next authorized phase: freeze the Observation/Action V2 contract, then P2 and
P3. P2-P5 remain unverified; no scratch production has been launched.
