# Showdown smoke-test readiness assessment

* **Test A** score against branch-wise uniform legal random: 0.9886 (W 8099 / L 93 / D 0).
* **Test B** score against lexicographic legal: 0.9990 (W 2046 / L 2 / D 0).

## A. Basic game competence

The checkpoint wins the large majority of seat-swapped games against the branch-wise uniform legal random opponent.

## B. Learned behaviour

The model is compared directly with the deterministic lexicographic legal baseline on a disjoint frozen fixture set (Test B).

## C. Gameplay quality

Decision-level traces were inspected for targeting, switching, Protect and Mega use. See `behavior_report.md`; no fabricated strategic explanation is used.

## D. Technical correctness

* Illegal actions: 0.
* Operational errors: 0.
* Coverage and duplicate checks were run per suite; hard gate: PASS.

## E. Showdown readiness

This evaluation covers the offline engine, observation/action contract and CPU policy path only. It does **not** prove that the live Showdown adapter, network protocol, timer handling or deployment wiring works.

## Recommendation

**READY_FOR_SHOWDOWN_SMOKE_TEST** — offline correctness gates pass and the model beats the random baseline by a wide margin. Test B is a sanity check only because the lexicographic baseline is itself much weaker than random. This is a smoke-test recommendation, not a ladder-strength claim.
