"""Native multi-battle fixed-policy evaluator.

This deliberately has no PPO learner/rollout buffer.  It shares only the
packed candidate-table conversion with ``NativeCollector`` and drives native
handles directly.  Its v2 sampling uses fixture/seat/decision/branch keyed
uniforms, so outcomes do not depend on batch grouping or another battle.
"""
from __future__ import annotations

import random
import time
from collections import defaultdict

import numpy as np
import torch

from pa3_engine.observation import parse_batch, parse_packed_candidates
from agent.evaluation.fixed_policy import FixedPolicyEvaluator, fixture_manifest_hash, _fixture_json
from agent.train.native_collector import NativeCollector, NativeCollectorConfig
from agent.types.observation import ObservationBatch, ObservationLayout
from agent.types.requests import RequestKind

BATCHED_EVALUATOR_VERSION = "pa3-native-batched-evaluator-v2"
SAMPLING_SEMANTICS = "fixture-seat-decision-branch-keyed-cdf-v2"


class BatchedPolicyEvaluator:
    def __init__(self, engine, *, device="cuda:0", layout=None, seed=0, batch_size=32):
        self.engine, self.device, self.layout = engine, torch.device(device), layout or ObservationLayout()
        self.seed, self.batch_size = int(seed), int(batch_size)
        self._helper = NativeCollector.__new__(NativeCollector)
        self._helper.config = NativeCollectorConfig(record_rows=False)
        self._helper.layout, self._helper.device = self.layout, self.device
        self.timing = defaultdict(float)

    def _uniform(self, fixture, side, decision, level):
        key = f"{self.seed}:{fixture.fixture_id}:{side}:{decision}:{level}"
        return random.Random(key).random()

    @staticmethod
    def _record(packed, row, pick):
        record = packed["records"][int(packed["offsets"][row]) + int(pick)]
        return (int(record[0]), int(record[1]), int(record[2]), int(np.int8(record[3])), int(record[4]), int(record[5]))

    def _sample_model(self, model, encoded, hidden, table, uniforms):
        keys = model.scorer.build_keys(encoded.tokens, table.action_ids[:, 0], table.entity_token[:, 0], table.move_token[:, 0])
        logits = model.scorer.score_keys(keys, hidden, table.mask[:, 0])
        _logprob, probs, _entropy, _kl = model._branch_stats(logits, table.mask[:, 0])
        cdf = probs.detach().float().cpu().cumsum(dim=-1).numpy()
        picks = np.asarray([min(int(np.searchsorted(cdf[i], value, side="right")), cdf.shape[1] - 1) for i, value in enumerate(uniforms)], dtype=np.int64)
        # The table mask ensures each selected index is legal; guard anyway.
        if any(not bool(table.mask[i, 0, int(pick)].item()) for i, pick in enumerate(picks)):
            raise RuntimeError("sampled illegal candidate index")
        pick_tensor = torch.as_tensor(picks, device=self.device, dtype=torch.long)
        return picks, model.scorer.advance_keys(keys[torch.arange(len(picks), device=self.device), pick_tensor], hidden)

    def run(self, fixtures, *, policy="checkpoint", model=None, opponent="checkpoint", opponent_model=None, max_rounds=2000):
        if policy == "heuristic": policy = "lexicographic_legal"
        if opponent == "heuristic": opponent = "lexicographic_legal"
        if policy == "checkpoint" and model is None or opponent == "checkpoint" and opponent_model is None:
            raise ValueError("checkpoint mode requires a frozen model")
        for candidate in (model, opponent_model):
            if candidate is not None:
                candidate.eval()
                for value in candidate.parameters(): value.requires_grad_(False)
        started = time.perf_counter(); peak_before = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        totals = {"wins": 0, "losses": 0, "draws": 0, "operational_errors": 0, "illegal_actions": 0}
        outcomes, groups = [], {}
        for start in range(0, len(fixtures), self.batch_size):
            cohort = fixtures[start:start + self.batch_size]
            handles = self.engine.reset_batch([x.team_a for x in cohort], [x.team_b for x in cohort], [x.seed for x in cohort], [(x.policy_side, 1-x.policy_side) for x in cohort])
            finished, decisions = [False] * len(cohort), [[0, 0] for _ in cohort]
            errors, winners, illegal = [None]*len(cohort), [None]*len(cohort), [False]*len(cohort)
            for _round in range(max_rounds):
                pending = [(env, side) for env in range(len(cohort)) if not finished[env] for side in (0, 1)]
                if not pending: break
                info_start = time.perf_counter(); infos = self.engine.request_info_batch([(handles[e][0], handles[e][1], s) for e, s in pending]); self.timing["request_info"] += time.perf_counter()-info_start
                decision = [(env, side, tuple(int(x) for x in slots)) for (env, side), (kind, slots) in zip(pending, infos) if RequestKind(int(kind)).is_decision]
                if not decision:
                    for env, _side in pending: errors[env] = "no actionable request"; finished[env] = True; totals["operational_errors"] += 1
                    break
                obs_start = time.perf_counter(); fixed, ragged = self.engine.observe_fixed_batch([(handles[e][0], handles[e][1]) for e,_,_ in decision], [s for _,s,_ in decision]); observations = ObservationBatch.from_native_payload(parse_batch(fixed, ragged, len(decision)), layout=self.layout).to(self.device); self.timing["observation"] += time.perf_counter()-obs_start
                by_policy = defaultdict(list)
                for index, (env, side, slots) in enumerate(decision):
                    mode, selected_model = ((policy, model) if side == cohort[env].policy_side else (opponent, opponent_model))
                    by_policy[(id(selected_model), mode, len(slots))].append(index)
                submissions = defaultdict(list)
                for (_model_id, mode, branches), indices in by_policy.items():
                    subset = observations.select(indices); sample_model = model if decision[indices[0]][1] == cohort[decision[indices[0]][0]].policy_side else opponent_model
                    encoded = hidden = None
                    if mode == "checkpoint":
                        model_start = time.perf_counter()
                        with torch.inference_mode():
                            encoded = sample_model.encode(subset); hidden = sample_model.scorer.initial_state(len(indices), device=self.device, dtype=encoded.tokens.dtype)
                        self.timing["model"] += time.perf_counter()-model_start
                    prefix = np.zeros((len(indices), branches, 6), dtype=np.uint8); walk_handles=[(handles[decision[i][0]][0], handles[decision[i][0]][1]) for i in indices]; walk_sides=[decision[i][1] for i in indices]
                    for level in range(branches):
                        candidate_start=time.perf_counter(); count_bytes, action_bytes, _ = self.engine.candidates_packed_walk(walk_handles, walk_sides, level, np.ascontiguousarray(prefix[:, :level, :]).tobytes()); packed=parse_packed_candidates(count_bytes, action_bytes, len(indices)); self.timing["candidates"] += time.perf_counter()-candidate_start
                        if np.any(packed["counts"] <= 0): raise RuntimeError("native engine returned an empty legal candidate set")
                        if mode == "checkpoint":
                            table=self._helper._level_table_packed(packed["counts"], packed["records"], packed["offsets"])
                            uniforms=[self._uniform(cohort[decision[idx][0]], decision[idx][1], decisions[decision[idx][0]][decision[idx][1]], level) for idx in indices]
                            model_start=time.perf_counter()
                            with torch.inference_mode(): picks, hidden=self._sample_model(sample_model, encoded, hidden, table, uniforms)
                            self.timing["model"] += time.perf_counter()-model_start
                        else:
                            picks=[]
                            for local, idx in enumerate(indices):
                                count=int(packed["counts"][local])
                                if mode == "lexicographic_legal":
                                    rows=[self._record(packed, local, p) for p in range(count)]; picks.append(min(range(count), key=lambda p: rows[p]))
                                else: picks.append(int(self._uniform(cohort[decision[idx][0]], decision[idx][1], decisions[decision[idx][0]][decision[idx][1]], level)*count))
                            picks=np.asarray(picks, dtype=np.int64)
                        for local, pick in enumerate(picks): prefix[local, level] = packed["records"][int(packed["offsets"][local])+int(pick)]
                    for local, idx in enumerate(indices):
                        env, side, _slots = decision[idx]; action=[self._record({"records": prefix[local], "offsets": np.arange(branches+1)}, level, 0) for level in range(branches)]
                        submissions[env].append((side, action)); decisions[env][side] += 1
                step_start=time.perf_counter(); result_rows=self.engine.step_batch([(handles[e][0], handles[e][1], submissions[e]) for e in sorted(submissions)]); self.timing["step"] += time.perf_counter()-step_start
                for env, result in zip(sorted(submissions), result_rows):
                    accepted, terminated, truncated, winner, error, *_ = result
                    if not accepted: totals["illegal_actions"] += 1; illegal[env]=True; errors[env]=error or "engine rejected action"; finished[env]=True
                    elif truncated or error is not None: totals["operational_errors"] += 1; errors[env]=error or "truncated"; finished[env]=True
                    elif terminated: winners[env]=winner; finished[env]=True
            for env, fixture in enumerate(cohort):
                if not finished[env]: totals["operational_errors"] += 1; errors[env]="round budget exhausted"
                if errors[env]: outcome="error"
                elif winners[env] is None: outcome="draw"; totals["draws"] += 1
                elif int(winners[env]) == fixture.policy_side: outcome="win"; totals["wins"] += 1
                else: outcome="loss"; totals["losses"] += 1
                outcomes.append({"fixture_id":fixture.fixture_id,"pair_id":fixture.pair_id,"policy_side":fixture.policy_side,"outcome":outcome,"team_a":fixture.team_a,"team_b":fixture.team_b,"winner":winners[env],"error":errors[env],"illegal_action":illegal[env]})
        scores=[1 if x["outcome"]=="win" else .5 if x["outcome"]=="draw" else 0 for x in outcomes if x["outcome"] != "error"]
        pairs=FixedPolicyEvaluator._paired_scores(outcomes); low, high=FixedPolicyEvaluator._bootstrap_ci(pairs, self.seed)
        wall=time.perf_counter()-started
        for row in outcomes:
            key=f"{row['team_a']}:{row['team_b']}"; bucket=groups.setdefault(key,{"team_a":row["team_a"],"team_b":row["team_b"],"matches":0,"wins":0,"losses":0,"draws":0,"operational_errors":0,"illegal_actions":0})
            bucket["matches"] += 1
            if row["outcome"] in {"win","loss","draw"}: bucket[{"win":"wins","loss":"losses","draw":"draws"}[row["outcome"]]] += 1
            else: bucket["operational_errors"] += 1; bucket["illegal_actions"] += int(row["illegal_action"])
        return {"fixtures":_fixture_json(fixtures),"fixture_manifest_hash":fixture_manifest_hash(fixtures),"matches":len(fixtures),**totals,"outcomes":outcomes,"team_group_results":groups,"score_mean":sum(scores)/len(scores) if scores else None,"paired_score_mean":sum(pairs)/len(pairs) if pairs else None,"paired_fixtures":len(pairs),"score_ci95":[low,high] if scores else None,"runtime_seconds":wall,"evaluator_version":BATCHED_EVALUATOR_VERSION,"sampling":SAMPLING_SEMANTICS,"timing_seconds":dict(self.timing),"gpu_peak_bytes":(torch.cuda.max_memory_allocated(self.device)-peak_before) if self.device.type=="cuda" else 0}
