"""Small fixed-policy panel runner for Track A and future Track C.

The evaluator owns no learner, optimizer, scheduler, or training counters.  A
fixture is a complete immutable battle input; both seat assignments are
represented explicitly so paired comparisons use the same teams and seeds.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch

from agent.model import PA3Config, build_model
from agent.model.pa3_model import PA3Model
from agent.train.native_collector import NativeCollector, NativeCollectorConfig
from agent.types.observation import ObservationBatch, ObservationLayout
from agent.types.requests import RequestKind


@dataclass(frozen=True)
class EvaluationFixture:
    fixture_id: int
    pair_id: int
    team_a: int
    team_b: int
    seed: tuple[int, int, int, int]
    policy_side: int


def build_fixture_manifest(*, pairs: int = 2, team_count: int = 32, seed: int = 20261011) -> list[EvaluationFixture]:
    """Create deterministic fixtures with one explicit seat-swapped pair."""
    rng = random.Random(int(seed))
    fixtures: list[EvaluationFixture] = []
    for pair_id in range(int(pairs)):
        team_a, team_b = rng.randrange(team_count), rng.randrange(team_count)
        game_seed = tuple(rng.randrange(1 << 16) for _ in range(4))
        fixtures.append(EvaluationFixture(len(fixtures), pair_id, team_a, team_b, game_seed, 0))
        fixtures.append(EvaluationFixture(len(fixtures), pair_id, team_a, team_b, game_seed, 1))
    return fixtures


def _fixture_json(fixtures: list[EvaluationFixture]) -> list[dict]:
    return [asdict(fixture) | {"seed": list(fixture.seed)} for fixture in fixtures]


def fixture_manifest_hash(fixtures: list[EvaluationFixture]) -> str:
    payload = json.dumps(_fixture_json(fixtures), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class FixedPolicyEvaluator:
    """Run frozen policy versus an anchor on the real engine surface."""

    def __init__(self, engine, *, device: str | torch.device = "cpu", layout: Optional[ObservationLayout] = None,
                 seed: int = 0):
        self.engine = engine
        self.device = torch.device(device)
        self.layout = layout or getattr(engine, "layout", ObservationLayout())
        self.seed = int(seed)
        self._helper = None

    def _observation(self, handle, side):
        if hasattr(self.engine, "observe_payload"):
            return ObservationBatch.from_native_payload(
                self.engine.observe_payload(handle, side), self.layout, device=self.device
            )
        from pa3_engine.observation import parse_view
        blob = self.engine.observe_encoded_batch([handle], [side])[0]
        return ObservationBatch.from_native_payload(parse_view(blob), self.layout).to(self.device)

    def _request_info(self, handle, side):
        if hasattr(self.engine, "request_info"):
            return self.engine.request_info(handle, side)
        return self.engine.request_info_batch([(handle[0], handle[1], side)])[0]

    def _candidates(self, handle, side, prefix):
        return self.engine.candidates_batch([(handle[0], handle[1], side, prefix)])[0]

    @staticmethod
    def _choose(candidates, *, kind, rng, heuristic):
        if not candidates:
            raise RuntimeError("native engine returned no legal candidates")
        if heuristic:
            # Public-information-only deterministic rule: prefer the lowest
            # canonical action tuple. It never invents or filters legality.
            return min(range(len(candidates)), key=lambda index: tuple(candidates[index]))
        return rng.randrange(len(candidates))

    def _model_action(self, model: PA3Model, observation, candidates_by_level, rng):
        if self._helper is None or self._helper.model is not model:
            self._helper = NativeCollector(
                self.engine, model,
                NativeCollectorConfig(record_rows=False, envs=1, workers=1),
                layout=self.layout, device=self.device,
            )
        with torch.inference_mode():
            encoded = model.encode(observation)
            hidden = model.scorer.initial_state(
                1, device=encoded.tokens.device, dtype=encoded.tokens.dtype
            )
            actions = []
            for candidates in candidates_by_level:
                table = self._helper._level_table([candidates])
                step = model.sample_level_step(encoded, hidden, table, generator=rng)
                pick = int(step.pick.item())
                actions.append(tuple(candidates[pick]))
                hidden = step.hidden
        return actions

    def run(self, fixtures: list[EvaluationFixture], *, policy: str = "random",
            model: Optional[PA3Model] = None, opponent: str = "heuristic",
            opponent_model: Optional[PA3Model] = None, max_rounds: int = 2000) -> dict:
        if policy == "checkpoint" and model is None:
            raise ValueError("checkpoint policy requires a frozen model")
        if opponent == "checkpoint" and opponent_model is None:
            raise ValueError("checkpoint opponent requires a frozen model")
        if model is not None:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        if opponent_model is not None:
            opponent_model.eval()
            for parameter in opponent_model.parameters():
                parameter.requires_grad_(False)
        rng = random.Random(self.seed)
        torch_rng = torch.Generator(device=self.device.type).manual_seed(self.seed)
        started = time.perf_counter()
        outcomes = []
        totals = {"wins": 0, "losses": 0, "draws": 0, "operational_errors": 0, "illegal_actions": 0}
        for fixture in fixtures:
            handle = self.engine.reset_batch(
                [fixture.team_a], [fixture.team_b], [fixture.seed],
                [(fixture.policy_side, 1 - fixture.policy_side)],
            )[0]
            terminated = False
            winner = None
            error = None
            for _round in range(max_rounds):
                submissions = []
                done = False
                for side in (0, 1):
                    kind, slots = self._request_info(handle, side)
                    if not RequestKind(int(kind)).is_decision:
                        continue
                    prefix = []
                    candidates_by_level = []
                    side_model = model if side == fixture.policy_side else opponent_model
                    side_kind = policy if side == fixture.policy_side else opponent
                    encoded = hidden = None
                    if side_kind == "checkpoint":
                        observation = self._observation(handle, side)
                        with torch.inference_mode():
                            encoded = side_model.encode(observation)
                            hidden = side_model.scorer.initial_state(
                                1, device=encoded.tokens.device, dtype=encoded.tokens.dtype
                            )
                    for _level in range(len(slots)):
                        candidates = self._candidates(handle, side, prefix)
                        candidates_by_level.append(candidates)
                        if side_kind == "checkpoint":
                            table = self._helper_for(side_model)._level_table([candidates])
                            with torch.inference_mode():
                                step = side_model.sample_level_step(encoded, hidden, table, generator=torch_rng)
                            pick = int(step.pick.item())
                            prefix.append(tuple(candidates[pick]))
                            hidden = step.hidden
                        else:
                            pick = self._choose(candidates, kind=kind, rng=rng,
                                                 heuristic=(side_kind == "heuristic"))
                            prefix.append(candidates[pick])
                    submissions.append((side, prefix))
                if not submissions:
                    error = "no actionable request"
                    totals["operational_errors"] += 1
                    break
                result = self.engine.step_batch([(handle[0], handle[1], submissions)])[0]
                accepted, terminated, truncated, winner, error, *_ = result
                if not accepted:
                    totals["illegal_actions"] += 1
                    error = error or "engine rejected action"
                    break
                if truncated or error is not None:
                    totals["operational_errors"] += 1
                    break
                if terminated:
                    break
            if not terminated and error is None:
                totals["operational_errors"] += 1
                error = "round budget exhausted"
            if error is not None:
                outcome = "error"
            elif winner is None:
                outcome = "draw"
                totals["draws"] += 1
            elif int(winner) == fixture.policy_side:
                outcome = "win"
                totals["wins"] += 1
            else:
                outcome = "loss"
                totals["losses"] += 1
            outcomes.append({"fixture_id": fixture.fixture_id, "pair_id": fixture.pair_id,
                             "policy_side": fixture.policy_side, "outcome": outcome,
                             "winner": winner, "error": error})
        scores = [1.0 if row["outcome"] == "win" else 0.5 if row["outcome"] == "draw" else 0.0
                  for row in outcomes if row["outcome"] in {"win", "loss", "draw"}]
        low, high = self._bootstrap_ci(scores, self.seed)
        return {
            "fixtures": _fixture_json(fixtures), "fixture_manifest_hash": fixture_manifest_hash(fixtures),
            "policy": policy, "opponent": opponent, "matches": len(fixtures), **totals,
            "score_mean": sum(scores) / len(scores) if scores else None,
            "score_ci95": [low, high] if scores else None,
            "confidence_status": "inconclusive" if scores and low <= 0.55 and high >= 0.45 else "descriptive_only",
            "runtime_seconds": time.perf_counter() - started, "outcomes": outcomes,
        }

    def _helper_for(self, model):
        if self._helper is None or self._helper.model is not model:
            self._helper = NativeCollector(
                self.engine, model,
                NativeCollectorConfig(record_rows=False, envs=1, workers=1),
                layout=self.layout, device=self.device,
            )
        return self._helper

    @staticmethod
    def _bootstrap_ci(scores, seed, samples=1000):
        if not scores:
            return None, None
        rng = random.Random(seed)
        means = []
        for _ in range(samples):
            means.append(sum(scores[rng.randrange(len(scores))] for _ in scores) / len(scores))
        means.sort()
        return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def load_frozen_model(path, *, device="cpu", config: Optional[PA3Config] = None) -> PA3Model:
    """Load only a trusted local model/learner checkpoint and freeze it."""
    import os
    path = os.path.abspath(os.fspath(path))
    if "://" in path or not os.path.isfile(path):
        raise ValueError("evaluation checkpoints must be existing local files")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if isinstance(payload, dict) and "learner" in payload:
        state = payload["learner"]["model"]
    model = build_model(config or PA3Config(), device=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
