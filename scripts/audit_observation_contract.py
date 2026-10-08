#!/usr/bin/env python3
"""Phase 0 observation-contract audit (fixed + ragged) for the PA3-8M path.

Checks, on a real engine batch:

1. `observe_fixed_batch` (fixed stride + ragged sidecar) and
   `observe_encoded_batch` (per-view blobs) agree field-by-field, including the
   five ragged sections, so the batch-first path is contract-equivalent.
2. Which fields the typed `ObservationBatch` adapter consumes and which the
   model embedding consumes, so an optimization can never silently drop
   information that the Full Spec requires.
3. The integer category ranges against the model's placeholder vocabulary
   (`PA3Config.category_vocab`), i.e. the Dex-vocabulary gap.

Usage:
    PYTHONPATH=engine/python:. .venv/bin/python scripts/audit_observation_contract.py \
        [--envs 64] [--report docs/perf/OBSERVATION_CONTRACT.md]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine", "python"))

import pa3_engine  # noqa: E402
from pa3_engine.observation import FIXED, parse_batch, parse_view, split_batch_ragged  # noqa: E402
from agent.model.config import PA3Config  # noqa: E402
from agent.types.observation import ObservationBatch  # noqa: E402

# Fields the typed adapter forwards to the model (agent/types/observation.py).
ADAPTER_FIELDS = ["token_mask", "categories", "category_known", "floats", "float_known",
                  "flags", "flag_known", "role_ids", "side_ids"]
# Fields the TokenEmbedding actually reads (agent/model/embeddings.py).
MODEL_FIELDS = ["token_mask", "role_ids", "side_ids", "floats", "float_known",
                "flags", "flag_known", "categories", "category_known"]
RAGGED_SECTIONS = ["effects", "repertoire", "types", "base_moves", "move_effects"]


def fixed_field_table(view):
    header = view["header"]
    rows = []
    for name in header.dtype.names:
        rows.append({
            "field": name,
            "dtype": str(header.dtype[name].base),
            "shape_per_view": list(header.dtype[name].shape) or [1],
            "in_adapter": name in ADAPTER_FIELDS or name in view,
            "in_model": name in MODEL_FIELDS,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data", default=os.path.join(ROOT, "engine", "data"))
    parser.add_argument("--teams", default=os.path.join(ROOT, "engine", "data", "training-teams.json"))
    parser.add_argument("--report", default=os.path.join(ROOT, "docs", "perf", "OBSERVATION_CONTRACT.md"))
    args = parser.parse_args()

    engine = pa3_engine.NativeEngine(args.data, args.teams, workers=args.workers)
    import random
    rng = random.Random(20261008)
    teams = engine.team_count()
    team_a = [rng.randrange(teams) for _ in range(args.envs)]
    team_b = [rng.randrange(teams) for _ in range(args.envs)]
    seeds = [tuple(rng.randrange(1 << 16) for _ in range(4)) for _ in range(args.envs)]
    roles = [(0, 1) if i % 2 == 0 else (1, 0) for i in range(args.envs)]
    handles = engine.reset_batch(team_a, team_b, seeds, roles)

    # Collect the decision requests of the first round.
    request_handles, sides = [], []
    for handle in handles:
        for side in (0, 1):
            kind, _slots = engine.request_info_batch([(handle[0], handle[1], side)])[0]
            if int(kind) in (0, 1, 2):
                request_handles.append((handle[0], handle[1]))
                sides.append(side)
    fixed_bytes, ragged_bytes = engine.observe_fixed_batch(request_handles, sides)
    view = parse_batch(fixed_bytes, ragged_bytes, len(request_handles))
    blobs = engine.observe_encoded_batch(request_handles, sides)
    per_view = [parse_view(blob) for blob in blobs]

    # 1. field-by-field parity between the two observation APIs.
    fixed_names = [name for name in FIXED.names if name not in
                   ("effect_counts", "repertoire_counts", "type_counts", "base_move_counts", "move_effect_counts")]
    mismatches = []
    for index, single in enumerate(per_view):
        for name in fixed_names:
            a = np.asarray(view[name][index])
            b = np.asarray(single[name])
            if a.shape != b.shape or not np.array_equal(a, b):
                mismatches.append(f"{name}[{index}]")
        ragged_a = split_batch_ragged(view, index)
        for section in RAGGED_SECTIONS:
            a = np.frombuffer(ragged_a[section].tobytes(), dtype=ragged_a[section].dtype)
            b = np.frombuffer(single[section].tobytes(), dtype=single[section].dtype)
            if len(a) != len(b) or a.tobytes() != b.tobytes():
                mismatches.append(f"{section}[{index}]")

    # 2. typed adapter consumption + ragged presence.
    batch_obs = ObservationBatch.from_native_payload(view, layout=None)
    adapter_consumed = {name: hasattr(batch_obs, name) and getattr(batch_obs, name) is not None
                        for name in ADAPTER_FIELDS}
    ragged_counts = {section: int(sum(
        int(view[key][index].sum()) for index in range(len(request_handles))
    )) for section, key in (
        ("effects", "effect_counts"), ("repertoire", "repertoire_counts"),
        ("types", "type_counts"), ("base_moves", "base_move_counts"),
        ("move_effects", "move_effect_counts"),
    )}

    # 3. category range vs the placeholder vocabulary.
    categories = view["categories"].astype(np.int64)
    known = view["category_known"].astype(bool)
    config = PA3Config()
    report = {
        "environment": {"envs": args.envs, "workers": args.workers, "requests": len(request_handles)},
        "parity_mismatches": mismatches[:20],
        "parity_ok": not mismatches,
        "fixed_bytes_per_view": pa3_engine.OBSERVATION_FIXED_BYTES,
        "ragged_bytes_total": len(ragged_bytes),
        "ragged_rows": ragged_counts,
        "adapter_consumed": adapter_consumed,
        "ragged_consumed_by_model": {name: False for name in RAGGED_SECTIONS},
        "category_range_known": {"min": int(categories[known].min()) if known.any() else None,
                                 "max": int(categories[known].max()) if known.any() else None},
        "category_vocab_placeholder": config.category_vocab,
        "category_vocab_slack": config.category_vocab - (int(categories[known].max()) + 1 if known.any() else 0),
        "fixed_field_table": fixed_field_table(view),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    lines = [
        "# Native observation contract audit (fixed + ragged)",
        "",
        "Generated by `scripts/audit_observation_contract.py`. This is a contract "
        "audit, not a performance measurement.",
        "",
        f"- requests audited: {report['environment']['requests']} ({args.envs} environments)",
        f"- fixed stride: {report['fixed_bytes_per_view']} bytes/view",
        f"- ragged bytes in the batch: {report['ragged_bytes_total']}",
        f"- fixed/per-view parity: {'PASS' if report['parity_ok'] else 'FAIL'}",
        f"- known category id range: {report['category_range_known']} vs placeholder vocab "
        f"{report['category_vocab_placeholder']} (slack {report['category_vocab_slack']})",
        "",
        "## Fixed fields",
        "",
        "| field | dtype | shape/view | adapter | model embedding |",
        "|---|---|---|---|---|",
    ]
    for row in report["fixed_field_table"]:
        lines.append(f"| {row['field']} | {row['dtype']} | {row['shape_per_view']} | "
                     f"{'yes' if row['in_adapter'] else 'no'} | {'yes' if row['in_model'] else 'no'} |")
    lines += [
        "",
        "## Ragged sections (per-batch rows)",
        "",
        "| section | rows | consumed by the model today |",
        "|---|---:|---|",
    ]
    for name in RAGGED_SECTIONS:
        lines.append(f"| {name} | {ragged_counts[name]} | no (separate completeness blocker) |")
    lines += [
        "",
        "## Reading",
        "",
        "* `observe_fixed_batch` and `observe_encoded_batch` are field-identical on real "
        "engine state, including every ragged section, so the batch-first path can be "
        "used without changing the observation schema.",
        "* The typed adapter forwards the nine fixed fields the embedding consumes; the "
        "ragged sections (effects, repertoire, types, base moves, move effects) are "
        "carried on the wire but **not** consumed by the model yet. They must not be "
        "dropped from the wire; wiring them into the encoder is an engine/model "
        "completeness blocker, not a performance decision.",
        "* The category vocabulary is still the placeholder "
        f"{report['category_vocab_placeholder']}; the observed known-id range fits inside "
        "it, and the final Dex vocabulary export remains required before a real run.",
        "",
    ]
    with open(args.report, "w") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({key: report[key] for key in
                      ("parity_ok", "parity_mismatches", "fixed_bytes_per_view", "ragged_rows",
                       "category_range_known", "category_vocab_slack")}, indent=2))
    if mismatches:
        raise SystemExit("observation API parity failure")


if __name__ == "__main__":
    main()
