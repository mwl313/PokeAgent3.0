"""Bridge-level determinism, snapshot/replay and layout checks (development only).

Build first:  bash scripts/build_python.sh
Run:          PYTHONPATH=engine/python .venv/bin/python engine/python/test_bridge.py

These checks exercise only the native engine through PyO3. The reference
implementation is never executed here.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pa3_engine  # noqa: E402
from pa3_engine.observation import parse_batch, parse_view, split_batch_ragged  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "engine", "data")
TEAMS = os.path.join(DATA, "training-teams.json")


def first_action(engine, handle, side):
    prefix = []
    while True:
        candidates = engine.candidates(handle[0], handle[1], side, prefix)
        if not candidates:
            return prefix
        prefix.append(candidates[0])


def pending(engine, handle):
    out = []
    for side in (0, 1):
        kind = engine.request_kind(handle[0], handle[1], side)
        if kind in (0, 1, 2):
            out.append((side, first_action(engine, handle, side)))
    return out


def test_layout(engine):
    handle = engine.reset_batch([0], [1], [(1, 2, 3, 4)], [(0, 1)])[0]
    blobs = engine.observe_encoded_batch([handle], [0])
    assert len(blobs) == 1
    view = pa3_engine.parse_view(blobs[0])
    assert view["schema_version"] == pa3_engine.SCHEMA_VERSION
    assert view["token_mask"].shape == (pa3_engine.OBSERVATION_TOKENS,)
    assert int(view["token_mask"].sum()) >= 1
    assert len(blobs[0]) == pa3_engine.OBSERVATION_FIXED_BYTES + int(
        view["effect_counts"].sum()
    ) * pa3_engine.EFFECT.itemsize + int(view["repertoire_counts"].sum()) * pa3_engine.REPERTOIRE.itemsize + int(
        view["type_counts"].sum()
    ) * pa3_engine.TYPE.itemsize + int(view["base_move_counts"].sum()) * pa3_engine.BASE_MOVE.itemsize + int(
        view["move_effect_counts"].sum()
    ) * pa3_engine.MOVE_EFFECT.itemsize
    return handle


def test_fixed_batch(engine):
    """`observe_fixed_batch` must decode to exactly the per-view blobs."""
    handles = engine.reset_batch(
        [0, 1, 2, 3], [4, 5, 6, 7], [(9, 8, 7, 6)] * 4, [(0, 1), (1, 0), (0, 1), (1, 0)]
    )
    queries = []
    for handle in handles:
        for side in (0, 1):
            queries.append((handle, side))
    hs = [h for h, _ in queries]
    sides = [s for _, s in queries]
    blobs = engine.observe_encoded_batch(hs, sides)
    fixed, ragged = engine.observe_fixed_batch(hs, sides)
    batch = parse_batch(fixed, ragged, len(blobs))
    for index, blob in enumerate(blobs):
        single = parse_view(blob)
        assert batch["schema_version"][index] == single["schema_version"]
        assert (batch["token_mask"][index] == single["token_mask"]).all()
        assert (batch["categories"][index] == single["categories"]).all()
        assert (batch["category_known"][index] == single["category_known"]).all()
        assert (batch["floats"][index] == single["floats"]).all()
        assert (batch["float_known"][index] == single["float_known"]).all()
        assert (batch["flags"][index] == single["flags"]).all()
        assert (batch["flag_known"][index] == single["flag_known"]).all()
        rows = split_batch_ragged(batch, index)
        for key, flat in (
            ("effects", "effect_counts"),
            ("repertoire", "repertoire_counts"),
            ("types", "type_counts"),
            ("base_moves", "base_move_counts"),
            ("move_effects", "move_effect_counts"),
        ):
            assert len(rows[key]) == int(batch[flat][index].sum())
        assert rows["effects"].tobytes() == single["effects"].tobytes()
        assert rows["repertoire"].tobytes() == single["repertoire"].tobytes()
        assert rows["types"].tobytes() == single["types"].tobytes()
        assert rows["base_moves"].tobytes() == single["base_moves"].tobytes()
        assert rows["move_effects"].tobytes() == single["move_effects"].tobytes()


def test_determinism(envs=8, rounds=40):
    """Two engines with identical seeds/teams/actions must match byte for byte."""
    specs = []
    for index in range(envs):
        specs.append((index, (index * 3 + 1) % 300, (index * 7 + 2) % 300,
                      ((index + 1) & 0xFFFF, (index * 2 + 3) & 0xFFFF, 7, 11),
                      (0, 1) if index % 2 else (1, 0)))
    engines = []
    handles = []
    for _ in range(2):
        engine = pa3_engine.NativeEngine(DATA, TEAMS, workers=4)
        engine_handles = engine.reset_batch(
            [spec[1] for spec in specs],
            [spec[2] for spec in specs],
            [spec[3] for spec in specs],
            [spec[4] for spec in specs],
        )
        engines.append(engine)
        handles.append(engine_handles)
    digest = hashlib.sha256()
    for _ in range(rounds):
        for env in range(envs):
            submissions = []
            for side in (0, 1):
                handle = handles[0][env]
                kind = engines[0].request_kind(handle[0], handle[1], side)
                if kind in (0, 1, 2):
                    submissions.append((handle, side))
            if not submissions:
                continue
            sides = [side for _, side in submissions]
            obs_a = engines[0].observe_encoded_batch([h for h, _ in submissions], sides)
            obs_b = engines[1].observe_encoded_batch([h for h, _ in submissions], sides)
            assert obs_a == obs_b, f"observation divergence at env {env}"
            for blob in obs_a:
                digest.update(blob)
            handle_a = submissions[0][0]
            handle_b = handles[1][env]
            sides_actions = pending(engines[0], handle_a)
            other = pending(engines[1], handle_b)
            assert sides_actions == other, "mask divergence"
            specs_a = [(handle_a[0], handle_a[1], sides_actions)]
            specs_b = [(handle_b[0], handle_b[1], other)]
            results_a = engines[0].step_batch(specs_a)
            results_b = engines[1].step_batch(specs_b)
            for result_a, result_b in zip(results_a, results_b):
                assert result_a == result_b, "step divergence"
    return digest.hexdigest()


def test_snapshot_restore(envs=4, rounds=12):
    engine_a = pa3_engine.NativeEngine(DATA, TEAMS, workers=4)
    engine_b = pa3_engine.NativeEngine(DATA, TEAMS, workers=4)
    handles_a = engine_a.reset_batch(
        [11, 12, 13, 14], [21, 22, 23, 24], [(3, 5, 7, 9)] * envs, [(0, 1)] * envs
    )
    handles_b = engine_b.reset_batch(
        [11, 12, 13, 14], [21, 22, 23, 24], [(3, 5, 7, 9)] * envs, [(0, 1)] * envs
    )
    for _ in range(rounds):
        specs = []
        for index in range(envs):
            slots = pending(engine_a, handles_a[index])
            if slots:
                specs.append((handles_a[index][0], handles_a[index][1], slots))
        if not specs:
            break
        engine_a.step_batch(specs)
    # Snapshot A into B and require identical observations afterwards.
    for index in range(envs):
        engine_b.restore(handles_b[index][0], handles_b[index][1],
                         engine_a.snapshot(handles_a[index][0], handles_a[index][1]))
    for index in range(envs):
        for side in (0, 1):
            blob_a = engine_a.observe_encoded_batch([handles_a[index]], [side])[0]
            blob_b = engine_b.observe_encoded_batch([handles_b[index]], [side])[0]
            assert blob_a == blob_b, f"restored observation mismatch env {index} side {side}"


def main():
    engine = pa3_engine.NativeEngine(DATA, TEAMS, workers=4)
    test_layout(engine)
    print("layout ok")
    test_fixed_batch(engine)
    print("fixed batch ok")
    digest = test_determinism()
    print(f"determinism ok: {digest[:16]}")
    test_snapshot_restore()
    print("snapshot/restore ok")


if __name__ == "__main__":
    main()
