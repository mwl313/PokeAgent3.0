"""D1: the fixed-step two-rank DDP protocol on a real process group.

Launches two CPU gloo ranks (``ddp_protocol_worker.py``). Each rank builds an
identical fixture, takes an uneven shard (including a truly empty shard), runs
the fixed-step protocol and compares against a single-process reference on the
same initial weights and rows.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import socket
import subprocess
import sys
import tempfile

import pytest

pytest.importorskip("torch")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


CASES = ["uneven_trajectories", "empty_rank"]


@pytest.fixture(scope="module", params=("none", "cpu"), ids=("streaming", "cpu_cache"))
def protocol_results(request):
    worker = pathlib.Path(__file__).parent / "ddp_protocol_worker.py"
    port = _free_port()
    with tempfile.TemporaryDirectory() as directory:
        out_path = pathlib.Path(directory) / "protocol.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(worker.parent),
                str(worker.resolve().parents[2]),
                str(worker.resolve().parents[2] / "engine" / "python"),
            ]
        )
        env["OMP_NUM_THREADS"] = "2"
        processes = [
            subprocess.Popen(
                [
                    sys.executable,
                    str(worker),
                    "--rank",
                    str(rank),
                    "--port",
                    str(port),
                    "--out",
                    str(out_path),
                    "--cache-device",
                    request.param,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(worker.resolve().parents[2]),
            )
            for rank in (0, 1)
        ]
        for process in processes:
            try:
                output = process.communicate(timeout=300)[0].decode()
            except subprocess.TimeoutExpired:
                process.kill()
                raise AssertionError("DDP protocol worker timed out (collective desync?)")
            assert process.returncode == 0, output[-4000:]
        assert out_path.exists(), "rank 0 did not write the protocol report"
        return json.loads(out_path.read_text())


def test_ddp_protocol_matches_single_process_reference(protocol_results):
    for case in CASES:
        ranks = [entry[case] for entry in protocol_results]
        assert ranks[0]["rows"] + ranks[1]["rows"] == ranks[0]["total_rows"]
        if case == "empty_rank":
            assert ranks[1]["rows"] == 0
        else:
            assert ranks[1]["rows"] >= 1, "the uneven case needs a non-empty small rank"
        expected_batches = math.ceil(max(entry["rows"] for entry in ranks) / 32)
        for entry in ranks:
            assert entry["optimizer_steps"] == entry["reference_optimizer_steps"]
            assert entry["optimizer_steps"] == expected_batches
            assert entry["skipped"] == 0
            assert entry["max_weight_delta"] <= 1e-6
            assert entry["max_optimizer_delta"] <= 1e-6
            assert entry["advantage_delta"] <= 1e-5
            assert abs(entry["learning_rate"] - entry["reference_learning_rate"]) < 1e-12
            assert entry["epoch_kl"] == pytest.approx(
                entry["reference_epoch_kl"], abs=1e-6
            )


def test_ranks_execute_the_same_collective_sequence(protocol_results):
    for case in CASES:
        ranks = [entry[case] for entry in protocol_results]
        assert len({entry["state_sha"] for entry in ranks}) == 1
        expected_batches = math.ceil(max(entry["rows"] for entry in ranks) / 32)
        # per_rank_minibatch 32 / microbatch 16 -> two micro steps per minibatch
        assert {entry["sync_calls"] for entry in ranks} == {expected_batches}
        assert {entry["no_sync_calls"] for entry in ranks} == {expected_batches}
        assert {entry["profile_sync_steps"] for entry in ranks} == {float(expected_batches)}
