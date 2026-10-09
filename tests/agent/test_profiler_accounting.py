"""Kernel-busy accounting must exclude double counting and profiler overhead."""
from scripts.v4_f0_profile import kernel_window


def test_kernel_intervals_merge_overlapping_streams_and_keep_idle_gaps():
    events = [{"ts": 10, "dur": 10}, {"ts": 15, "dur": 10},
              {"ts": 40, "dur": 5}, {"ts": 16, "dur": 1}]
    busy, span = kernel_window(events)
    assert busy == 20
    assert span == 35
    assert busy <= span


def test_kernel_accounting_handles_empty_trace_and_zero_duration():
    assert kernel_window([]) == (0, 0)
    assert kernel_window([{"ts": 5, "dur": 0}]) == (0, 0)
