import pytest

from agent.ppo.checkpoint_schedule import CommittedMatchCheckpointSchedule


def test_schedule_uses_absolute_clock_and_catches_up_once():
    schedule = CommittedMatchCheckpointSchedule.from_clock(99_000)
    assert not schedule.due(99_999)
    assert schedule.consume(100_001) == [100_000]
    assert schedule.consume(100_001) == []
    assert schedule.consume(300_000) == [200_000, 300_000]


def test_schedule_resume_does_not_restart_from_zero():
    schedule = CommittedMatchCheckpointSchedule.from_clock(250_000)
    assert schedule.consume(299_999) == []
    assert schedule.consume(300_000) == [300_000]


def test_schedule_rejects_invalid_interval():
    with pytest.raises(ValueError):
        CommittedMatchCheckpointSchedule(0)
