"""Absolute committed-match checkpoint scheduling for production runs."""

from dataclasses import dataclass


@dataclass
class CommittedMatchCheckpointSchedule:
    interval: int = 100_000
    next_boundary: int | None = None

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("checkpoint interval must be positive")
        if self.next_boundary is not None and self.next_boundary < 1:
            raise ValueError("next checkpoint boundary must be positive")

    @classmethod
    def from_clock(cls, committed_matches: int, interval: int = 100_000):
        if committed_matches < 0:
            raise ValueError("committed matches cannot be negative")
        return cls(interval, ((committed_matches // interval) + 1) * interval)

    def due(self, committed_matches: int) -> bool:
        return committed_matches >= int(self.next_boundary or self.interval)

    def consume(self, committed_matches: int) -> list[int]:
        if not self.due(committed_matches):
            return []
        crossed = []
        while self.next_boundary <= committed_matches:
            crossed.append(self.next_boundary)
            self.next_boundary += self.interval
        return crossed

