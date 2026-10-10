from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Job:
    job_id: int
    type_id: int
    q: float


@dataclass
class Batch:
    job_ids: list[int]
    q_values: list[float]
    source_label: str = ""

    @property
    def size(self) -> int:
        return len(self.job_ids)

    @property
    def min_q(self) -> float:
        return min(self.q_values)

    @property
    def max_q(self) -> float:
        return max(self.q_values)

    def processing_time(self, b: int) -> float:
        return self.max_q * (1.0 + (self.size - 1) / float(b))


@dataclass
class DPSolution:
    objective: float
    batches: list[Batch]
    runtime: float
    candidate_count: int
    theta: float


@dataclass
class MachineSchedule:
    machine_id: int
    batches: list[Batch] = field(default_factory=list)

    def load(self, b: int) -> float:
        return sum(batch.processing_time(b) for batch in self.batches)


@dataclass
class AlgorithmResult:
    algorithm: str
    makespan: float
    runtime: float
    machines: list[MachineSchedule]
    metadata: dict[str, Any] = field(default_factory=dict)
