from __future__ import annotations

from collections import Counter

from config import ValidationConfig
from models import AlgorithmResult, Job


def validate_algorithm_result(
    result: AlgorithmResult,
    jobs: list[Job],
    config: ValidationConfig,
    tol: float = 1e-8,
) -> dict[str, float | int | bool | str]:
    by_id = {job.job_id: job for job in jobs}
    seen: list[int] = []
    interval_violations = 0
    q_mismatches = 0
    empty_batches = 0
    machine_loads: list[float] = []

    for machine in result.machines:
        load = 0.0
        for batch in machine.batches:
            if batch.size == 0:
                empty_batches += 1
                continue
            if len(batch.job_ids) != len(batch.q_values):
                raise AssertionError("Batch job_ids and q_values have different lengths.")
            seen.extend(batch.job_ids)
            for job_id, q in zip(batch.job_ids, batch.q_values):
                source = by_id.get(job_id)
                if source is None or abs(source.q - q) > tol:
                    q_mismatches += 1
            if batch.max_q > config.beta * batch.min_q + tol:
                interval_violations += 1
            load += batch.processing_time(config.b)
        machine_loads.append(load)

    expected_ids = [job.job_id for job in jobs]
    seen_counter = Counter(seen)
    expected_counter = Counter(expected_ids)
    assignment_ok = seen_counter == expected_counter
    reconstructed_makespan = max(machine_loads, default=0.0)
    objective_ok = abs(reconstructed_makespan - result.makespan) <= tol * max(1.0, abs(result.makespan))

    return {
        "validation_passed": bool(
            assignment_ok
            and interval_violations == 0
            and q_mismatches == 0
            and empty_batches == 0
            and objective_ok
        ),
        "assignment_ok": assignment_ok,
        "interval_violations": interval_violations,
        "q_mismatches": q_mismatches,
        "empty_batches": empty_batches,
        "objective_ok": objective_ok,
        "reconstructed_makespan": reconstructed_makespan,
        "reported_makespan": result.makespan,
        "max_machine_load": reconstructed_makespan,
        "min_machine_load": min(machine_loads, default=0.0),
        "total_batches": sum(len(machine.batches) for machine in result.machines),
        "total_jobs_seen": len(seen),
    }
