from __future__ import annotations

from collections import deque
from time import perf_counter

from models import AlgorithmResult, Batch, Job, MachineSchedule
from single_machine_dp import solve_single_machine_dp


def _least_loaded_machine(loads: list[float]) -> int:
    """Smallest load, then smallest machine index."""
    return min(range(len(loads)), key=lambda i: (loads[i], i))


def solve_dp_lpt_in(jobs: list[Job], m: int, b: int, alpha: float) -> AlgorithmResult:
    """Algorithm 2: global single-machine DP, then LPT assignment of whole batches."""
    start = perf_counter()
    dp = solve_single_machine_dp(jobs, b=b, alpha=alpha)

    indexed = list(enumerate(dp.batches))
    indexed.sort(key=lambda pair: (-pair[1].processing_time(b), pair[0]))

    machines = [MachineSchedule(machine_id=i + 1) for i in range(m)]
    loads = [0.0] * m
    for _, batch in indexed:
        i = _least_loaded_machine(loads)
        machines[i].batches.append(batch)
        loads[i] += batch.processing_time(b)

    runtime = perf_counter() - start
    return AlgorithmResult(
        algorithm="DP-LPT-IN",
        makespan=max(loads, default=0.0),
        runtime=runtime,
        machines=machines,
        metadata={
            "global_dp_objective": dp.objective,
            "global_dp_runtime": dp.runtime,
            "global_dp_batch_count": len(dp.batches),
            "global_dp_candidate_count": dp.candidate_count,
        },
    )


def solve_lpt_dp_in(jobs: list[Job], m: int, b: int, alpha: float) -> AlgorithmResult:
    """Algorithm 3: LPT assignment by q-load, then exact single-machine DP on each machine."""
    start = perf_counter()

    # Nonincreasing q; job ID gives deterministic tie-breaking.
    ordered = sorted(jobs, key=lambda job: (-job.q, job.job_id))
    subsets: list[list[Job]] = [[] for _ in range(m)]
    temporary_loads = [0.0] * m

    for job in ordered:
        i = _least_loaded_machine(temporary_loads)
        subsets[i].append(job)
        temporary_loads[i] += job.q

    machines: list[MachineSchedule] = []
    dp_times: list[float] = []
    dp_objectives: list[float] = []
    dp_candidate_counts: list[int] = []
    final_loads: list[float] = []

    for i, subset in enumerate(subsets):
        if subset:
            dp = solve_single_machine_dp(subset, b=b, alpha=alpha)
            batches = dp.batches
            load = dp.objective
            dp_times.append(dp.runtime)
            dp_objectives.append(dp.objective)
            dp_candidate_counts.append(dp.candidate_count)
        else:
            batches = []
            load = 0.0
            dp_times.append(0.0)
            dp_objectives.append(0.0)
            dp_candidate_counts.append(0)
        machines.append(MachineSchedule(machine_id=i + 1, batches=batches))
        final_loads.append(load)

    runtime = perf_counter() - start
    return AlgorithmResult(
        algorithm="LPT-DP-IN",
        makespan=max(final_loads, default=0.0),
        runtime=runtime,
        machines=machines,
        metadata={
            "temporary_q_loads": temporary_loads,
            "machine_dp_objectives": dp_objectives,
            "machine_dp_runtimes": dp_times,
            "machine_dp_candidate_counts": dp_candidate_counts,
        },
    )


def _split_from_largest_q_side(batch: Batch, required_processing: float, b: int) -> tuple[Batch | None, Batch]:
    """Return (remainder, boundary_subbatch).

    The boundary subbatch is the smallest suffix in ascending-q order (equivalently,
    the smallest prefix when the batch is read from the largest-q side) whose
    processing time is at least required_processing. If even one job suffices,
    the boundary subbatch is a singleton. The original batch is assumed to be in
    nondecreasing q order, as produced by the global DP.
    """
    if batch.size <= 0:
        raise ValueError("Cannot split an empty batch.")
    required_processing = max(0.0, float(required_processing))

    # Preserve the order inherited from the global DP.
    n = batch.size
    chosen = n
    for t in range(1, n + 1):
        start = n - t
        q_values = batch.q_values[start:]
        qmax = max(q_values)
        p = qmax * (1.0 + (t - 1) / float(b))
        if p + 1e-12 >= required_processing:
            chosen = t
            break

    cut = n - chosen
    boundary = Batch(
        job_ids=batch.job_ids[cut:],
        q_values=batch.q_values[cut:],
        source_label=f"{batch.source_label}-split-high",
    )
    remainder = None
    if cut > 0:
        remainder = Batch(
            job_ids=batch.job_ids[:cut],
            q_values=batch.q_values[:cut],
            source_label=f"{batch.source_label}-split-low",
        )
    return remainder, boundary


def solve_dp_bbs_in(jobs: list[Job], m: int, b: int, alpha: float) -> AlgorithmResult:
    """Algorithm 4: balanced batch splitting based on the global DP solution."""
    start = perf_counter()
    dp = solve_single_machine_dp(jobs, b=b, alpha=alpha)
    qmax = max(job.q for job in jobs)
    lb = max(dp.objective / float(m), qmax)
    delta = 1.0 - 1.0 / float(b)
    lam = max(0.0, ((m - 1) * delta - 1.0) / float(m))
    threshold = (1.0 + lam) * lb

    machines = [MachineSchedule(machine_id=i + 1) for i in range(m)]
    loads = [0.0] * m
    queue: deque[Batch] = deque(dp.batches)
    split_count = 0

    for i in range(m - 1):
        while queue:
            batch = queue[0]
            p = batch.processing_time(b)
            if loads[i] + p < threshold - 1e-12:
                queue.popleft()
                machines[i].batches.append(batch)
                loads[i] += p
                continue

            # Boundary batch. Take the smallest subbatch from the largest-q side
            # that reaches/crosses the threshold.
            queue.popleft()
            needed = threshold - loads[i]
            remainder, boundary = _split_from_largest_q_side(batch, needed, b)
            machines[i].batches.append(boundary)
            loads[i] += boundary.processing_time(b)
            if remainder is not None:
                split_count += 1
                queue.appendleft(remainder)
            break

    while queue:
        batch = queue.popleft()
        machines[m - 1].batches.append(batch)
        loads[m - 1] += batch.processing_time(b)

    runtime = perf_counter() - start
    return AlgorithmResult(
        algorithm="DP-BBS-IN",
        makespan=max(loads, default=0.0),
        runtime=runtime,
        machines=machines,
        metadata={
            "global_dp_objective": dp.objective,
            "global_dp_runtime": dp.runtime,
            "global_dp_batch_count": len(dp.batches),
            "global_dp_candidate_count": dp.candidate_count,
            "LB": lb,
            "delta": delta,
            "lambda": lam,
            "threshold_T": threshold,
            "split_count": split_count,
        },
    )
