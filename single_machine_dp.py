from __future__ import annotations

from time import perf_counter

import numpy as np

from models import Batch, DPSolution, Job


def _candidate_ends(q_sorted: np.ndarray, theta: float) -> np.ndarray:
    n = q_sorted.size
    r = np.empty(n, dtype=np.int64)
    right = -1
    for left in range(n):
        if right < left:
            right = left
        limit = theta * float(q_sorted[left])
        while right + 1 < n and float(q_sorted[right + 1]) <= limit + 1e-12:
            right += 1
        r[left] = right
    return r


def _select_last_near_minimum(values: np.ndarray, rel_tol: float = 1e-10) -> tuple[int, float]:
    """Deterministic tie rule: among near-equal minima, choose the largest batch."""
    minimum = float(np.min(values))
    tol = rel_tol * max(1.0, abs(minimum))
    close = values <= minimum + tol
    reverse_pos = int(np.argmax(close[::-1]))
    idx = values.size - 1 - reverse_pos
    return idx, float(values[idx])


def solve_single_machine_dp(jobs: list[Job], b: int, alpha: float) -> DPSolution:
    """Exact O(n^2) DP from Section 3, with NumPy-vectorized transitions."""
    if not jobs:
        return DPSolution(objective=0.0, batches=[], runtime=0.0, candidate_count=0, theta=min(1 + alpha, b))
    if b <= 1:
        raise ValueError("b must be greater than 1.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    start = perf_counter()
    beta = 1.0 + float(alpha)
    theta = min(beta, float(b))

    # Stable order by q then original ID. This makes all ties reproducible.
    jobs_sorted = sorted(jobs, key=lambda job: (job.q, job.job_id))
    q = np.asarray([job.q for job in jobs_sorted], dtype=np.float64)
    n = q.size
    r = _candidate_ends(q, theta)

    f = np.full(n + 1, np.inf, dtype=np.float64)
    f[n] = 0.0
    nxt = np.full(n, -1, dtype=np.int64)
    indices = np.arange(n, dtype=np.float64)
    candidate_count = 0

    for k in range(n - 1, -1, -1):
        end = int(r[k])
        candidate_count += end - k + 1
        q_slice = q[k : end + 1]
        offsets = indices[k : end + 1] - float(k)
        future = f[k + 1 : end + 2]
        values = q_slice * (1.0 + offsets / float(b)) + future
        rel, value = _select_last_near_minimum(values)
        f[k] = value
        nxt[k] = k + rel + 1

    batches: list[Batch] = []
    k = 0
    batch_no = 1
    while k < n:
        next_k = int(nxt[k])
        if not (k < next_k <= n):
            raise RuntimeError(f"Invalid DP backtracking pointer at k={k}: {next_k}")
        block = jobs_sorted[k:next_k]
        batch = Batch(
            job_ids=[job.job_id for job in block],
            q_values=[job.q for job in block],
            source_label=f"DP-B{batch_no}",
        )
        if batch.max_q > beta * batch.min_q + 1e-9:
            raise RuntimeError("DP recovered an interval-infeasible batch.")
        if batch.max_q > theta * batch.min_q + 1e-9:
            raise RuntimeError("DP recovered a batch outside the candidate window.")
        batches.append(batch)
        k = next_k
        batch_no += 1

    recovered = sum(batch.processing_time(b) for batch in batches)
    objective = float(f[0])
    if abs(recovered - objective) > 1e-8 * max(1.0, abs(objective)):
        raise RuntimeError(f"DP objective mismatch: {objective} vs reconstructed {recovered}")

    return DPSolution(
        objective=objective,
        batches=batches,
        runtime=perf_counter() - start,
        candidate_count=int(candidate_count),
        theta=float(theta),
    )
