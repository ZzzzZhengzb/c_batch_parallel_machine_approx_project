from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from config import DEFAULT_CONFIG, ValidationConfig
from models import Job


def build_fixed_instance(config: ValidationConfig = DEFAULT_CONFIG) -> list[Job]:
    config.validate()
    jobs: list[Job] = []
    job_id = 1
    for type_id, (q, quantity) in enumerate(zip(config.q_types, config.quantities), start=1):
        for _ in range(quantity):
            jobs.append(Job(job_id=job_id, type_id=type_id, q=float(q)))
            job_id += 1
    if len(jobs) != config.n:
        raise RuntimeError("Instance generation produced the wrong number of jobs.")
    return jobs


def write_fixed_instance_csv(path: str | Path, config: ValidationConfig = DEFAULT_CONFIG) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    jobs = build_fixed_instance(config)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["job_id", "type_id", "q", "upper_bound"])
        writer.writeheader()
        for job in jobs:
            writer.writerow({
                "job_id": job.job_id,
                "type_id": job.type_id,
                "q": f"{job.q:.2f}",
                "upper_bound": f"{config.beta * job.q:.6f}",
            })
    return path


def write_type_summary_csv(path: str | Path, config: ValidationConfig = DEFAULT_CONFIG) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["type_id", "q", "quantity", "upper_bound"])
        writer.writeheader()
        for type_id, (q, quantity) in enumerate(zip(config.q_types, config.quantities), start=1):
            writer.writerow({
                "type_id": type_id,
                "q": f"{q:.2f}",
                "quantity": quantity,
                "upper_bound": f"{config.beta * q:.6f}",
            })
    return path


def validate_instance(jobs: list[Job], config: ValidationConfig = DEFAULT_CONFIG) -> None:
    config.validate()
    if len(jobs) != config.n:
        raise AssertionError(f"Expected {config.n} jobs, got {len(jobs)}.")
    ids = [job.job_id for job in jobs]
    if sorted(ids) != list(range(1, config.n + 1)):
        raise AssertionError("Job IDs must be exactly 1..n.")
    if any(not (5.0 <= job.q <= 10.0) for job in jobs):
        raise AssertionError("A q value lies outside [5, 10].")
    counts = Counter(round(job.q, 2) for job in jobs)
    expected = Counter({round(q, 2): qty for q, qty in zip(config.q_types, config.quantities)})
    if counts != expected:
        raise AssertionError("Job-type counts do not match the experiment specification.")
