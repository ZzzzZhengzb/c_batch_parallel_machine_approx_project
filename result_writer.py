from __future__ import annotations

import json
from pathlib import Path

from config import ValidationConfig
from excel_writer import write_table_workbook
from models import AlgorithmResult


def write_algorithm_outputs(
    output_dir: str | Path,
    result: AlgorithmResult,
    config: ValidationConfig,
    lb: float,
    validation: dict,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = result.algorithm.lower().replace("-", "_")

    machine_rows = []
    batch_rows = []
    for machine in result.machines:
        load = machine.load(config.b)
        machine_rows.append({
            "algorithm": result.algorithm,
            "machine_id": machine.machine_id,
            "load": load,
            "num_batches": len(machine.batches),
            "num_jobs": sum(batch.size for batch in machine.batches),
        })
        for position, batch in enumerate(machine.batches, start=1):
            batch_rows.append({
                "algorithm": result.algorithm,
                "machine_id": machine.machine_id,
                "batch_position": position,
                "source_label": batch.source_label,
                "size": batch.size,
                "min_q": batch.min_q,
                "max_q": batch.max_q,
                "processing_time": batch.processing_time(config.b),
                "job_ids": " ".join(map(str, batch.job_ids)),
            })

    write_table_workbook(out / f"{slug}_machines.xlsx", machine_rows, "Machines")
    write_table_workbook(out / f"{slug}_batches.xlsx", batch_rows, "Batches")

    payload = {
        "algorithm": result.algorithm,
        "parameters": {
            "n": config.n,
            "m": config.m,
            "b": config.b,
            "alpha": config.alpha,
            "beta": config.beta,
        },
        "makespan": result.makespan,
        "LB": lb,
        "LB_gap_percent": (result.makespan - lb) / lb * 100.0,
        "runtime_seconds": result.runtime,
        "machine_loads": [m.load(config.b) for m in result.machines],
        "num_batches": sum(len(m.batches) for m in result.machines),
        "average_batch_size": config.n / max(1, sum(len(m.batches) for m in result.machines)),
        "max_batch_size": max((b.size for m in result.machines for b in m.batches), default=0),
        "metadata": result.metadata,
        "validation": validation,
    }
    with (out / f"{slug}_result.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_summary_table(path: str | Path, rows: list[dict]) -> Path:
    return write_table_workbook(Path(path).with_suffix(".xlsx"), rows, "Summary")
