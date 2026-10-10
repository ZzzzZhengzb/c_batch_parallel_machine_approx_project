from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from algorithms import solve_dp_bbs_in, solve_dp_lpt_in, solve_lpt_dp_in
from config import DEFAULT_SEED, ValidationConfig, make_config
from fixed_instance import (
    build_fixed_instance,
    validate_instance,
    write_fixed_instance_csv,
    write_type_summary_csv,
)
from result_writer import write_algorithm_outputs, write_summary_table
from single_machine_dp import solve_single_machine_dp
from validator import validate_algorithm_result


def run_experiment(config: ValidationConfig, output_dir: Path) -> None:
    config.validate()
    jobs = build_fixed_instance(config)
    validate_instance(jobs, config)

    data_dir = output_dir.parent.parent / "data" / config.distribution
    write_fixed_instance_csv(data_dir / "jobs.csv", config)
    write_type_summary_csv(data_dir / "types.csv", config)

    # Reference lower bound. This calculation is deliberately performed outside
    # each algorithm's runtime. Algorithms that require the global DP compute it
    # again internally, so their reported runtime remains faithful to their steps.
    ref_start = perf_counter()
    reference_dp = solve_single_machine_dp(jobs, b=config.b, alpha=config.alpha)
    reference_time = perf_counter() - ref_start
    qmax = max(job.q for job in jobs)
    lb = max(reference_dp.objective / float(config.m), qmax)

    algorithms = [solve_dp_lpt_in, solve_lpt_dp_in, solve_dp_bbs_in]
    summary_rows: list[dict] = []

    for solver in algorithms:
        result = solver(jobs, m=config.m, b=config.b, alpha=config.alpha)
        validation = validate_algorithm_result(result, jobs, config)
        write_algorithm_outputs(output_dir, result, config, lb, validation)

        machine_loads = [machine.load(config.b) for machine in result.machines]
        total_batches = sum(len(machine.batches) for machine in result.machines)
        batch_sizes = [batch.size for machine in result.machines for batch in machine.batches]
        summary_rows.append({
            "algorithm": result.algorithm,
            "n": config.n,
            "m": config.m,
            "b": config.b,
            "alpha": config.alpha,
            "beta": config.beta,
            "makespan": result.makespan,
            "LB": lb,
            "LB_gap_percent": (result.makespan - lb) / lb * 100.0,
            "runtime_seconds": result.runtime,
            "machine_1_load": machine_loads[0],
            "machine_2_load": machine_loads[1],
            "machine_3_load": machine_loads[2],
            "machine_4_load": machine_loads[3],
            "total_batches": total_batches,
            "average_batch_size": config.n / total_batches,
            "max_batch_size": max(batch_sizes, default=0),
            "split_count": result.metadata.get("split_count", ""),
            "validation_passed": validation["validation_passed"],
        })

    summary_path = write_summary_table(output_dir / "validation_summary.xlsx", summary_rows)
    reference_payload = {
        "distribution": config.distribution,
        "seed": config.seed,
        "n": config.n,
        "m": config.m,
        "b": config.b,
        "alpha": config.alpha,
        "beta": config.beta,
        "single_machine_DP": reference_dp.objective,
        "q_max": qmax,
        "LB": lb,
        "reference_dp_runtime_seconds": reference_time,
        "reference_dp_num_batches": len(reference_dp.batches),
        "reference_dp_candidate_count": reference_dp.candidate_count,
        "note": "Reference-LB runtime is not included in any approximation algorithm runtime.",
    }
    with (output_dir / "reference_lower_bound.json").open("w", encoding="utf-8") as f:
        json.dump(reference_payload, f, ensure_ascii=False, indent=2)

    (output_dir / f"{config.distribution}_results.xlsx").unlink(missing_ok=True)

    print(f"{config.distribution.capitalize()} validation instance completed (seed={config.seed}).")
    print(f"Reference single-machine DP = {reference_dp.objective:.6f}")
    print(f"LB = max(C_DP/m, q_max) = {lb:.6f}")
    print(f"Summary workbook = {summary_path}")
    for row in summary_rows:
        print(
            f"{row['algorithm']:11s}  Cmax={row['makespan']:.6f}  "
            f"LB-gap={row['LB_gap_percent']:.6f}%  "
            f"time={row['runtime_seconds']:.6f}s  "
            f"valid={row['validation_passed']}"
        )


def run(output_root: Path, distributions: list[str], seed: int) -> None:
    legacy_csv_names = (
        "dp_bbs_in_batches.csv", "dp_bbs_in_machines.csv",
        "dp_lpt_in_batches.csv", "dp_lpt_in_machines.csv",
        "lpt_dp_in_batches.csv", "lpt_dp_in_machines.csv",
        "validation_summary.csv",
    )
    for name in legacy_csv_names:
        (output_root / name).unlink(missing_ok=True)
    for distribution in distributions:
        run_experiment(make_config(distribution, seed), output_root / distribution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs", help="Root directory for result files.")
    parser.add_argument(
        "--distribution",
        choices=("all", "normal", "uniform"),
        default="all",
        help="Run both experiments or only one distribution (default: all).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42).")
    args = parser.parse_args()
    selected = ["normal", "uniform"] if args.distribution == "all" else [args.distribution]
    run(Path(args.output_dir), selected, args.seed)
