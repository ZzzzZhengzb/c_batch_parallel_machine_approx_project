# Parallel-machine approximation experiment

This project implements the three approximation algorithms in Section 4 for the identical parallel-machine semi-continuous batch scheduling problem with interval processing times:

- `DP-LPT-IN`
- `LPT-DP-IN`
- `DP-BBS-IN`

It also includes the exact single-machine dynamic program used by the algorithms and by the common lower bound.

## Reproducible random validation instances

The project runs two independent experiments. Their 13 lower-processing-time types are
generated from either a truncated normal distribution or a uniform distribution on
`[5, 10]`. A fixed random seed makes each experiment reproducible, and each generated
value is rounded to two decimal places.

- `n = 10000`
- `m = 4`
- `b = 5`
- `alpha = 0.2`
- `beta = 1.2`
- distributions: truncated `Normal(7.5, 1.25^2)` and `Uniform(5, 10)`
- default random seed: `42`
- exactly 13 distinct `q` types in `[5, 10]`
- mandatory types `q = 5, 7, 9`, with 3000 jobs each
- ten other generated types, with 100 jobs each

Total: `3*3000 + 10*100 = 10000` jobs.

## Common lower bound and reported LB gap

The reference lower bound is

`LB = max(C_DP / m, q_max)`,

where `C_DP` is the exact single-machine DP objective for all 10000 jobs.

The reported experimental gap is

`LB_gap(%) = (C_ALG - LB) / LB * 100`.

This is a gap to a valid lower bound, not a Gurobi MIP optimality gap and not necessarily the true empirical approximation ratio unless the lower bound is attained.

## Deterministic tie-breaking

- DP: among numerically tied optimal transitions, choose the larger batch.
- LPT machine assignment: choose the smallest-index machine among equal loads.
- LPT job order: nonincreasing `q`, then increasing `job_id`.
- DP-LPT batch order: nonincreasing batch processing time, then original DP batch order.

## Run

```bash
python run_validation.py
```

This runs both distributions. Run only one experiment or choose another reproducible seed with:

```bash
python run_validation.py --distribution normal --seed 42
python run_validation.py --distribution uniform --seed 42
```

Results are isolated under `outputs/normal/` and `outputs/uniform/`. Generated input
data are written under `data/normal/` and `data/uniform/`.

Every tabular result is written directly as an individual formatted `.xlsx` file.
The project does not emit CSV files in `outputs/`.

## Main outputs

- `outputs/<distribution>/validation_summary.xlsx`: comparison of the three algorithms.
- `outputs/<distribution>/reference_lower_bound.json`: reference DP and LB.
- `outputs/<distribution>/*_result.json`: detailed metrics for each algorithm.
- `outputs/<distribution>/*_machines.xlsx`: per-machine summaries.
- `outputs/<distribution>/*_batches.xlsx`: batch-level schedules.
- `data/<distribution>/jobs.csv`: the generated 10000-job instance.
- `data/<distribution>/types.csv`: the generated type values and quantities.

Every algorithm result is independently reconstructed and checked for job coverage, interval compatibility, q consistency, and makespan consistency.

## Tests

```bash
python -m unittest discover -s tests -v
```

Only NumPy is required; Gurobi is not used in this approximation-algorithm project.
