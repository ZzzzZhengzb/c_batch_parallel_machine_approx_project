from __future__ import annotations

import unittest

from algorithms import solve_dp_bbs_in, solve_dp_lpt_in, solve_lpt_dp_in
from config import DEFAULT_CONFIG, ValidationConfig, generate_q_types, make_config
from fixed_instance import build_fixed_instance, validate_instance
from models import Job
from single_machine_dp import solve_single_machine_dp
from validator import validate_algorithm_result


class ProjectTests(unittest.TestCase):
    def test_reproducible_distribution_generation(self) -> None:
        for distribution in ("normal", "uniform"):
            first = generate_q_types(distribution, seed=42)
            self.assertEqual(first, generate_q_types(distribution, seed=42))
            self.assertEqual(len(first), 13)
            self.assertEqual(len(set(first)), 13)
            self.assertTrue({5.0, 7.0, 9.0}.issubset(first))
            self.assertTrue(all(5.0 <= q <= 10.0 and round(q, 2) == q for q in first))

            cfg = make_config(distribution, seed=42)
            self.assertEqual(sum(cfg.quantities), 10_000)
            for q, quantity in zip(cfg.q_types, cfg.quantities):
                self.assertEqual(quantity, 3000 if q in {5.0, 7.0, 9.0} else 100)

    def test_fixed_instance_counts(self):
        cfg = DEFAULT_CONFIG
        jobs = build_fixed_instance(cfg)
        validate_instance(jobs, cfg)
        self.assertEqual(len(jobs), 10_000)
        self.assertEqual(sum(1 for j in jobs if j.q == 5.0), 3000)
        self.assertEqual(sum(1 for j in jobs if j.q == 7.0), 3000)
        self.assertEqual(sum(1 for j in jobs if j.q == 9.0), 3000)

    def test_single_machine_dp_simple(self):
        jobs = [Job(1, 1, 5.0), Job(2, 2, 5.5), Job(3, 3, 9.0)]
        sol = solve_single_machine_dp(jobs, b=5, alpha=0.2)
        # {5,5.5} can batch, 9 is incompatible with them.
        expected = 5.5 * (1 + 1/5) + 9.0
        self.assertAlmostEqual(sol.objective, expected, places=9)

    def test_three_algorithms_feasible_on_small_instance(self):
        cfg = ValidationConfig(
            n=20,
            m=4,
            b=5,
            alpha=0.2,
            beta=1.2,
            q_types=(5.0, 5.81, 6.54, 6.85, 7.0, 7.21, 7.5, 7.94, 8.12, 8.28, 8.63, 9.0, 9.43),
            quantities=(2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 6, 1),
        )
        jobs = build_fixed_instance(cfg)
        for solver in (solve_dp_lpt_in, solve_lpt_dp_in, solve_dp_bbs_in):
            result = solver(jobs, m=cfg.m, b=cfg.b, alpha=cfg.alpha)
            check = validate_algorithm_result(result, jobs, cfg)
            self.assertTrue(check["validation_passed"], msg=(solver.__name__, check))


if __name__ == "__main__":
    unittest.main()
