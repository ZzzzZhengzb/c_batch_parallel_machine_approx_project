from __future__ import annotations

from dataclasses import dataclass
import random


MANDATORY_Q_VALUES = (5.0, 7.0, 9.0)
Q_MIN = 5.0
Q_MAX = 10.0
Q_TYPE_COUNT = 13
DEFAULT_SEED = 42


def generate_q_types(distribution: str, seed: int = DEFAULT_SEED) -> tuple[float, ...]:
    """Generate 13 reproducible, distinct q values rounded to two decimals."""
    if distribution not in {"normal", "uniform"}:
        raise ValueError("distribution must be 'normal' or 'uniform'.")

    rng = random.Random(seed)
    values = set(MANDATORY_Q_VALUES)
    while len(values) < Q_TYPE_COUNT:
        if distribution == "normal":
            candidate = rng.normalvariate(7.5, 1.25)
            if not Q_MIN <= candidate <= Q_MAX:
                continue
        else:
            candidate = rng.uniform(Q_MIN, Q_MAX)
        values.add(round(candidate, 2))
    return tuple(sorted(values))


def quantities_for(q_types: tuple[float, ...]) -> tuple[int, ...]:
    return tuple(3000 if q in MANDATORY_Q_VALUES else 100 for q in q_types)


@dataclass(frozen=True)
class ValidationConfig:
    n: int = 10_000
    m: int = 4
    b: int = 5
    alpha: float = 0.2
    beta: float = 1.2
    distribution: str = "normal"
    seed: int = DEFAULT_SEED
    q_types: tuple[float, ...] = generate_q_types("normal", DEFAULT_SEED)
    quantities: tuple[int, ...] = quantities_for(q_types)

    def validate(self) -> None:
        if len(self.q_types) != 13 or len(self.quantities) != 13:
            raise ValueError("A validation instance must have exactly 13 job types.")
        if sum(self.quantities) != self.n:
            raise ValueError("Type quantities do not sum to n.")
        if any(q < 5.0 or q > 10.0 for q in self.q_types):
            raise ValueError("All q type values must lie in [5, 10].")
        if len(set(self.q_types)) != 13:
            raise ValueError("All 13 q type values must be distinct.")
        if not set(MANDATORY_Q_VALUES).issubset(self.q_types):
            raise ValueError("q_types must contain q = 5, 7, and 9.")
        if self.n == 10_000 and self.quantities != quantities_for(self.q_types):
            raise ValueError("q = 5, 7, 9 must have 3000 jobs; all other types must have 100.")
        if any(round(q, 2) != q for q in self.q_types):
            raise ValueError("Every q type value must have at most two decimal places.")
        if self.beta != 1.0 + self.alpha:
            raise ValueError("beta must equal 1 + alpha.")
        if self.m < 2:
            raise ValueError("m must be at least 2.")
        if not (1 < self.b < self.n):
            raise ValueError("b must satisfy 1 < b < n.")


def make_config(distribution: str, seed: int = DEFAULT_SEED) -> ValidationConfig:
    q_types = generate_q_types(distribution, seed)
    return ValidationConfig(
        distribution=distribution,
        seed=seed,
        q_types=q_types,
        quantities=quantities_for(q_types),
    )


NORMAL_CONFIG = make_config("normal")
UNIFORM_CONFIG = make_config("uniform")
DEFAULT_CONFIG = NORMAL_CONFIG
