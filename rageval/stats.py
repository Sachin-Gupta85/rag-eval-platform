"""Bootstrap confidence intervals so aggregate RAG scores carry uncertainty."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def as_dict(self) -> dict:
        return {"point": round(self.point, 4), "low": round(self.low, 4), "high": round(self.high, 4)}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    idx = q * (len(sorted_xs) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (idx - lo)


def bootstrap_ci(data: list[float], confidence: float = 0.95,
                 iters: int = 2000, seed: int = 0) -> Interval:
    """Percentile bootstrap CI for the mean of ``data``."""
    if not data:
        return Interval(0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(data)
    means = []
    for _ in range(iters):
        means.append(sum(data[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    alpha = (1 - confidence) / 2
    return Interval(mean(data), _percentile(means, alpha), _percentile(means, 1 - alpha))
