"""Ranking metrics for the *retrieval* half of a RAG system.

Each function takes a ranked list of retrieved document ids (best first) and the
set of ids that are actually relevant (the gold set), and returns a score in
[0, 1]. These are the standard IR metrics — implemented from the definitions so
the maths is auditable, not borrowed from a black box.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _topk(retrieved: Sequence[str], k: int) -> list[str]:
    if k < 1:
        raise ValueError("k must be >= 1")
    return list(retrieved[:k])


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    top = _topk(retrieved, k)
    if not top:
        return 0.0
    hits = sum(1 for d in top if d in rel)
    return hits / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    top = _topk(retrieved, k)
    hits = sum(1 for d in top if d in rel)
    return hits / len(rel)


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    return 1.0 if any(d in rel for d in _topk(retrieved, k)) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant document (0 if none retrieved)."""
    rel = set(relevant)
    for i, d in enumerate(retrieved, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Average precision — the MAP component for one query.

    Mean of precision@k taken at each rank k where a relevant item appears,
    normalised by the total number of relevant items.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = 0
    running = 0.0
    for i, d in enumerate(retrieved, start=1):
        if d in rel:
            hits += 1
            running += hits / i          # precision@i at this relevant position
    return running / len(rel)


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    rel = set(relevant)
    top = _topk(retrieved, k)

    def dcg(gains: list[int]) -> float:
        return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))

    actual = dcg([1 if d in rel else 0 for d in top])
    ideal = dcg([1] * min(len(rel), k))
    return actual / ideal if ideal > 0 else 0.0
