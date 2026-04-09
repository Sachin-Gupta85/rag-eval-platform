"""Evaluate a RAG dataset end to end and aggregate with confidence intervals.

Each example yields a bag of per-metric scores; the runner averages each metric
across the dataset and attaches a bootstrap CI, so you report "faithfulness
0.82 [0.74, 0.90]" rather than a bare number that hides sample noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import generation as gen
from . import retrieval as ret
from .dataset import RAGExample
from .providers import Embedder, SupportChecker
from .stats import Interval, bootstrap_ci


@dataclass
class EvalConfig:
    k: int = 5
    checker: SupportChecker | None = None
    embedder: Embedder | None = None
    bootstrap_iters: int = 2000
    seed: int = 0


@dataclass
class RAGReport:
    aggregates: dict[str, Interval]
    per_example: list[dict[str, float]] = field(default_factory=list)
    n: int = 0


def score_example(ex: RAGExample, cfg: EvalConfig) -> dict[str, float]:
    out: dict[str, float] = {}
    if ex.has_retrieval():
        out["precision@k"] = ret.precision_at_k(ex.retrieved, ex.relevant, cfg.k)
        out["recall@k"] = ret.recall_at_k(ex.retrieved, ex.relevant, cfg.k)
        out["hit@k"] = ret.hit_at_k(ex.retrieved, ex.relevant, cfg.k)
        out["mrr"] = ret.reciprocal_rank(ex.retrieved, ex.relevant)
        out["map"] = ret.average_precision(ex.retrieved, ex.relevant)
        out["ndcg@k"] = ret.ndcg_at_k(ex.retrieved, ex.relevant, cfg.k)
    if ex.has_generation():
        out["faithfulness"] = gen.faithfulness(ex.answer, ex.contexts, cfg.checker)
        out["context_relevance"] = gen.context_relevance(ex.question, ex.contexts, cfg.checker)
        out["answer_relevance"] = gen.answer_relevance(ex.question, ex.answer, cfg.embedder)
        if ex.reference:
            out["context_recall"] = gen.context_recall(ex.reference, ex.contexts, cfg.checker)
    return out


def evaluate(dataset: list[RAGExample], cfg: EvalConfig | None = None) -> RAGReport:
    cfg = cfg or EvalConfig()
    per_example = [score_example(ex, cfg) for ex in dataset]

    # Collect each metric's values across examples that produced it.
    columns: dict[str, list[float]] = {}
    for row in per_example:
        for metric, value in row.items():
            columns.setdefault(metric, []).append(value)

    aggregates = {
        metric: bootstrap_ci(values, iters=cfg.bootstrap_iters, seed=cfg.seed)
        for metric, values in columns.items()
    }
    return RAGReport(aggregates=aggregates, per_example=per_example, n=len(dataset))
