"""rag-eval-platform — measure a RAG pipeline's retrieval AND generation.

Standard IR ranking metrics (recall@k, MRR, MAP, nDCG) plus grounding metrics
(faithfulness, context recall/relevance, answer relevance), all aggregated with
bootstrap confidence intervals. Runs offline with deterministic judges.
"""
from .dataset import RAGExample
from .generation import (
    answer_relevance,
    context_recall,
    context_relevance,
    faithfulness,
    split_claims,
)
from .providers import BagOfWordsEmbedder, SupportChecker, TokenOverlapChecker, content_words
from .report import to_markdown
from .retrieval import (
    average_precision,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .runner import EvalConfig, RAGReport, evaluate, score_example
from .stats import Interval, bootstrap_ci

__all__ = [
    "RAGExample",
    "answer_relevance", "context_recall", "context_relevance", "faithfulness", "split_claims",
    "BagOfWordsEmbedder", "SupportChecker", "TokenOverlapChecker", "content_words",
    "to_markdown",
    "average_precision", "hit_at_k", "ndcg_at_k", "precision_at_k", "recall_at_k", "reciprocal_rank",
    "EvalConfig", "RAGReport", "evaluate", "score_example",
    "Interval", "bootstrap_ci",
]
