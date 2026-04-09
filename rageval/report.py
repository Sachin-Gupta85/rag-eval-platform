"""Render a RAGReport as a Markdown table."""
from __future__ import annotations

from .runner import RAGReport

_ORDER = ["precision@k", "recall@k", "hit@k", "mrr", "map", "ndcg@k",
          "faithfulness", "context_relevance", "context_recall", "answer_relevance"]


def to_markdown(report: RAGReport, title: str = "RAG evaluation") -> str:
    lines = [f"# {title}", "", f"_n = {report.n} examples; 95% bootstrap CIs._", "",
             "| Metric | Score | 95% CI |", "|---|---|---|"]
    ordered = [m for m in _ORDER if m in report.aggregates]
    ordered += [m for m in report.aggregates if m not in _ORDER]
    for metric in ordered:
        iv = report.aggregates[metric]
        lines.append(f"| {metric} | {iv.point:.3f} | [{iv.low:.3f}, {iv.high:.3f}] |")
    return "\n".join(lines)
