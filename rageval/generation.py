"""Grounding metrics for the *generation* half of a RAG system.

These catch the failure retrieval metrics can't see: the model retrieved good
context but then hallucinated, ignored it, or answered a different question.

  faithfulness      fraction of the answer's claims supported by the contexts
  context_recall    fraction of the reference answer's claims the contexts cover
  context_relevance fraction of retrieved contexts relevant to the question
  answer_relevance  similarity of the answer to the question
"""
from __future__ import annotations

import re

from .providers import (
    BagOfWordsEmbedder,
    Embedder,
    SupportChecker,
    TokenOverlapChecker,
    cosine,
)

_SENT = re.compile(r"[^.!?]+[.!?]?")


def split_claims(text: str) -> list[str]:
    """Split a block of text into claim-sized sentences."""
    return [s.strip() for s in _SENT.findall(text) if s.strip()]


def faithfulness(answer: str, contexts: list[str],
                 checker: SupportChecker | None = None) -> float:
    """Share of answer sentences entailed by the retrieved contexts.

    A low score means the answer asserts things the context doesn't back — the
    classic RAG hallucination signal.
    """
    checker = checker or TokenOverlapChecker()
    claims = split_claims(answer)
    if not claims:
        return 0.0
    source = "\n".join(contexts)
    supported = sum(1 for c in claims if checker.supports(c, source))
    return supported / len(claims)


def context_recall(reference: str, contexts: list[str],
                   checker: SupportChecker | None = None) -> float:
    """Share of the reference answer's claims that the contexts can support.

    A low score means retrieval missed evidence the correct answer needs.
    """
    checker = checker or TokenOverlapChecker()
    claims = split_claims(reference)
    if not claims:
        return 0.0
    source = "\n".join(contexts)
    covered = sum(1 for c in claims if checker.supports(c, source))
    return covered / len(claims)


def context_relevance(question: str, contexts: list[str],
                      checker: SupportChecker | None = None) -> float:
    """Share of retrieved contexts that are actually about the question —
    measures retrieval precision in semantic rather than id-match terms."""
    checker = checker or TokenOverlapChecker(threshold=0.3)
    if not contexts:
        return 0.0
    relevant = sum(1 for ctx in contexts if checker.supports(question, ctx))
    return relevant / len(contexts)


def answer_relevance(question: str, answer: str,
                     embedder: Embedder | None = None) -> float:
    """Cosine similarity between the answer and the question — a proxy for
    whether the answer actually addresses what was asked."""
    embedder = embedder or BagOfWordsEmbedder()
    return cosine(embedder.embed(question), embedder.embed(answer))
