"""Pluggable judges for the generation metrics.

Grounding metrics ask a yes/no question — "does this source support this claim?"
In production that's an LLM or an NLI model; here a deterministic token-overlap
checker stands in, so the whole suite runs offline. Swap in a real
``SupportChecker`` without touching the metrics.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of to in on at for and or is are was were be been it this that "
    "with as by from what which who whom how".split()
)


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


class SupportChecker(Protocol):
    def supports(self, claim: str, source: str) -> bool:
        ...


class TokenOverlapChecker:
    """Supports a claim if a high-enough fraction of its content words appear
    in the source. A cheap, deterministic stand-in for an NLI/LLM judge."""

    def __init__(self, threshold: float = 0.6) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold

    def supports(self, claim: str, source: str) -> bool:
        cw = content_words(claim)
        if not cw:
            return True                       # nothing asserted -> vacuously supported
        src = content_words(source)
        covered = len(cw & src) / len(cw)
        return covered >= self.threshold


class Embedder(Protocol):
    def embed(self, text: str) -> dict[str, float]:
        ...


class BagOfWordsEmbedder:
    def embed(self, text: str) -> dict[str, float]:
        counts = Counter(_WORD.findall(text.lower()))
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        return {w: c / norm for w, c in counts.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    return sum(a[k] * b[k] for k in a.keys() & b.keys())
