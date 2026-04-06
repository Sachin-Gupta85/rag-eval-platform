"""The evaluation record for one RAG query."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RAGExample:
    question: str
    retrieved: list[str] = field(default_factory=list)   # ranked doc ids, best first
    relevant: set[str] = field(default_factory=set)       # gold-relevant doc ids
    answer: str = ""                                       # the system's answer
    contexts: list[str] = field(default_factory=list)     # context text passed to the LLM
    reference: str = ""                                    # gold / reference answer

    def has_retrieval(self) -> bool:
        return bool(self.retrieved) and bool(self.relevant)

    def has_generation(self) -> bool:
        return bool(self.answer) and bool(self.contexts)
