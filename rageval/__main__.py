"""Offline demo — evaluate a tiny RAG dataset and print a Markdown report.

    python -m rageval
"""
from __future__ import annotations

from .dataset import RAGExample
from .report import to_markdown
from .runner import EvalConfig, evaluate


def main() -> None:
    dataset = [
        RAGExample(
            question="What is the capital of France?",
            retrieved=["doc_france", "doc_italy", "doc_spain"],
            relevant={"doc_france"},
            answer="The capital of France is Paris.",
            contexts=["Paris is the capital and most populous city of France."],
            reference="Paris is the capital of France.",
        ),
        RAGExample(
            question="Who wrote Hamlet?",
            retrieved=["doc_dickens", "doc_shakespeare", "doc_austen"],
            relevant={"doc_shakespeare"},
            # a grounded but imperfectly-ranked case: right doc at rank 2
            answer="Hamlet was written by William Shakespeare.",
            contexts=["William Shakespeare wrote the tragedy Hamlet around 1600."],
            reference="William Shakespeare wrote Hamlet.",
        ),
        RAGExample(
            question="What is the boiling point of water?",
            retrieved=["doc_chem", "doc_bio", "doc_phys"],
            relevant={"doc_chem"},
            # retrieval is perfect, but the answer wanders off the context entirely
            answer="Water is a shiny metal that is mined deep underground.",
            contexts=["Water boils at 100 degrees Celsius at sea level pressure."],
            reference="Water boils at 100 degrees Celsius.",
        ),
    ]

    report = evaluate(dataset, EvalConfig(k=3, bootstrap_iters=2000, seed=0))
    print(to_markdown(report, title="RAG demo (3 queries)"))
    print()
    print("Note how query 3's hallucination drags faithfulness below 1.0 even")
    print("though retrieval found the right document (mrr stays high) — the two")
    print("halves are scored independently, which is the whole point.")
    print("(The token-overlap judge is lexical; swap in an NLI/LLM judge to also")
    print(" catch numeric contradictions like '50°' vs '100°'.)")


if __name__ == "__main__":
    main()
