# 🔎 rag-eval-platform — evaluate retrieval *and* generation

> A RAG evaluation toolkit that scores **both halves** of a pipeline: the
> standard IR ranking metrics for retrieval and grounding metrics (faithfulness,
> context recall/relevance, answer relevance) for generation — every aggregate
> reported with a **bootstrap confidence interval**. Runs fully offline with
> deterministic judges.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Focus](https://img.shields.io/badge/focus-RAG%20evaluation-6f42c1)
![Tests](https://img.shields.io/badge/tests-pytest%20(19)-brightgreen)

Most RAG failures hide in the seam: retrieval finds the right document and the
model **still** hallucinates, or the answer is grounded but retrieval missed the
evidence the gold answer needs. A single accuracy number can't see either —
so this scores the two sides separately.

## 📐 Metrics (all implemented from their definitions)
**Retrieval** (`retrieval`) — ranked ids vs. a gold set:
`precision@k` · `recall@k` · `hit@k` · `reciprocal_rank` (MRR) ·
`average_precision` (MAP) · `ndcg@k`

**Generation** (`generation`) — grounding, via a swappable `SupportChecker`:
| Metric | Question it answers |
|---|---|
| `faithfulness` | are the answer's claims supported by the contexts? (hallucination) |
| `context_recall` | do the contexts cover what the reference answer needs? |
| `context_relevance` | are the retrieved contexts actually about the question? |
| `answer_relevance` | does the answer address the question asked? |

## 🧪 Rigour
- **Bootstrap CIs on every aggregate** (`stats.bootstrap_ci`) — you report
  `faithfulness 0.82 [0.74, 0.90]`, not a point estimate that hides noise.
- **Judges are injectable.** Grounding asks "does this source support this
  claim?" — an LLM/NLI call in production, a deterministic token-overlap checker
  here. The metrics never change; only the judge does.

## 🚀 Run
```bash
python -m rageval        # offline demo: 3 queries incl. a planted hallucination
```

## ✅ Test
```bash
pip install -r requirements-dev.txt && python -m pytest -q
```

## Why it's Level 5
Retrieval metrics are undergraduate IR; the staff-level part is scoring
*grounding* — separating a retrieval miss from a generation hallucination — and
reporting it with uncertainty so a dashboard number is trustworthy. It's the
companion to [`llm-eval-platform`](../llm-eval-platform): that one proves a
difference is real; this one tells you *which half of the pipeline* to fix.

---
MIT licensed. Part of a portfolio of runnable systems projects.
