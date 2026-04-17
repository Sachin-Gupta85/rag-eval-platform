from rageval import to_markdown
from rageval.dataset import RAGExample
from rageval.runner import EvalConfig, evaluate, score_example


def make_dataset():
    return [
        RAGExample(
            question="What is the capital of France?",
            retrieved=["d2", "d1", "d3"],
            relevant={"d2"},
            answer="Paris is the capital of France.",
            contexts=["Paris is the capital city of France."],
            reference="Paris is the capital of France.",
        ),
        RAGExample(
            question="Where is the Eiffel Tower?",
            retrieved=["d9", "d8", "d7"],
            relevant={"d7"},
            answer="The Eiffel Tower is located in Paris.",
            contexts=["The Eiffel Tower stands in Paris."],
            reference="The Eiffel Tower is in Paris.",
        ),
    ]


def test_score_example_has_both_halves():
    ex = make_dataset()[0]
    row = score_example(ex, EvalConfig())
    assert "recall@k" in row and "faithfulness" in row
    assert row["mrr"] == 1.0                       # relevant doc ranked first
    assert row["faithfulness"] == 1.0


def test_evaluate_aggregates_with_cis():
    report = evaluate(make_dataset(), EvalConfig(bootstrap_iters=500, seed=1))
    assert report.n == 2
    faith = report.aggregates["faithfulness"]
    assert faith.point == 1.0
    assert faith.low <= faith.point <= faith.high
    # recall is 1.0 on ex1 (rank1) and 0.0... actually d7 at rank3 within k=5 -> 1.0
    assert report.aggregates["recall@k"].point == 1.0


def test_retrieval_only_example_skips_generation_metrics():
    ex = RAGExample(question="q", retrieved=["a", "b"], relevant={"a"})
    row = score_example(ex, EvalConfig())
    assert "mrr" in row
    assert "faithfulness" not in row


def test_markdown_report_renders():
    report = evaluate(make_dataset(), EvalConfig(bootstrap_iters=200))
    md = to_markdown(report, title="demo")
    assert "# demo" in md
    assert "faithfulness" in md
    assert "95% CI" in md
