import pytest

from rageval import generation as g
from rageval.providers import TokenOverlapChecker

CONTEXTS = [
    "Paris is the capital city of France.",
    "The Eiffel Tower stands in Paris.",
]


def test_split_claims():
    claims = g.split_claims("Paris is the capital. The tower is tall! Is it?")
    assert len(claims) == 3


def test_faithful_answer_scores_one():
    answer = "Paris is the capital of France. The Eiffel Tower is in Paris."
    assert g.faithfulness(answer, CONTEXTS) == pytest.approx(1.0)


def test_hallucinated_answer_scores_zero():
    answer = "The moon is made of green cheese."
    assert g.faithfulness(answer, CONTEXTS) == 0.0


def test_partially_grounded_answer():
    answer = "Paris is the capital of France. The moon is made of cheese."
    assert g.faithfulness(answer, CONTEXTS) == pytest.approx(0.5)


def test_context_recall_tracks_reference_coverage():
    reference = "Paris is the capital of France."
    assert g.context_recall(reference, CONTEXTS) == pytest.approx(1.0)
    missing = "The population of Paris is two million people."
    assert g.context_recall(missing, CONTEXTS) == 0.0


def test_context_relevance_mixes_good_and_bad():
    question = "What is the capital of France?"
    mixed = [CONTEXTS[0], "Bananas are a yellow tropical fruit."]
    assert g.context_relevance(question, mixed) == pytest.approx(0.5)


def test_answer_relevance_is_higher_for_on_topic_answer():
    q = "What is the capital of France?"
    on = g.answer_relevance(q, "The capital of France is Paris.")
    off = g.answer_relevance(q, "I enjoy playing tennis on weekends.")
    assert on > off


def test_checker_threshold_is_respected():
    strict = TokenOverlapChecker(threshold=0.95)
    # only 2 of 3 content words present -> below 0.95
    assert not strict.supports("alpha beta gamma", "alpha beta only")
