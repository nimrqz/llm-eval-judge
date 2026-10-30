import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm_judge.judge import (  # noqa: E402
    CriterionScore, JudgeError, Pairwise, Rubric, Scores, compare_pair, score_response,
)
from llm_judge.report import render_html  # noqa: E402
from llm_judge.runner import bootstrap_ci, evaluate, load_cases  # noqa: E402

RUBRIC = Rubric.from_dict(json.loads((ROOT / "examples" / "rubric.json").read_text(encoding="utf-8")))

class FakeJudge:
    """Dá nota alta a respostas que contenham 'BOA' e baixa às demais; veredicto pareado por 'BOA'."""

    def __init__(self, position_bias=False):
        self.position_bias = position_bias
        self.messages = NS(parse=self._parse)

    def _parse(self, output_format, messages, **_):
        text = messages[0]["content"]
        if output_format is Scores:
            good = "BOA" in text.split("<resposta>")[1]
            return NS(parsed_output=Scores(scores=[
                CriterionScore(name=c.name, score=5 if good else 2, reason="ok") for c in RUBRIC.criteria]))
        if self.position_bias:
            return NS(parsed_output=Pairwise(reason="sempre A", winner="A"))
        a = text.split("<resposta_A>")[1].split("</resposta_A>")[0]
        return NS(parsed_output=Pairwise(reason="x", winner="A" if "BOA" in a else "B"))

CASES = [
    {"id": "c1", "prompt": "p1", "outputs": {"v1": "ruim", "v2": "BOA resposta"}},
    {"id": "c2", "prompt": "p2", "outputs": {"v1": "ruim", "v2": "BOA resposta"}},
]

def test_weighted_overall_score():
    result = score_response(FakeJudge(), RUBRIC, "p", "BOA")
    assert result["overall"] == 5 and set(result["criteria"]) == {c.name for c in RUBRIC.criteria}

def test_missing_criteria_raises():
    class Partial(FakeJudge):
        def _parse(self, **kw):
            return NS(parsed_output=Scores(scores=[CriterionScore(name="correção", score=5, reason="x")]))

    with pytest.raises(JudgeError):
        score_response(Partial(), RUBRIC, "p", "r")

def test_scores_are_clamped():
    class Wild(FakeJudge):
        def _parse(self, **kw):
            return NS(parsed_output=Scores(scores=[
                CriterionScore(name=c.name, score=99, reason="x") for c in RUBRIC.criteria]))

    assert score_response(Wild(), RUBRIC, "p", "r")["overall"] == 5

def test_pairwise_detects_position_bias():
    assert compare_pair(FakeJudge(position_bias=True), RUBRIC, "p", "x", "y") == {"winner": "tie", "consistent": False}

def test_pairwise_consistent_winner():
    out = compare_pair(FakeJudge(), RUBRIC, "p", "ruim", "BOA resposta")
    assert out == {"winner": "B", "consistent": True}

def test_evaluate_ranks_variants_and_reports():
    result = evaluate(FakeJudge(), RUBRIC, CASES, workers=2)
    assert result["summary"]["v2"]["overall"] > result["summary"]["v1"]["overall"]
    pair = result["pairwise"][0]
    assert (pair["a"], pair["b_wins"], pair["position_bias_rate"]) == ("v1", 2, 0.0)
    html = render_html(result)
    assert "v2" in html and "IC95%" in html

def test_bootstrap_ci_contains_mean_and_is_deterministic():
    values = [3.0, 4.0, 5.0, 4.0, 2.0]
    low, high = bootstrap_ci(values)
    assert low <= sum(values) / len(values) <= high
    assert bootstrap_ci(values) == (low, high)

def test_load_example_cases():
    assert len(load_cases(ROOT / "examples" / "cases.jsonl")) == 5
