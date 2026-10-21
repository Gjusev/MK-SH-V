import importlib.util
import sys
from pathlib import Path

from mksh_v.train_compare import case_of_row

_spec = importlib.util.spec_from_file_location(
    "e39r2_comparators", Path(__file__).resolve().parents[1] / "kaggle" / "e39r2_comparators.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["e39r2_comparators"] = R
_spec.loader.exec_module(R)


def _row(option_ids, target, **kw):
    return {"id": "r1", "state": "s", "question": "q?", "target_option": target,
            "options": [{"id": o, "description": f"desc {o}"} for o in option_ids], **kw}


def _decide(questions):
    """jev decide() contract (modeling_jevvlm.py @8c56e46a): noul -> {probability};
    choice -> {choice, confidence, probabilities keyed by the criteria keys}."""
    q = questions["q"]
    if q["type"] == "noul":
        return {"q": {"type": "noul", "probability": 0.8}}
    keys = list(q["criteria"])
    probs = {k: (0.7 if i == 1 else 0.3 / (len(keys) - 1)) for i, k in enumerate(keys)}
    return {"q": {"type": "choice", "choice": keys[1], "probabilities": probs}}


def test_noul_cases_use_noul_and_align_with_option_order():
    for ids in (["false", "true"], ["true", "false"]):
        case = case_of_row(_row(ids, "true"))
        q, kind = R.jev_question(case)
        assert kind == "noul" and q == {"type": "noul", "instructions": "q?"}
        probs = R.jev_probs(case, _decide({"q": q})["q"], kind)
        assert probs[case.option_ids.index("true")] == 0.8
        assert abs(sum(probs) - 1) < 1e-9
        assert R.row_out({"id": "r1"}, case, probs)["correct"] == 1


def test_choice_cases_keep_option_ids_as_criteria_keys():
    case = case_of_row(_row(["invoice", "receipt", "letter"], "receipt"))
    q, kind = R.jev_question(case)
    assert kind == "choice" and list(q["criteria"]) == ["invoice", "receipt", "letter"]
    out = R.row_out({"id": "r1"}, case, R.jev_probs(case, _decide({"q": q})["q"], kind))
    assert out["pred_idx"] == 1 and out["correct"] == 1 and out["answered"]


def test_unanswered_rows_are_kept_and_count_as_wrong():
    case = case_of_row(_row(["a", "b"], "a"))
    out = R.row_out({"id": "r1"}, case, None, "ValueError: state is 9999 tokens")
    assert out == {**out, "answered": False, "correct": 0, "pred_idx": None}
    assert R.summarise([out]) == {"n": 1, "answered": 0, "accuracy": 0.0}
