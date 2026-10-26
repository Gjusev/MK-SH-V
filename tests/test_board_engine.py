"""Board engine mapping tests (fast; no weights loaded).

The full-engine check is slow: MKSHV_MODEL_DIR=release python -m pytest -m slow tests/test_board_engine.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mksh_v.board_engine import STATE, Unsupported, case_of_payload  # noqa: E402


def payload(state={}, instructions="Which letter is a vowel?", criteria=None, images=("img/a.png",)):
    return {"state": state, "prompt": None,
            "questions": {"answer": {"type": "choice", "instructions": instructions,
                                     "criteria": criteria or {"A": "apple", "B": "table", "C": "orange", "D": "chair"}}},
            "images": list(images)}


def test_vision_row_maps_to_case():
    case, image = case_of_payload(payload())
    assert image == "img/a.png" and case.state == STATE
    assert case.question == "Which letter is a vowel?"
    assert [o.id for o in case.options] == ["A", "B", "C", "D"]
    assert [o.description for o in case.options] == ["apple", "table", "orange", "chair"]


def test_warmup_state_passes_through_and_image_is_optional():
    case, image = case_of_payload(payload(state="The color is red.", images=()))
    assert case.state == "The color is red." and image is None


def test_multi_image_is_declared_unsupported():
    with pytest.raises(Unsupported):
        case_of_payload(payload(images=("a.png", "b.png")))


def test_non_choice_question_is_declared_unsupported():
    p = payload()
    p["questions"] = {"answer": {"type": "noul", "instructions": "x", "criteria": {"yes": 1, "no": 0}}}
    with pytest.raises(Unsupported):
        case_of_payload(p)


@pytest.mark.slow
def test_engine_answers_one_row():  # loads the released weights
    import os

    os.environ.setdefault("MKSHV_MODEL_DIR", str(ROOT / "release"))
    from mksh_v.board_engine import MkshvVisionEngine

    eng = MkshvVisionEngine(image_root=str(ROOT))
    resp, raw = eng(**payload(images=(str(ROOT / "demo/examples/cord-receipt-1.png"),)))
    a = resp["answers"]["answer"]
    assert a["type"] == "choice" and a["choice"] in "ABCD"
    assert math.isclose(sum(a["probabilities"].values()), 1.0, abs_tol=1e-4)
    assert raw["T"] == 1.124
