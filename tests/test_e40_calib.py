import importlib.util
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")

_spec = importlib.util.spec_from_file_location(
    "e40_calib", Path(__file__).resolve().parents[1] / "kaggle" / "e40_calib.py")
E = importlib.util.module_from_spec(_spec)
sys.modules["e40_calib"] = E
_spec.loader.exec_module(E)


def _cases(n, t_true, seed=7):
    """Cases whose probabilities are a true softmax sharpened by 1/t_true."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        k = rng.randint(2, 6)
        logits = [rng.gauss(0, 1.5) for _ in range(k)]
        m = max(logits)
        true = [pow(2.718281828, x - m) for x in logits]
        s = sum(true)
        true = [x / s for x in true]
        r, acc, gold = rng.random(), 0.0, k - 1
        for i, p in enumerate(true):
            acc += p
            if r <= acc:
                gold = i
                break
        shown = E.temp_probs(true, 1.0 / t_true)  # overconfident by t_true
        out.append({"probs": shown, "gold": gold,
                    "pred": max(range(k), key=lambda i: shown[i]), "family": "x"})
    return out


def test_temperature_is_recovered_and_matches_mokka():
    cases = _cases(3000, t_true=1.8)
    t = E.fit_T(cases)
    assert abs(t - 1.8) < 0.15
    assert abs(E.mokka_T(cases) - t) < 0.02  # same optimum, different search


def test_clopper_pearson_matches_mksh1_model_card():
    # MK-SH-1 model card (E36): 21/432 -> 3.0 to 7.3 %; 1/47 -> 0.1 to 11.3 %
    lo, hi = E.clopper_pearson(21, 432)
    assert (round(lo, 3), round(hi, 3)) == (0.030, 0.073)
    lo, hi = E.clopper_pearson(1, 47)
    assert (round(lo, 3), round(hi, 3)) == (0.001, 0.113)
    assert E.clopper_pearson(0, 10)[0] == 0.0 and E.clopper_pearson(10, 10)[1] == 1.0


def test_pick_tau_is_mksh1_rule_and_risk_at_agrees():
    cases = _cases(3000, t_true=1.5, seed=3)
    tau = E.pick_tau(cases, 1.0)
    assert tau["threshold"] in E.TAU_GRID
    r = E.risk_at(cases, tau["threshold"], 1.0)
    assert r["n_acc"] == tau["accepted"]
    if tau["met_target"]:
        assert r["risk"] <= E.RISK_BUDGET
        prev = [x for x in E.TAU_GRID if x < tau["threshold"]]
        if prev:  # smallest grid value meeting the budget
            assert E.risk_at(cases, prev[-1], 1.0)["risk"] > E.RISK_BUDGET
    assert len(E.curve(cases, 1.0)) == len(E.TAU_GRID)
