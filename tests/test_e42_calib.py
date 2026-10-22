import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "e42_calib", Path(__file__).resolve().parents[1] / "kaggle" / "e42_calib.py")
K = importlib.util.module_from_spec(_spec)
sys.modules["e42_calib"] = K
_spec.loader.exec_module(K)


def _case(probs, gold):
    pred = max(range(len(probs)), key=lambda i: probs[i])
    return {"probs": probs, "gold": gold, "pred": pred}


def test_tau_is_the_smallest_grid_value_with_cp95_upper_within_budget():
    # the E40 situation, built to order: 80 confident-correct + 4 wrong at
    # p=0.60. tau=0.30: point risk 4/84 = 4.8 % <= 5 % (the old rule stops
    # here) but CP upper ~11.9 % > 5 %. tau=0.65 drops the 4 wrong rows:
    # k=0 over n=80, CP upper ~4.5 % <= 5 %. The CP rule must go higher.
    cases = [_case([0.99, 0.01], 0) for _ in range(80)]
    cases += [_case([0.60, 0.40], 1) for _ in range(4)]  # wrong, low-confidence
    tau = K.pick_tau(cases, 1.0)
    assert tau["met_target"] is True and tau["rule"] == "cp95_upper"
    assert tau["threshold"] == 0.65 and tau["cal_risk"] == 0.0
    assert tau["cal_cp95"][1] <= K.RISK_BUDGET


def test_tau_falls_back_to_095_when_no_grid_value_qualifies():
    cases = [_case([0.99, 0.01], 1) for _ in range(50)]  # every row confidently wrong
    tau = K.pick_tau(cases, 1.0)
    assert tau == {"threshold": 0.95, "rule": "cp95_upper", "met_target": False}


def test_point_rule_still_available_as_crosscheck_and_is_weaker():
    cases = [_case([0.99, 0.01], 0) for _ in range(80)]
    cases += [_case([0.60, 0.40], 1) for _ in range(4)]
    point = K.pick_tau_point_crosscheck(cases, 1.0)
    assert point["threshold"] == 0.30  # point risk 4/84 already <= 5 %
    cp = K.pick_tau(cases, 1.0)
    assert point["threshold"] is not None  # mokka answers on the same grid
    assert cp["threshold"] >= point["threshold"]  # CP bound is the stricter rule


def test_checkpoint_pins_are_the_e41_ones():
    assert set(K.SEED_CKPTS) == {"42", "43"}
    assert {v[0] for v in K.SEED_CKPTS.values()} == {"mkshv-e41-runs"}
    assert all(v[1].startswith("a1v_real_s") for v in K.SEED_CKPTS.values())
