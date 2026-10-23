"""E43 report math: extended-grid tau rule, no-feasible outcome, AUROC, cluster bootstrap."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e43_report as r  # noqa: E402


def case(p_true: float, gold: int, gid: str = "g") -> dict:
    probs = [1 - p_true, p_true]
    return {"probs": probs, "gold": gold, "pred": int(p_true >= 0.5), "group_id": gid}


def test_extended_grid_reaches_above_095():
    # 400 correct at 0.9995, plus 30 wrong at 0.975: only tau >= 0.98 excludes the wrong ones
    cases = [case(0.9995, 1, f"c{i}") for i in range(400)] + [case(0.975, 0, f"w{i}") for i in range(30)]
    got = r.pick_tau_ext(cases, 1.0)
    assert got["feasible"] and got["threshold"] == 0.98


def test_no_feasible_gate():
    cases = [case(0.999, 1, f"c{i}") for i in range(50)] + [case(0.9999, 0, f"w{i}") for i in range(50)]
    assert r.pick_tau_ext(cases, 1.0) == {"threshold": None, "feasible": False}


def test_auroc_perfect_and_inverted():
    good = [case(0.99, 1), case(0.98, 1), case(0.6, 0), case(0.55, 0)]
    assert r.auroc(good, 1.0) == 1.0
    bad = [case(0.6, 1), case(0.55, 1), case(0.99, 0), case(0.98, 0)]
    assert r.auroc(bad, 1.0) == 0.0


def test_cluster_bootstrap_deterministic():
    cases = [case(0.99, 1, f"g{i // 3}") for i in range(60)] + [case(0.99, 0, f"h{i // 3}") for i in range(6)]
    a = r.cluster_bootstrap(cases, 0.5, 1.0, n=200)
    assert a == r.cluster_bootstrap(cases, 0.5, 1.0, n=200) and a[0] <= 6 / 66 <= a[1]
