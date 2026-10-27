"""E50 report: M3T measure (swap/own/both-right/any-exclusion/p_true) and the
two exact-count readings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e50_report as r  # noqa: E402

OPTS = [{"id": "false"}, {"id": "true"}]


def row(eid: str, m3: str, goal: str) -> dict:
    return {"id": f"e50-m3t-{eid}-{m3}-en", "group_id": eid, "m3": m3, "goal": goal,
            "state": f"... {goal} ...", "options": OPTS}


def case(eid: str, m3: str, p: float) -> dict:
    return {"id": f"e50-m3t-{eid}-{m3}-en", "probs": [1 - p, p], "gold": 1 if m3 == "own" else 0,
            "pred": int(p >= 0.5)}


def m3t(per_ep: list[tuple[str, str, float, float]]) -> tuple[list[dict], dict[str, dict]]:
    """(eid, swap_goal, p_own, p_swap); pred = p >= 0.5, gold own=true swap=false."""
    cases, meta = [], {}
    for eid, goal, po, ps in per_ep:
        for m3, p in (("own", po), ("swap", ps)):
            meta[f"e50-m3t-{eid}-{m3}-en"] = row(eid, m3, goal)
            cases.append(case(eid, m3, p))
    return cases, meta


def test_measure_counts_any_exclusion_and_ptrue():
    cases, meta = m3t([("e1", "book any flight", 0.9, 0.4),  # own ok, swap ok (p<0.5 rejects), " any " goal
                       ("e2", "send email", 0.3, 0.2),       # own wrong, swap ok
                       ("e3", "send email", 0.8, 0.7),       # own ok, swap wrong (accepts swap)
                       ("e4", "send email", 0.6, 0.9)])      # own ok, swap wrong
    m = r.measure(cases, meta)
    assert m["episodes"] == 4 and m["own_n"] == 3 and m["swap_n"] == 2
    assert m["own_acc"] == 0.75 and m["swap_acc"] == 0.5 and m["both_right"] == 1
    assert m["swap_without_any"] == {"acc": round(1 / 3, 4), "n": 3}  # e1 excluded by " any "
    assert m["mean_p_true"]["own"] == round((0.9 + 0.3 + 0.8 + 0.6) / 4, 4)
    assert m["mean_p_true"]["swap"] == round((0.4 + 0.2 + 0.7 + 0.9) / 4, 4)


def test_readings_boundaries_exact():
    def m(n: int, swap_n: int, own_n: int) -> dict:
        return {"episodes": n, "swap_n": swap_n, "own_n": own_n}
    # transfer boundary at exactly 0.80 (320/400), own cost exactly 3 pp (12 episodes) per seed
    ok = r.readings({"e47": {42: m(400, 330, 340), 43: m(400, 335, 345)},
                     "e49": {42: m(400, 320, 328), 43: m(400, 321, 333)}})
    assert ok["transfer_e49_swap_ge_080_both_seeds"] and ok["own_cost_e49_ge_e47_minus_3pp_both_seeds"]
    assert ok["outcome"] == "1_and_2"
    # one seed at 319/400 fails transfer even if the other holds; own cost fine
    t = r.readings({"e47": {42: m(400, 330, 340), 43: m(400, 335, 345)},
                    "e49": {42: m(400, 319, 328), 43: m(400, 321, 333)}})
    assert not t["transfer_e49_swap_ge_080_both_seeds"] and t["outcome"] == "transfer_fails"
    # own cost 3.25 pp on s42 (340 -> 327) fails with transfer holding
    o = r.readings({"e47": {42: m(400, 330, 340), 43: m(400, 335, 345)},
                    "e49": {42: m(400, 320, 327), 43: m(400, 321, 333)}})
    assert o["transfer_e49_swap_ge_080_both_seeds"] and o["outcome"] == "1_only"
