"""Amber Checkpoint days: plan determinism, gold per discrepancy, verdict rules."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gen_amber_checkpoint import DAYS, gold_of, plan  # noqa: E402


def test_plan_is_deterministic():
    import json
    assert json.dumps(plan(), sort_keys=True) == json.dumps(plan(), sort_keys=True)


def test_shape_and_mix():
    c = plan()
    assert [d["day"] for d in c["days"]] == [1, 2, 3, 4]
    assert all(len(d["docs"]) == 8 for d in c["days"])
    for d in c["days"]:
        verdicts = [x["verdict"] for x in d["docs"]]
        assert 3 <= verdicts.count("deny") <= 5, (d["day"], verdicts)
        assert set(x["id"] for x in d["docs"]) == {f"d{d['day']}-{i}" for i in range(1, 9)}


def test_check_matrix_accumulates():
    c = plan()
    kinds = {d["day"]: {k for x in d["docs"] for k in (ck["kind"] for ck in x["checks"])} for d in c["days"]}
    assert kinds[1] == {"type"}
    assert kinds[2] == {"type", "manifest", "permit", "seal"}
    assert kinds[3] == {"type", "manifest", "seal", "permit", "expiry"}
    # day 4 has no Korvenian documents, so no permit check that day
    assert kinds[4] == {"type", "manifest", "seal", "expiry", "stay", "amount"}
    # every discrepancy type appears somewhere in the campaign
    assert {v for d in c["days"] for x in d["docs"] for v in x["violations"]} == \
           {"issuer", "permit", "seal", "manifest", "expiry", "stay", "amount"}


def test_gold_matches_injected_discrepancies():
    import datetime as dt
    c = plan()
    for day in c["days"]:
        d = day["day"]
        for x in day["docs"]:
            for ck in x["checks"]:
                assert ck["gold"] == gold_of(ck["kind"], x, d)
                # gold derivable from the content, and the model input never carries it
                assert "gold" not in ck["state"] and x["verdict"] not in ck["state"]
            viol = set(x["violations"])
            assert x["verdict"] == ("deny" if viol else "approve")
            assert ("issuer" in viol) == (x["nationality"] == "KORVENIA")
            assert ("seal" in viol) == (not x["seal"] and d >= 2)
            assert ("manifest" in viol) == (not x["manifest_listed"] and d >= 2)
            assert ("permit" in viol) == (x["nationality"] == "KORVENIA"
                                          and not x["work_permit"] and d >= 2)
            expired = dt.date.fromisoformat(x["expiry_iso"]) < DAYS[d]["today"]
            assert ("expiry" in viol) == (expired and d >= 3)
            assert ("stay" in viol) == (x["stay_days"] > 30 and d >= 4)
            assert ("amount" in viol) == (x["declared"] > 5000 and d >= 4)
            # day-1 docs never carry day>=2 discrepancies
            if d == 1:
                assert viol <= {"issuer"}


def test_manifest_lists_every_listed_holder():
    c = plan()
    for day in c["days"]:
        listed = {x["holder"] for x in day["docs"] if x["manifest_listed"]}
        assert listed <= set(day["manifest"]) and len(day["manifest"]) == 8
        assert not {x["holder"] for x in day["docs"] if not x["manifest_listed"]} & set(day["manifest"])


def test_written_days_json_matches_plan():
    import json
    p = ROOT / "demo/game/days.json"
    if not p.exists():
        import pytest
        pytest.skip("days.json not generated yet")
    written = json.loads(p.read_text(encoding="utf-8"))
    fresh = plan()
    assert written["days"] == fresh["days"]  # images/shas aside, the days are the plan
