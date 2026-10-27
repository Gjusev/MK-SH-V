"""Amber Checkpoint days v2 (customs desk): determinism, gold per discrepancy,
measured check framings, verdict rules."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gen_amber_checkpoint import CAP, plan  # noqa: E402


def test_plan_is_deterministic():
    import json
    assert json.dumps(plan(), sort_keys=True) == json.dumps(plan(), sort_keys=True)


def test_shape_and_mix():
    c = plan()
    assert [d["day"] for d in c["days"]] == [1, 2, 3, 4]
    for d in c["days"]:
        verdicts = [x["verdict"] for x in d["docs"]]
        assert 3 <= verdicts.count("deny") <= 5, (d["day"], verdicts)
        assert all(1 <= len(x["checks"]) <= 3 for x in d["docs"])


def test_check_matrix_and_measured_framings():
    c = plan()
    kinds = {d["day"]: {ck["kind"] for x in d["docs"] for ck in x["checks"]} for d in c["days"]}
    assert kinds[1] == {"type"}
    assert kinds[2] == {"type", "manifest", "form"}
    assert kinds[3] == {"type", "manifest", "form", "amount"}
    assert kinds[4] == {"type", "manifest", "form", "amount"}
    for d in c["days"]:
        for x in d["docs"]:
            for ck in x["checks"]:
                assert ck["state"] == "Scanned document page."
                if ck["kind"] == "type":
                    assert ck["question"] == "Which document type is this?" and len(ck["options"]) == 6
                elif ck["kind"] == "form":
                    assert ck["question"].startswith("Is this form complete?") and len(ck["options"]) == 3
                elif ck["kind"] == "amount":
                    assert ck["question"] == "Is the total on this document at least 5,000.00 VEL?"
                elif ck["kind"] == "manifest":
                    assert len(ck["options"]) == 5
                assert ck["gold"] in {o["id"] for o in ck["options"]}
                # the model input never carries the gold or the verdict
                assert "gold" not in ck["state"] and x["verdict"] not in ck["question"]


def test_gold_matches_the_paper():
    c = plan()
    for day in c["days"]:
        for x in day["docs"]:
            viol = set(x["violations"])
            assert x["verdict"] == ("deny" if viol else "approve")
            assert ("paper type" in viol) == (x["kind"] == "wrong")
            assert ("incomplete form" in viol) == (x["kind"] == "form" and x["form_label"] != "complete")
            assert ("not on manifest" in viol) == (x["kind"] == "inv" and not x["listed"])
            assert ("over the cap" in viol) == (x["kind"] == "inv" and x["total"] >= CAP)
            for ck in x["checks"]:
                if ck["kind"] == "type":
                    assert ck["gold"] == x["doc_type"]
                elif ck["kind"] == "form":
                    assert ck["gold"] == x["form_label"]
                elif ck["kind"] == "amount":
                    assert ck["gold"] == ("true" if x["total"] >= CAP else "false")
                elif ck["kind"] == "manifest":
                    gold_name = next(o["description"] for o in ck["options"] if o["id"] == ck["gold"])
                    assert gold_name == x["sender"]
                    on_paper = [o["description"] for o in ck["options"] if o["id"] != ck["gold"]]
                    assert x["sender"] not in on_paper  # unique options


def test_manifest_and_gap_design():
    c = plan()
    for day in c["days"]:
        listed = {x["sender"] for x in day["docs"] if x["kind"] == "inv" and x["listed"]}
        unlisted = {x["sender"] for x in day["docs"] if x["kind"] == "inv" and not x["listed"]}
        assert listed <= set(day["manifest"]) and not unlisted & set(day["manifest"])
    totals = lambda day, side: [x["total"] for x in c["days"][day - 1]["docs"]
                                if x["kind"] == "inv" and (x["total"] >= CAP) == (side == "over")]
    assert all(4000 < t < 4996 for t in totals(4, "in")), "day-4 fine gaps within the cap"
    assert all(5005 < t <= 5250 for t in totals(4, "over")), "day-4 fine gaps over the cap"
    assert any(t < 4000 for t in totals(3, "in")), "day-3 keeps wide gaps too"


def test_written_days_json_matches_plan():
    import json
    p = ROOT / "demo/game/days.json"
    if not p.exists():
        import pytest
        pytest.skip("days.json not generated yet")
    written = json.loads(p.read_text(encoding="utf-8"))
    assert written["days"] == plan()["days"]
