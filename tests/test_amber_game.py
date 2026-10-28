"""Amber Checkpoint game module: state machine, cache, decide path with a fake
model, and stage rendering. No weights (the slow, real-weights test is
tests/test_amber_slow.py, opt in with -m slow)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))       # src/mksh_v must win over demo/mksh_v
sys.path.insert(1, str(ROOT / "demo"))

import game  # noqa: E402
import mksh_v.arch_a1v as arch  # noqa: E402


class FakeModel:
    def encode_cases(self, cases, **kw):
        return list(cases)

    def build_batch(self, enc, imgs, device, max_side, upscale_to):
        assert max_side == 1024 and upscale_to == 1024, "the U1024 input rule"
        assert device == "cpu" and len(imgs) == len(enc)
        return list(enc)


def _gold_desc(check):
    return next(o["description"] for o in check["options"] if o["id"] == check["gold"])


def seed_cache(backed_all=True, p=0.97):
    """Prefill the shared cache: the model is right everywhere."""
    game._cache.clear()
    for d in game.DAYS:
        for doc in d["docs"]:
            for c in doc["checks"]:
                game._cache[c["id"]] = {"p_pred": p, "pred": c["gold"],
                                        "pred_desc": _gold_desc(c), "backed": backed_all}


def test_decide_checks_batches_and_caches(monkeypatch):
    game._cache.clear()
    checks = [c for c in game.DAYS[3]["docs"][0]["checks"] if len(c["options"]) == 2]  # amount: TF
    calls = {"n": 0}

    def fwd(model, batch):
        calls["n"] += len(batch)
        import torch
        return torch.tensor([[0.9, 0.1] for _ in batch])

    monkeypatch.setattr(arch, "forward_a1v", fwd)
    monkeypatch.setattr(game, "load_model", lambda: FakeModel())
    from PIL import Image

    img = Image.new("RGB", (460, 560))
    r1 = game.decide_checks(img, checks)
    assert r1[0]["p_pred"] > 0.5 and r1[0]["pred"] == "false" and r1[0]["backed"]  # option order: false, true
    n1 = calls["n"]
    r2 = game.decide_checks(img, checks)  # served from cache, no model call
    assert r2 == r1 and calls["n"] == n1


def test_gate_routes_below_tau(monkeypatch):
    game._cache.clear()
    checks = [c for c in game.DAYS[3]["docs"][0]["checks"] if len(c["options"]) == 2]
    monkeypatch.setattr(game, "load_model", lambda: FakeModel())

    def fwd(model, batch):
        import torch
        return torch.tensor([[0.52, 0.48] for _ in batch])

    monkeypatch.setattr(arch, "forward_a1v", fwd)
    from PIL import Image
    r = game.decide_checks(Image.new("RGB", (8, 8)), checks)[0]
    assert not r["backed"] and 0.5 < r["p_pred"] < 0.55  # top prob under the gate


def test_full_campaign_state_machine():
    seed_cache()
    s = game.new_session()
    docs_seen = 0
    while True:
        s = game.advance(s)  # intro -> doc 0
        while s["phase"] == "doc":
            html = game.render_stage(s)
            assert "prob-bar" in html and "tau" in html
            assert "—" not in html and "–" not in html  # no em/en dashes (rubric)
            assert "assets/docs/d" in html and "class='window" in html  # the traveler
            s = game.stamp(s, game._doc(s["day"], s["doc_i"])["verdict"])
            docs_seen += 1
            s = game.advance(s)
        assert s["phase"] == "day_end"
        assert "BOOTH REPORT" in game.render_stage(s)
        s = game.advance(s)
        if s["phase"] == "intro":
            continue
        break
    assert s["phase"] == "campaign_end" and docs_seen == 24
    assert "THE WEEK IN INK" in game.render_stage(s)
    sc = game.day_scores(s, 1)
    assert sc["player_acc"] == 1.0 and sc["model_acc"] == 1.0 and sc["routed"] == 0
    assert sc["checks"] == 6  # day 1: one type check per document
    assert game.advance(s)["phase"] == "intro"  # replay resets to day 1
    assert s["credits"] >= 0 and s["best_streak"] >= 0


def test_wrong_stamp_scores_against_gold():
    seed_cache()
    s = game.advance(game.new_session())
    d = game._doc(1, 0)
    s = game.stamp(s, "deny" if d["verdict"] == "approve" else "approve")
    assert s["history"][0]["player"] != s["history"][0]["gold"]
    assert "WRONG CALL" in game.render_stage(s)
    assert game.stamp(s, "deny")["history"] == s["history"]  # one stamp per document


def test_unstamped_filing_and_routing_display():
    seed_cache(backed_all=False, p=0.52)  # every check routed to the player
    s = game.advance(game.new_session())
    html = game.render_stage(s)
    assert "SECONDARY INSPECTION" in html and "tag" in html  # routing tag on the doc
    s = game.file_unstamped(s)
    assert "FILED UNSTAMPED" in game.render_stage(s)
    sc = game.day_scores({**s, "history": [s["history"][0]]}, 1)
    assert sc["routed"] >= 1 and sc["player_acc"] == 0.0


def test_bribe_flow_and_inspection():
    seed_cache()
    s = game.new_session()
    # walk to day 2 doc 3 (d2-3, the liar, carries the +40 gift): one full day first
    for _ in range(1):
        s = game.advance(s)  # intro -> day1 doc0
        while s["phase"] == "doc":
            s = game.stamp(s, game._doc(s["day"], s["doc_i"])["verdict"])
            s = game.advance(s)
        s = game.advance(s)  # day_end -> next intro
    assert s["day"] == 2
    while not (s["phase"] == "doc" and game._doc(2, s["doc_i"])["id"] == "d2-3"):
        s = game.stamp(s, game._doc(2, s["doc_i"])["verdict"])
        s = game.advance(s)
    d = game._doc(2, s["doc_i"])
    assert "bribe" in d and d["verdict"] == "deny"
    html = game.render_stage(s)
    assert "GIFT" in html and "TAKE" not in html  # tag renders; the button is gradio-side
    before = s["credits"]
    s = game.accept_gift(s)
    assert s["credits"] == before + 40 and s["contraband"] == 1
    assert game.accept_gift(s)["credits"] == before + 40  # once only
    s = game.stamp(s, d["verdict"])
    while s["phase"] == "doc":  # finish the day: remaining docs, then day_end
        s = game.stamp(s, game._doc(s["day"], s["doc_i"])["verdict"])
        s = game.advance(s)
    caught = any(b["caught"] for b in s["bribes"])
    # day 2 fully correct: gift +40, four remaining stamps +20, clean shift +10
    assert s["credits"] == before + 40 + 20 + 10 - (60 if caught else 0)
    if caught:
        assert s["streak"] == 0
    # the coin is fixed per document: replay must agree
    s2 = game.new_session()
    for _ in range(1):
        s2 = game.advance(s2)
        while s2["phase"] == "doc":
            s2 = game.stamp(s2, game._doc(s2["day"], s2["doc_i"])["verdict"])
            s2 = game.advance(s2)
        s2 = game.advance(s2)
    while not (s2["phase"] == "doc" and game._doc(2, s2["doc_i"])["id"] == "d2-3"):
        s2 = game.stamp(s2, game._doc(2, s2["doc_i"])["verdict"])
        s2 = game.advance(s2)
    s2 = game.accept_gift(s2)
    s2 = game.stamp(s2, "deny")
    while s2["phase"] == "doc":
        s2 = game.stamp(s2, game._doc(s2["day"], s2["doc_i"])["verdict"])
        s2 = game.advance(s2)
    assert any(b["caught"] for b in s2["bribes"]) == caught


def test_rank_thresholds_and_summary():
    seed_cache()
    s = game.advance(game.new_session())
    while True:
        while s["phase"] == "doc":
            s = game.stamp(s, game._doc(s["day"], s["doc_i"])["verdict"])  # perfect week
            s = game.advance(s)
        if s["phase"] == "campaign_end":
            break
        s = game.advance(s)
    html = game.render_stage(s)
    # 24 correct x 5 + 4 clean days x 10 = 160 -> Officer of the Month
    assert "OFFICER OF THE MONTH" in html and s["credits"] == 160
    assert "acp-summary" in html and "Player accuracy 100%" in html


def test_personas_reactions_and_sfx():
    seed_cache()
    s = game.advance(game.new_session())
    d = game._doc(1, 0)
    html = game.render_stage(s)
    assert d["traveler"]["persona"].upper() in html
    assert "data-sfx='ding'" not in html  # day-1 docs never route
    assert "polaroid" in html
    s = game.stamp(s, d["verdict"])
    html = game.render_stage(s)
    assert "data-sfx='thump'" in html
    # reaction bust swaps in (smile on approve)
    base = d["traveler"]["bust"].rsplit("/", 1)[-1].replace("_24.png", "")
    assert f"{base}_smile_24.png" in html


def test_constants_match_release_policy():
    assert game.TAU == 0.55 and game.T == 1.124
    assert game.COVERAGE == 0.9855 and game.CP95_UPPER == 0.0406
    assert game.WEIGHTS_SHA.startswith("ff4a4759")
