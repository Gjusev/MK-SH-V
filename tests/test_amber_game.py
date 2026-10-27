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
            assert "assets/docs/d" in html
            s = game.stamp(s, game._doc(s["day"], s["doc_i"])["verdict"])
            docs_seen += 1
            s = game.advance(s)
        assert s["phase"] == "day_end"
        assert "LEDGER" in game.render_stage(s)
        s = game.advance(s)
        if s["phase"] == "intro":
            continue
        break
    assert s["phase"] == "campaign_end" and docs_seen == 32
    assert "THE WEEK IN INK" in game.render_stage(s)
    sc = game.day_scores(s, 1)
    assert sc["player_acc"] == 1.0 and sc["model_acc"] == 1.0 and sc["routed"] == 0
    assert sc["checks"] == 8  # day 1: one type check per document
    assert game.advance(s)["phase"] == "intro"  # replay resets to day 1


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


def test_constants_match_release_policy():
    assert game.TAU == 0.55 and game.T == 1.124
    assert game.COVERAGE == 0.9855 and game.CP95_UPPER == 0.0406
    assert game.WEIGHTS_SHA.startswith("ff4a4759")
