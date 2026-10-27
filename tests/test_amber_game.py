"""Amber Checkpoint game module: state machine, cache, decide path with a fake
model, and stage rendering. No weights (the slow, real-weights test is
tests/test_amber_slow.py, opt in with -m slow)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))       # src/mksh_v must win over demo/mksh_v
sys.path.insert(1, str(ROOT / "demo"))      # the game package lives in demo/game

import game  # noqa: E402
import mksh_v.arch_a1v as arch  # noqa: E402


class FakeModel:
    def __init__(self):
        self.encode_calls = 0

    def encode_cases(self, cases, **kw):
        self.encode_calls += 1
        return list(cases)

    def build_batch(self, enc, imgs, device, max_side, upscale_to):
        assert max_side == 1024 and upscale_to == 1024, "the U1024 input rule"
        assert device == "cpu"
        assert len(imgs) == len(enc)
        return list(enc)


def _fake_forward(logits_for):
    def fwd(model, batch):
        return type("L", (), {"softmax": lambda self, dim: self})  # pragma: no cover
    raise NotImplementedError  # replaced below


def install_fake(monkeypatch, p_yes_by_id):
    calls = {"n": 0}

    def fwd(model, batch):
        calls["n"] += len(batch)
        import torch
        return torch.tensor([[p_yes_by_id[c.id], 1 - p_yes_by_id[c.id]] for c in batch])

    monkeypatch.setattr(arch, "forward_a1v", fwd)
    monkeypatch.setattr(game, "load_model", lambda: FakeModel())
    return calls


def test_decide_checks_batches_and_caches(monkeypatch):
    game._cache.clear()
    checks = game.DAYS[0]["docs"][0]["checks"]  # day 1: one check per doc
    p = {"c0": 0.9}
    keyed = {checks[0]["id"]: 0.9}
    calls = install_fake(monkeypatch, keyed)
    from PIL import Image

    img = Image.new("RGB", (460, 560))
    r1 = game.decide_checks(img, checks)
    assert r1[0]["p_yes"] > 0.5 and r1[0]["pred"] == "yes" and r1[0]["backed"]
    n1 = calls["n"]
    r2 = game.decide_checks(img, checks)  # served from cache, no model call
    assert r2 == r1 and calls["n"] == n1


def test_gate_routes_below_tau(monkeypatch):
    game._cache.clear()
    checks = game.DAYS[0]["docs"][0]["checks"]
    monkeypatch.setattr(game, "load_model", lambda: FakeModel())

    def fwd(model, batch):
        import torch
        # p(yes)=0.52 -> top prob 0.52 < tau 0.55: SECONDARY INSPECTION
        return torch.tensor([[0.52, 0.48] for _ in batch])

    monkeypatch.setattr(arch, "forward_a1v", fwd)
    from PIL import Image
    r = game.decide_checks(Image.new("RGB", (8, 8)), checks)[0]
    assert not r["backed"] and r["pred"] == "yes"


def test_full_campaign_state_machine(monkeypatch):
    game._cache.clear()
    every = {c["id"]: 0.97 for d in game.DAYS for doc in d["docs"] for c in doc["checks"]}
    install_fake(monkeypatch, every)
    s = game.new_session()
    docs_seen = 0
    while True:
        s = game.advance(s)  # intro -> doc 0
        while s["phase"] == "doc":
            html = game.render_stage(s)
            assert "prob-bar" in html and "tau" in html
            assert "—" not in html and "–" not in html  # no em/en dashes (rubric)
            assert "assets/docs/d" in html  # the document itself
            s = game.stamp(s, game._doc(s["day"], s["doc_i"])["verdict"])
            docs_seen += 1
            s = game.advance(s)
        assert s["phase"] == "day_end"
        end = game.render_stage(s)
        assert "LEDGER" in end and "player accuracy" in end
        s = game.advance(s)
        if s["phase"] == "intro":
            continue
        break
    assert s["phase"] == "campaign_end" and docs_seen == 32
    final = game.render_stage(s)
    assert "THE WEEK IN INK" in final and game.HONESTY[:30] in final
    sc = game.day_scores(s, 1)
    # the fake model answers yes with 0.97 everywhere: player perfect, model right
    # exactly on the gold-yes checks of day 1 (5 of 8), gate never routes
    assert sc["player_acc"] == 1.0 and sc["model_acc"] == 0.625 and sc["routed"] == 0
    assert game.advance(s)["phase"] == "intro"  # replay resets to day 1


def test_wrong_stamp_scores_against_gold(monkeypatch):
    game._cache.clear()
    every = {c["id"]: 0.97 for d in game.DAYS for doc in d["docs"] for c in doc["checks"]}
    install_fake(monkeypatch, every)
    s = game.new_session()
    s = game.advance(s)
    d = game._doc(1, 0)
    s = game.stamp(s, "deny" if d["verdict"] == "approve" else "approve")
    assert s["history"][0]["player"] != s["history"][0]["gold"]
    assert "WRONG CALL" in game.render_stage(s)
    # double stamping one document is not allowed
    s2 = game.stamp(s, "deny")
    assert s2["history"] == s["history"]


def test_unstamped_filing(monkeypatch):
    game._cache.clear()
    every = {c["id"]: 0.97 for d in game.DAYS for doc in d["docs"] for c in doc["checks"]}
    install_fake(monkeypatch, every)
    s = game.advance(game.new_session())
    s = game.file_unstamped(s)
    assert "FILED UNSTAMPED" in game.render_stage(s)


def test_constants_match_release_policy():
    assert game.TAU == 0.55 and game.T == 1.124
    assert game.COVERAGE == 0.9855 and game.CP95_UPPER == 0.0406
    assert game.CHECKPOINT_SHA.startswith("afcbeaa4")
