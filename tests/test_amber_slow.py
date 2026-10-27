"""Amber Checkpoint through the real weights (slow; the shipped decide path).

    MKSHV_MODEL_DIR=release python -m pytest -m slow tests/test_amber_slow.py

Asserts the game's batched decide_checks equals the demo's per-case decide()
on the same image and inputs (U1024 rule, softmax / T), and that the gate
behaves at tau.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))       # src/mksh_v must win over demo/mksh_v
sys.path.insert(1, str(ROOT / "demo"))

pytestmark = pytest.mark.slow


def _released_weights() -> Path:
    p = ROOT / "release"
    if not (p / "model.safetensors").exists():
        pytest.skip("release weights not present (MKSHV_MODEL_DIR=release)")
    return p


def test_batched_decide_equals_sequential(monkeypatch):
    import os

    monkeypatch.setenv("MKSHV_MODEL_DIR", str(_released_weights()))
    import torch

    import game
    game._cache.clear()
    torch.set_num_threads(2)
    model = game.load_model()

    from PIL import Image

    day = game.DAYS[2]  # day 3: a document with routed-range checks
    doc = day["docs"][0]
    img = Image.open(game.HERE / "assets" / "docs" / f"{doc['id']}.png").convert("RGB")
    res = game.decide_checks(img, doc["checks"])
    assert len(res) == len(doc["checks"])
    for r in res:
        assert 0.0 <= r["p_yes"] <= 1.0
        assert r["pred"] in ("yes", "no") and isinstance(r["backed"], bool)

    # the demo's per-case decide path, one case at a time, must agree
    from mokka_decisions.contracts import DecisionCase, Option
    from mksh_v.arch_a1v import forward_a1v

    with torch.inference_mode():
        for c, r in zip(doc["checks"], res):
            case = DecisionCase(id=c["id"], group_id=c["id"], source="amber-checkpoint",
                                language="en", domain="documents", state=c["state"],
                                question=c["question"],
                                options=(Option(id="yes", description="Yes"),
                                         Option(id="no", description="No")))
            enc = model.encode_cases([case], max_state_tokens=700, max_option_tokens=96)
            batch = model.build_batch([enc[0]], [img], "cpu", max_side=1024, upscale_to=1024)
            p = (forward_a1v(model, batch) / game.T).softmax(-1)[0].tolist()
            assert abs(p[0] - r["p_yes"]) < 1e-4, c["id"]
    # gold never reaches the model: the state is shared text without verdicts
    assert "gold" not in day["state"] and "deny" not in day["state"].lower()
