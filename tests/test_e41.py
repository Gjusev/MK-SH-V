import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "e41_train", Path(__file__).resolve().parents[1] / "kaggle" / "e41_train.py")
K = importlib.util.module_from_spec(_spec)
sys.modules["e41_train"] = K
_spec.loader.exec_module(K)


def test_slots_per_step_match_the_criteria():
    # effective batch 32 = 8 microbatches x 4: 2 replay (as E39), 2 real, 4 synthetic
    assert Counter(K.slot_pool(s) for s in range(8)) == {"replay": 2, "real": 2, "synthetic": 4}
    assert [K.slot_pool(s) for s in (3, 7)] == ["replay", "replay"]


def test_selection_is_the_mean_with_synthetic_acc_tiebreak():
    assert K.selection_key(0.9, 0.5, 0.8) > K.selection_key(0.95, 0.4, 0.99)
    assert K.selection_key(0.9, 0.5, 0.81) > K.selection_key(0.9, 0.5, 0.80)


def test_realtrain_sha_is_enforced(tmp_path, monkeypatch):
    for name in ("cord", "gui_odyssey"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "rows.jsonl").write_text(
            json.dumps({"id": name, "split": "train"}) + "\n" + json.dumps({"id": name + "d", "split": "dev"}) + "\n")
    import hashlib
    shas = {n: hashlib.sha256((tmp_path / n / "rows.jsonl").read_bytes()).hexdigest()
            for n in ("cord", "gui_odyssey")}
    monkeypatch.setattr(K, "REALTRAIN_ROWS_SHA", shas)
    train, dev = K.load_realtrain(tmp_path)
    assert [r["id"] for r in train] == ["cord", "gui_odyssey"] and len(dev) == 2
    monkeypatch.setattr(K, "REALTRAIN_ROWS_SHA", {**shas, "cord": "0" * 64})
    with pytest.raises(SystemExit):
        K.load_realtrain(tmp_path)
