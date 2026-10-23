import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
import e44_train as K  # noqa: E402


def test_slots_match_e41_plus_bind():
    c = Counter(K.slot_pool(s) for s in range(8))
    assert c == {"synthetic": 2, "real": 2, "replay": 2, "bind": 2}


def test_selection_key_mean_of_three():
    assert K.selection_key(0.9, 0.6, 0.3, 0.8) == ((0.9 + 0.6 + 0.3) / 3, 0.8)


def test_bind_rows_pinned_and_paired(monkeypatch):
    monkeypatch.setattr(K, "find_file_dir", lambda *a, **k: ROOT / "data/e44/bind")
    pairs, dev = K.load_bind()
    assert set(pairs) == set(K.BIND_WEIGHTS) and dev
    for kind, ps in pairs.items():
        for a, b in ps:
            if kind != "range":
                assert a["image"] == b["image"] and {a["target_option"], b["target_option"]} == {"true", "false"}
                assert a["question"] != b["question"] or a["state"] != b["state"]
    probe = [json.loads(l) for l in (ROOT / "data/e44_probe/rows.jsonl").read_text(encoding="utf-8").splitlines()]
    used = {r["image"] for ps in pairs.values() for p in ps for r in p} | {r["image"] for r in dev}
    assert not used & {r["image"].removeprefix("data/e38a/").removeprefix("data/e41/") for r in probe}
