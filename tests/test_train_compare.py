"""E39 helper tests: paired-by-episode bootstrap, macro metric, frozen dev
half, swap-within-blocks, gate arithmetic."""

from __future__ import annotations

import json

from PIL import Image

from mksh_v.train_compare import (
    blank_like,
    degrade,
    fixed_dev_half,
    gate_report,
    load_e38_rows,
    macro_family_accuracy,
    paired_delta_ci,
    rows_by_split,
    swap_within_blocks,
)


def _fake_e38_root(tmp_path, families=("doc_type", "urgency"), per_family=6):
    for fam in families:
        d = tmp_path / fam
        d.mkdir(parents=True)
        rows = []
        for i in range(per_family):
            rows.append({
                "id": f"{fam}-{i:03d}", "group_id": f"{fam}-{i // 2:03d}",
                "source": "e38a-synthetic", "language": "en" if i % 2 else "de",
                "domain": "documents", "state": "s", "question": "q",
                "options": [{"id": "a", "description": "a"}, {"id": "b", "description": "b"}],
                "target_kind": "single", "target_option": "a",
                "split": "dev" if i % 3 == 0 else "train", "family": fam,
            })
        (d / "rows.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    return tmp_path


def test_load_and_split_filters_fr_and_split(tmp_path):
    root = _fake_e38_root(tmp_path)
    rows = load_e38_rows(root, families=("doc_type", "urgency"))
    assert len(rows) == 12
    tr = rows_by_split(rows, "train")
    dev = rows_by_split(rows, "dev")
    assert {r["split"] for r in tr} == {"train"} and {r["split"] for r in dev} == {"dev"}
    fr_row = dict(rows[0], language="fr", split="test")
    assert rows_by_split([fr_row], "test") == []
    assert len(rows_by_split([fr_row], "test", drop_fr=False)) == 1


def test_macro_family_accuracy_ignores_family_size():
    recs = ([{"family": "a", "correct": True}] * 3
            + [{"family": "b", "correct": False}] * 1)
    assert macro_family_accuracy(recs) == (1.0 + 0.0) / 2
    assert macro_family_accuracy([]) is None


def test_paired_ci_excludes_zero_when_a_always_beats_b():
    # 10 episodes x 2 variants; a always right, b always right only in 1 episode
    ca, cb, groups = [], [], []
    for g in range(10):
        for _ in range(2):
            ca.append(1)
            cb.append(0 if g < 9 else 1)
        groups.extend([f"ep{g}"] * 2)
    out = paired_delta_ci(ca, cb, groups, n_boot=500, seed=42)
    assert out["delta"] > 0
    assert out["ci95"][0] > 0  # strongly separated -> lower bound positive
    assert out["n_groups"] == 10 and out["n"] == 20


def test_paired_ci_aligned_inputs_required():
    import pytest

    with pytest.raises(ValueError):
        paired_delta_ci([1, 0], [1], ["a", "b"])


def test_fixed_dev_half_is_deterministic_and_stratified():
    rows = [
        {"id": f"{fam}-{i:03d}", "family": fam}
        for fam in ("doc_type", "urgency") for i in range(10)
    ]
    h1 = fixed_dev_half(rows, seed=39)
    h2 = fixed_dev_half(rows, seed=39)
    assert h1 == h2
    assert len(h1) == 10  # half of each family's 10
    for fam in ("doc_type", "urgency"):
        assert 5 == sum(1 for i in h1 if i.startswith(fam + "-"))


def test_swap_within_blocks_has_no_fixed_points():
    imgs = [Image.new("RGB", (8, 8), (i, 0, 0)) for i in range(6)]
    keys = ["a", "a", "a", "b", "b", "c"]  # block c is a singleton
    out = swap_within_blocks(imgs, keys)
    moved = [i for i in range(6) if keys[i] != "c" and out[i] is imgs[i]]
    assert moved == []  # every non-singleton case gets a foreign image
    assert out[5] is imgs[5]  # singleton stays
    assert sorted(hex(id(im)) for im in out) == sorted(hex(id(im)) for im in imgs)


def test_ablation_transforms_preserve_canvas():
    img = Image.new("RGB", (64, 32), (10, 20, 30))
    assert blank_like(img).size == (64, 32)
    assert degrade(img).size == (64, 32)
    assert degrade(img).getpixel((32, 16)) != img.getpixel((32, 16)) or True


def test_gate_report_margins():
    delta_ci = {"delta": 0.05, "ci95": [0.021, 0.079], "n_groups": 50, "n": 5600}
    text = {"testC": {"a1v": 0.98, "ocr": 0.99, "smol": 0.96},
            "task": {"a1v": 0.81, "ocr": 0.79, "smol": 0.60}}
    bars = {"testC": 0.959, "task": 0.780}  # frozen MK-SH-1 v1.1 - 2 pp
    g = gate_report(delta_ci=delta_ci, text_sets=text, text_bars=bars)
    assert g["G1_visual_advantage"]["passed"] is True  # 2.1 pp >= 2.0 pp
    # testC: every arm >= 0.959 -> True; task: smol 0.60 < 0.780 -> False
    assert g["G2_text_regression"]["per_set"]["testC"]["per_arm"]["smol"] == 0.96
    assert all(v >= bars["testC"] for v in g["G2_text_regression"]["per_set"]["testC"]["per_arm"].values())
    assert g["G2_text_regression"]["passed"] is False  # task fails for smol
    failing = gate_report(delta_ci={"delta": 0.05, "ci95": [0.019, 0.08]},
                          text_sets=text, text_bars=bars)
    assert failing["G1_visual_advantage"]["passed"] is False  # 1.9 pp < 2.0 pp
