"""Unit tests for the E38b generator (criteria: docs/E38B_CRITERIA_2026-10-14.md)."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mksh_v.gen_e38b import (  # noqa: E402
    CLASSES, FAMILIES, TASK_TOKENS, assert_clean, generate_split, sc_edit, source_conflict_content,
    split_pairs, ss_content, task_done_content, task_done_edit, urgency_content, _pool,
)

forbidden = pytest.mark.parametrize("text", ["see <img> here", "<doc> x", "[O3] bad", "<row>", "x <eos> y"])


@forbidden
def test_assert_clean_rejects_markers(text):
    with pytest.raises(AssertionError):
        assert_clean(text)


def test_assert_clean_accepts_normal_text():
    assert_clean('Export "q3_report.xlsx" to PDF', "Ticket #2043. Impact: minor. Ref ZC-123.")


def test_split_pairs_disjoint_and_sized():
    parts = split_pairs()
    assert {k: len(v) for k, v in parts.items()} == {"train": 58, "heldout": 19, "test": 19}
    tr, ho, te = (set(parts[k]) for k in ("train", "heldout", "test"))
    assert not (tr & ho or tr & te or ho & te)


def test_task_done_label_matches_rows():
    rng = random.Random(1)
    for _ in range(200):
        label = rng.choice(("false", "true"))
        c = task_done_content(label, rng, "en")
        goal_row = next(lab for name, lab in c["rows"] if name == c["token"])
        assert goal_row == (label == "true")
        assert sum(1 for name, _ in c["rows"] if name == c["token"]) == 1


def test_task_done_irrelevant_keeps_label():
    rng = random.Random(2)
    for _ in range(100):
        label = rng.choice(("false", "true"))
        c = task_done_content(label, rng, "en")
        other = [t for t in TASK_TOKENS if t != c["token"]]
        c2 = task_done_edit(c, label, rng, "irrelevant")
        goal_row = next(lab for name, lab in c2["rows"] if name == c["token"])
        assert goal_row == (label == "true")


def test_source_conflict_label_by_construction():
    rng = random.Random(3)
    for _ in range(200):
        label = rng.choice(("agree", "conflict"))
        c = source_conflict_content(label, rng, "en")
        mismatch = any(c["shown"][f] != c["text_vals"][f] for f in c["claimed"])
        assert mismatch == (label == "conflict")
        assert c["state"].count(":") >= len(c["claimed"])


def test_source_conflict_decisive_flips():
    rng = random.Random(4)
    for _ in range(100):
        label = rng.choice(("agree", "conflict"))
        c = source_conflict_content(label, rng, "en")
        flip = "conflict" if label == "agree" else "agree"
        from mksh_v.gen_e38b import sc_edit
        c2 = sc_edit(c, flip, rng, "en", "decisive")
        mismatch = any(c2["shown"][f] != c["text_vals"][f] for f in c["claimed"])
        assert mismatch == (flip == "conflict")


def test_urgency_label_formula():
    rng = random.Random(5)
    for _ in range(200):
        label = rng.choice(("1", "2", "3", "4", "5"))
        c = urgency_content(label, rng, "en")
        pts = c["banner"] + (1 if c["hours"] <= 4 else 0)
        assert c["u"] == 1 + pts == int(label)
        # the deadline lives only in the image: the state text has no deadline field
        from mksh_v.gen_e38b import STATE
        assert "{hours}" not in STATE["urgency"]["en"]


def test_screen_state_single_kind():
    rng = random.Random(6)
    for label in ("success", "error", "loading", "needs_input", "blocked"):
        for _ in range(20):
            c = ss_content(label, rng, "en")
            assert c["kind"] == label
        if label == "needs_input":
            c = ss_content(label, rng, "en")
            assert any(v == "" for _, v in c["fields"])


def test_generate_split_balanced_all_families():
    for fam in FAMILIES:
        cases = generate_split(fam, "dev", "en", seed=999)
        assert len(cases) == 100
        counts = {}
        for c in cases:
            counts[c.case.target_option] = counts.get(c.case.target_option, 0) + 1
        k = len(CLASSES[fam])
        assert len(counts) == k
        assert max(counts.values()) - min(counts.values()) <= 1


def test_fr_only_in_test():
    with pytest.raises(ValueError):
        generate_split("urgency", "dev", "fr", seed=999)
