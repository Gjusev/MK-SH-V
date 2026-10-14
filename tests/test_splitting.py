"""Splitting tests: deterministic hash assignment and the leakage guard."""

from __future__ import annotations

import pytest
from mokka_decisions.contracts import DecisionCase, Option

from mksh_v.splitting import (
    assert_no_leakage,
    split_keys_by_hash,
    stratified_template_split,
)
from mksh_v.synth_doc_type import ALL_TEMPLATES, DOC_TYPES


def test_stratified_split_covers_every_class_on_both_sides():
    train, dev = stratified_template_split(ALL_TEMPLATES)
    assert set(train) | set(dev) == set(ALL_TEMPLATES)
    assert not (set(train) & set(dev))
    for doc_type in DOC_TYPES:
        assert any(t.startswith(doc_type + "-") for t in train), doc_type
        assert any(t.startswith(doc_type + "-") for t in dev), doc_type


def test_stratified_split_is_deterministic():
    assert stratified_template_split(ALL_TEMPLATES) == stratified_template_split(ALL_TEMPLATES)


def test_stratified_split_refuses_starved_classes():
    with pytest.raises(ValueError, match="cannot spare"):
        stratified_template_split(["invoice-classic", "invoice-compact"], dev_layouts_per_class=2)


def test_hash_assignment_is_deterministic_and_disjoint():
    keys = [f"invoice-{l}" for l in ("classic", "compact", "boxed")] + ["receipt-classic"]
    a = split_keys_by_hash(keys, dev_fraction=0.25)
    b = split_keys_by_hash(keys, dev_fraction=0.25)
    assert a == b  # pure function of the key set
    train, dev = a
    assert set(train) | set(dev) == set(keys)
    assert not (set(train) & set(dev))
    assert len(dev) == 1


def test_leakage_guard_passes_clean_and_catches_leak():
    def case(group, split):
        return DecisionCase(
            id=f"{group}-{split}", group_id=group, source="t", language="en",
            domain="t", state="s", question="q",
            options=(Option(id="a", description="a"), Option(id="b", description="b")),
            target_kind="single", target_option="a", split=split,
        )

    clean = [case("t1", "train"), case("t2", "train"), case("d1", "dev")]
    assert assert_no_leakage(clean)["dev"] == {"d1"}
    leak = clean + [case("t1", "dev")]  # same group, two splits
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(leak)
