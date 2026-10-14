import random
from collections import Counter

import pytest

from mksh_v.gen_e38a import (
    CLASSES, COUNTS, FAMILIES, FC_CLASSES, LANGS, TRAIN_LANGS, fmt_amount, form_content, form_edit,
    form_label, generate_split, split_templates,
)


@pytest.fixture(scope="module")
def data():
    return {(f, s, l): generate_split(f, s, l)
            for f in FAMILIES for s in COUNTS for l in (LANGS if s == "test" else TRAIN_LANGS)}


def test_counts_and_exact_balance(data):
    for (fam, split, lang), cases in data.items():
        assert len(cases) == COUNTS[split]
        counts = Counter(c.case.target_option for c in cases)
        assert set(counts) == set(CLASSES[fam])
        assert max(counts.values()) - min(counts.values()) <= 1, (fam, split, lang, counts)


def test_templates_disjoint_and_used_per_split(data):
    for fam in FAMILIES:
        parts = split_templates(fam)
        assert [len(parts[k]) for k in ("train", "heldout", "test")] == [14, 5, 5]
        assert not set(parts["train"]) & set(parts["heldout"]) | set(parts["train"]) & set(parts["test"]) \
            | set(parts["heldout"]) & set(parts["test"])
        for (f, split, _), cases in data.items():
            if f == fam:
                allowed = parts["heldout"] if split in ("dev", "cal_temperature", "cal_policy") else parts[split]
                assert {c.template for c in cases} <= set(allowed)


def test_groups_never_cross_splits_and_counterfactuals(data):
    seen = {}
    for cases in data.values():
        by_group = {}
        for c in cases:
            assert seen.setdefault(c.case.group_id, c.case.split) == c.case.split
            by_group.setdefault(c.case.group_id, []).append(c)
        for g in by_group.values():
            base = next(c for c in g if c.role == "base")
            assert len({c.template for c in g}) == 1
            for c in g:
                if c.role == "decisive":
                    assert c.case.target_option != base.case.target_option
                    assert c.theme_seed == base.theme_seed  # only the decisive content changes
                if c.role == "irrelevant":
                    assert c.case.target_option == base.case.target_option


def test_field_threshold_text_never_holds_total(data):
    for (fam, _, lang), cases in data.items():
        if fam != "field_threshold":
            continue
        for c in cases:
            total, thr = c.meta["total"], c.meta["threshold"]
            assert total != thr
            assert (total >= thr) == (c.case.target_option == "true")
            assert fmt_amount(total, lang) not in c.case.question + c.case.state
            assert f'data-k="total">{fmt_amount(total, lang)}<' in c.body_html


def test_form_edit_reaches_every_label():
    rng = random.Random(1)
    for lang in LANGS:
        for start in FC_CLASSES:
            _, fields = form_content(start, rng, lang)
            assert form_label(fields) == start
            for target in FC_CLASSES:
                assert form_label(form_edit(fields, target, rng)) == target


def test_number_format():
    assert fmt_amount(1234.5, "en") == "1,234.50"
    assert fmt_amount(1234.5, "de") == "1.234,50"
    assert fmt_amount(0.07, "es") == "0,07"
