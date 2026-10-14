"""Generator tests: label derives from content, variants stay in-split,
rendering is deterministic, and the class is recoverable from structure."""

from __future__ import annotations

import io
import random

from mksh_v.splitting import assert_no_leakage
from mksh_v.synth_doc_type import (
    ALL_TEMPLATES,
    DOC_TYPES,
    derive_doc_type,
    generate_doc_type_cases,
    make_content,
    render_document,
)


def png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_label_derives_from_content_for_every_type():
    for doc_type in DOC_TYPES:
        for seed in range(8):
            content = make_content(doc_type, random.Random(seed + 991))
            assert derive_doc_type(content) == doc_type


def test_decisive_change_flips_label_irrelevant_change_does_not():
    rng = random.Random(7)
    content = make_content("invoice", rng)
    # irrelevant: same content re-rendered with another style seed
    a = render_document(content, "invoice-classic", 1)
    b = render_document(content, "invoice-classic", 2)
    assert derive_doc_type(content) == "invoice"  # label unchanged
    assert png_bytes(a) != png_bytes(b)  # pixels did change
    # decisive: removing the invoice-only reference flips the derived label
    stripped = type(content)(**{**content.__dict__, "reference": ""})
    assert derive_doc_type(stripped) == "receipt"


def test_rendering_is_deterministic_per_seed():
    rng = random.Random(3)
    content = make_content("contract", rng)
    assert png_bytes(render_document(content, "contract-boxed", 11)) == \
        png_bytes(render_document(content, "contract-boxed", 11))
    assert png_bytes(render_document(content, "contract-boxed", 11)) != \
        png_bytes(render_document(content, "contract-boxed", 12))


def test_generated_cases_split_by_template_before_variants():
    templates = ALL_TEMPLATES[:6]
    cases = generate_doc_type_cases(templates, n=12, seed=5, split="train", variants_per_case=2)
    assert len(cases) == 24
    assert all(c["case"].split == "train" for c in cases)
    assert all(c["template"] in templates for c in cases)
    by_group = assert_no_leakage([c["case"] for c in cases])  # must not raise
    assert set(by_group["train"]) <= set(templates)
    # variants of one content share group_id (same template) and target
    pairs = {}
    for c in cases:
        base = c["case"].id.rsplit("-v", 1)[0]
        pairs.setdefault(base, []).append(c)
    assert all(len(v) == 2 for v in pairs.values())
    assert all(
        v[0]["case"].target_option == v[1]["case"].target_option for v in pairs.values()
    )


def test_train_and_dev_template_splits_are_disjoint():
    train_templates = ALL_TEMPLATES[:12]
    dev_templates = ALL_TEMPLATES[12:]
    train = generate_doc_type_cases(train_templates, n=24, seed=1, split="train")
    dev = generate_doc_type_cases(dev_templates, n=6, seed=2, split="dev")
    assert not (set(c["template"] for c in train) & set(c["template"] for c in dev))
    assert_no_leakage([c["case"] for c in train + dev])


def test_other_class_has_a_wide_content_space():
    # 334 train + 67 dev draws of "other" must not collapse to a handful of
    # verbatim contents (template-split dev would then be content-duplicate)
    seen = set()
    for seed in range(120):
        c = make_content("other", random.Random(seed + 5000))
        seen.add((c.title, c.paragraphs))
    assert len(seen) >= 100  # wide, not 18 fixed posters


def test_state_carries_no_label_information():
    cases = generate_doc_type_cases(ALL_TEMPLATES[:6], n=6, seed=9, split="train")
    states = {c["case"].state for c in cases}
    questions = {c["case"].question for c in cases}
    assert len(states) == 1 and len(questions) == 1  # constant across classes
