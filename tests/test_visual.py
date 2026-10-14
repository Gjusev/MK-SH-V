"""VisualCase wrapper: accepts PIL images, rejects anything else."""

from __future__ import annotations

import pytest
from PIL import Image

from mokka_decisions.contracts import DecisionCase, Option

from mksh_v.visual import VisualCase


def make_case():
    return DecisionCase(
        id="v-1", group_id="v-1", source="t", language="en", domain="t",
        state="s", question="q",
        options=(Option(id="a", description="a"), Option(id="b", description="b")),
        target_kind="single", target_option="a",
    )


def test_accepts_pil_rgb_and_none():
    vc = VisualCase(case=make_case(), image=Image.new("RGB", (64, 64)))
    assert vc.image.size == (64, 64)
    assert VisualCase(case=make_case()).image is None


def test_rejects_non_pil_and_bad_mode():
    with pytest.raises(TypeError):
        VisualCase(case=make_case(), image="not-an-image")
    with pytest.raises(ValueError):
        VisualCase(case=make_case(), image=Image.new("RGBA", (8, 8)))


def test_to_dict_has_no_image_bytes():
    vc = VisualCase(case=make_case(), image=Image.new("RGB", (8, 8)), image_ref="imgs/a.png")
    d = vc.to_dict()
    assert d["has_image"] and d["image_ref"] == "imgs/a.png"
    assert "image" not in d  # never inline pixels
