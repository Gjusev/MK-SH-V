"""E46 input rule: enlarge small images to the longest side, leave large ones alone."""
from PIL import Image

from mksh_v.visual import upscale_to_side


def test_small_document_fills_longest_side_and_keeps_aspect():
    out = upscale_to_side(Image.new("RGB", (794, 453)), 1024)
    assert max(out.size) == 1024
    assert abs(out.size[0] / out.size[1] - 794 / 453) < 0.01


def test_portrait_receipt_upscales_on_height():
    assert upscale_to_side(Image.new("RGB", (380, 485)), 1536).size[1] == 1536


def test_large_or_disabled_is_identity():
    big = Image.new("RGB", (1080, 2400))
    assert upscale_to_side(big, 1024) is big  # the processor's max_side still caps it
    small = Image.new("RGB", (100, 50))
    assert upscale_to_side(small, None) is small
    assert upscale_to_side(None, 1024) is None
