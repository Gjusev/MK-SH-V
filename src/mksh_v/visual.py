"""Visual decision case: the MK-SH-1 contract plus an optional image.

The contract (`mokka_decisions.contracts.DecisionCase`) stays text-only and
untouched; visual evidence rides alongside it. `state` never carries label
information for generated visual families (a constant scan note there), so
the label is only derivable from the rendered image content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mokka_decisions.contracts import DecisionCase


@dataclass(frozen=True)
class VisualCase:
    """One decision problem with optional image evidence."""

    case: DecisionCase
    image: Any | None = None  # PIL.Image.Image when present; None = text-only
    image_ref: str = ""  # path/URI for manifests; never inline bytes

    def __post_init__(self) -> None:
        if self.image is None and not self.image_ref:
            pass  # text-only case, fine
        if self.image is not None:
            # PIL import kept local so the contract layer stays dependency-free
            from PIL.Image import Image as PILImage

            if not isinstance(self.image, PILImage):
                raise TypeError(
                    f"image must be a PIL.Image.Image, got {type(self.image).__name__}"
                )
            if self.image.mode not in ("RGB", "L"):
                raise ValueError(f"image mode {self.image.mode!r} not in ('RGB', 'L')")

    def to_dict(self) -> dict:
        return {
            "case": self.case.to_dict(),
            "image_ref": self.image_ref,
            "has_image": self.image is not None,
        }
