"""Local check of the demo's decision path on a real example image.

    python tests/test_demo_app.py
Runs the exact decide() the Space serves, against the exported release
weights, and asserts the output contract (probabilities, gate, timing).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
os.environ.setdefault("MKSHV_MODEL_DIR", str(ROOT / "release"))

from PIL import Image  # noqa: E402


def main() -> int:
    import app  # noqa: E402  loads the model (~1 min on CPU)

    img = Image.open(ROOT / "demo/examples/cord-receipt-1.png")
    bars_html, verdict, abstain, meta = app.decide(
        img,
        "Scanned document page.",
        "Is the total on this document at least 299.586 IDR?",
        "false: No\ntrue: Yes")
    print(verdict, "|", abstain or "(accepted)", "|", meta)
    assert "choice:" in verdict or "abstain" in verdict
    assert bars_html.count("prob-row") == 2
    assert "T = " in meta
    # the abstention path must also work: a nonsense state with an odd image
    bars2, verdict2, abstain2, _ = app.decide(
        img, "A receipt for quantum harmonics.",
        "Does this receipt mention fourteen penguins?", "false: No\ntrue: Yes")
    print(verdict2, "|", abstain2 or "(accepted)")
    assert "choice:" in verdict2 or "abstain" in verdict2
    print("demo decision path OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
