"""Materialise demo/cases.json: copy each picked image into demo/examples/
(resized, long side <= 900 px, like the existing examples) and point the case
at it. Run after pick_demo_cases.py.

    python assets/src/materialize_cases.py
"""
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
cases = json.loads((ROOT / "demo/cases.json").read_text(encoding="utf-8"))
order = {"syn": 0, "cord": 1, "gui": 2}
cases.sort(key=lambda c: (order[c["kind"]], c["verdict"] != "answer", c["family"]))
for i, c in enumerate(cases, 1):
    dst = ROOT / f"demo/examples/case-{i:02d}.png"
    im = Image.open(ROOT / c["src"]).convert("RGB")
    im.thumbnail((900, 900), Image.LANCZOS)
    im.save(dst, optimize=True)
    c["image"] = f"examples/case-{i:02d}.png"
    del c["src"]
json.dump(cases, open(ROOT / "demo/cases.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(cases), "cases;", sum(c["verdict"] == "answer" for c in cases), "answers,",
      sum(c["verdict"] == "abstain" for c in cases), "refusals")
