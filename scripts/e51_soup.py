"""E51 Arm C Stage A: build the uniform weight-average soup of the two pinned E47
checkpoints and the 200-id smoke rows file.

    .venv/Scripts/python.exe -I scripts/e51_soup.py

Prints the soup sha256 and the smoke rows sha256 (recorded in the E51 results).
"""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "checkpoints/e51"
PINS = {"s42": "8b166fb995069d0f80f95aec9fd759ef47a72fc0cde1338ca453b8adb6d19452",
        "s43": "afcbeaa45a4c033b5e7d3f033b85fdda2642c0fef78397fcecc5bef53cd83c29"}


def main() -> None:
    for name, pin in PINS.items():
        got = hashlib.sha256((CK / f"best_{name}.pt").read_bytes()).hexdigest()
        assert got == pin, (name, got)
    a = torch.load(CK / "best_s42.pt", map_location="cpu", weights_only=True)
    b = torch.load(CK / "best_s43.pt", map_location="cpu", weights_only=True)
    assert set(a["model"].keys()) == set(b["model"].keys())
    soup = {k: (a["model"][k].float() + b["model"][k].float()) / 2 for k in a["model"]}
    out = CK / "soup.pt"
    torch.save({"model": soup}, out)
    print("soup sha256:", hashlib.sha256(out.read_bytes()).hexdigest())

    ids = sorted(json.loads(l)["id"] for l in (ROOT / "runs/e48/percase_s42__cal_temperature.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    smoke = set(ids[:200])
    rows = [json.loads(l) for l in (ROOT / "data/e38a/doc_type/rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sub = [r for r in rows if r["id"] in smoke]
    assert len(sub) == 200, len(sub)
    p = CK / "smoke_rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in sub), encoding="utf-8")
    print("smoke rows:", len(sub), "sha256:", hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
