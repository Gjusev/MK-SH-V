"""E48 kernel: the E42 calibration kernel on the E47 checkpoints (criteria
docs/E48_CRITERIA_2026-10-25.md).

E42 protocol verbatim with the E46 U1024 input rule on every image, plus the
fresh numeric set (dataset mkshv-e48-fresh, rows sha-pinned) as one more
held-out set, read once.
"""
from __future__ import annotations

import hashlib
import json

import e42_calib as K

FRESH_SLUG = "mkshv-e48-fresh"
FRESH_SHA = "0e5a69cfe1000dde71e042866e142ddf7c022104e3974907fc275bf4592385da"


def fresh_sets() -> dict:
    base = K.dataset_base(FRESH_SLUG)
    root = K.find_dir(base, "rows.jsonl") if base else None
    if root is None:
        raise SystemExit(f"E48 ABORT: dataset {FRESH_SLUG} not attached")
    got = hashlib.sha256((root / "rows.jsonl").read_bytes()).hexdigest()
    if got != FRESH_SHA:
        raise SystemExit(f"E48 ABORT: fresh rows sha {got}")
    K.EXTRA_ROOTS["e48s"] = root
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert len(rows) == 3000 and all(r["id"].startswith("e48s") for r in rows)
    missing = [r["image"] for r in rows if not (root / r["image"]).is_file()]
    if missing:  # fail before any GPU work (v1 ran without its images)
        raise SystemExit(f"E48 ABORT: {len(missing)} fresh images missing, first {missing[0]}")
    K.log(f"fresh set: {len(rows)} rows (sha {got[:16]}), all images present")
    return {"fresh_ft": rows}


if __name__ == "__main__":
    K.SEED_CKPTS = {
        "42": ("mkshv-e47-train", "a1v_e47_s42",
               "8b166fb995069d0f80f95aec9fd759ef47a72fc0cde1338ca453b8adb6d19452"),
        "43": ("mkshv-e47-train", "a1v_e47_s43",
               "afcbeaa45a4c033b5e7d3f033b85fdda2642c0fef78397fcecc5bef53cd83c29"),
    }
    K.UPSCALE_TO = 1024
    K.CRITERIA = "docs/E48_CRITERIA_2026-10-25.md"
    K.SUMMARY_NAME = "e48_summary.json"
    K.extra_sets = fresh_sets
    K.main()
