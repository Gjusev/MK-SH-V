"""E53 kernel: the E42 calibration kernel on the E52 checkpoints (criteria
docs/E53_CRITERIA_2026-10-30.md).

E42 protocol verbatim with the U1024 input rule, plus two extra held-out sets
read once: the E48 fresh numeric set (fresh_ft) and the E52 receipt fresh set
(dataset mkshv-e52-fresh, rows sha-pinned).
"""
from __future__ import annotations

import hashlib
import json

import e42_calib as K

FT_SLUG = "mkshv-e48-fresh"
FT_SHA = "0e5a69cfe1000dde71e042866e142ddf7c022104e3974907fc275bf4592385da"
RC_SLUG = "mkshv-e52-fresh"
RC_SHA = "398d367019118b7a40911cfd70e7db5f8ef75e1efbe4d98e0aa50d683042bf73"


def fresh_sets() -> dict:
    base = K.dataset_base(FT_SLUG)
    root = K.find_dir(base, "rows.jsonl") if base else None
    if root is None:
        raise SystemExit(f"E53 ABORT: dataset {FT_SLUG} not attached")
    if hashlib.sha256((root / "rows.jsonl").read_bytes()).hexdigest() != FT_SHA:
        raise SystemExit("E53 ABORT: fresh_ft rows sha mismatch")
    K.EXTRA_ROOTS["e48s"] = root
    ft = [json.loads(l) for l in (root / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    base2 = K.dataset_base(RC_SLUG)
    root2 = K.find_dir(base2, "rows.jsonl") if base2 else None
    if root2 is None:
        raise SystemExit(f"E53 ABORT: dataset {RC_SLUG} not attached")
    got = hashlib.sha256((root2 / "rows.jsonl").read_bytes()).hexdigest()
    if got != RC_SHA:
        raise SystemExit(f"E53 ABORT: receipt fresh rows sha {got}")
    K.EXTRA_ROOTS["e52-rcp"] = root2
    rc = [json.loads(l) for l in (root2 / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rc) == 2400 and all(r["id"].startswith("e52-rcp") for r in rc)
    missing = [r["image"] for r in rc if not (root2 / r["image"]).is_file()]
    if missing:
        raise SystemExit(f"E53 ABORT: {len(missing)} receipt fresh images missing, first {missing[0]}")
    K.log(f"extra sets: fresh_ft {len(ft)}, receipt fresh {len(rc)} (sha {got[:16]}), images present")
    return {"fresh_ft": ft, "fresh_receipt": rc}


if __name__ == "__main__":
    K.SEED_CKPTS = {
        "42": ("mkshv-e52-train", "a1v_e52_s42",
               "3d2ffdbcfe6eb9e851716079796cf9a2a04e1e0e90f30b9ffacecf66093c9aa3"),
        "43": ("mkshv-e52-train", "a1v_e52_s43",
               "dfc6d60e15247c9e4b5902f6e84cbca9c70eba8a5489a06a580b6b43330de37d"),
    }
    K.UPSCALE_TO = 1024
    K.CRITERIA = "docs/E53_CRITERIA_2026-10-30.md"
    K.SUMMARY_NAME = "e53_summary.json"
    K.extra_sets = fresh_sets
    K.main()
