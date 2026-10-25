"""E47 bind pool v3 (criteria: docs/E47_CRITERIA_2026-10-25.md).

    python scripts/gen_e47.py   # -> data/e47/bind/rows.jsonl (+ manifest)

Every E45 bind row is kept unchanged except the `digits` kind: on the same
images and in the same pair slots, each digits row now asks either the digit
count of the whole-number part (1 in 4) or the digit at a position k drawn
uniformly over the printed total without separators (EUR keeps two decimals).
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_e38c import sha256_file  # noqa: E402
from gen_e45 import DIGIT_Q, WEIGHTS as E45_WEIGHTS, sig_digits  # noqa: E402
from mksh_v.gen_e38a import stable_seed  # noqa: E402

SRC = ROOT / "data/e45/bind/rows.jsonl"
SRC_SHA = "78c82e21d97f868dd38ff5a81f39453a9c521e715099dea46bbfdb6e01783aab"
OUT = ROOT / "data/e47/bind"
POS_Q = {
    "en": "What is digit {k} of the total on this document, counting from the left and ignoring separators?",
    "es": "¿Cuál es la cifra {k} del total de este documento, contando desde la izquierda y sin separadores?",
    "de": "Welche Ziffer steht an Stelle {k} der Gesamtsumme auf diesem Dokument, von links gezählt und ohne Trennzeichen?",
}


def digits_question(row: dict) -> tuple[str, str, list[dict], str]:
    """(digit_q, question, options, gold) for one digits row, deterministic in its id."""
    rng = random.Random(stable_seed("e47-digits", row["id"]))
    lang, total, idr = row["language"], row["total"], row["img_src"] == "e41"
    d = sig_digits(total, idr)
    if rng.random() < 0.25:
        n_int = len(str(int(math.floor(total))))
        options = [{"id": f"n{k}", "description": str(k)} for k in range(1, 8)]
        return "ndigits", DIGIT_Q["ndigits"][lang], options, f"n{n_int}"
    k = rng.randrange(len(d)) + 1
    first = 1 if k == 1 else 0
    options = [{"id": f"d{v}", "description": str(v)} for v in range(first, 10)]
    return f"pos{k}", POS_Q[lang].format(k=k), options, f"d{d[k - 1]}"


def main() -> int:
    if sha256_file(SRC) != SRC_SHA:
        raise SystemExit(f"E47 ABORT: E45 bind rows sha {sha256_file(SRC)}")
    rows = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["bind"] == "digits":
            q_kind, question, options, gold = digits_question(r)
            assert gold in {o["id"] for o in options}, (r["id"], r["total"], q_kind)
            r = dict(r, id="e47-" + r["id"].removeprefix("e45-"), source="e47-bind",
                     label_origin="e47-bind", question=question, options=options,
                     target_option=gold, digit_q=q_kind)
        rows.append(r)

    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)
    by_pair: dict[str, list[dict]] = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    assert all(len(v) == 2 for v in by_pair.values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                    encoding="utf-8")
    dig = [r for r in rows if r["bind"] == "digits"]
    by_q: dict[str, int] = {}
    for r in dig:
        by_q[r["digit_q"]] = by_q.get(r["digit_q"], 0) + 1
    n = {sp: {k: sum(r["split"] == sp and r["bind"] == k for r in rows) for k in E45_WEIGHTS}
         for sp in ("train", "dev")}
    manifest = {"criteria": "docs/E47_CRITERIA_2026-10-25.md", "source_rows_sha256": SRC_SHA,
                "n_rows": n, "digits_by_question": dict(sorted(by_q.items())),
                "rows_sha256": sha256_file(OUT / "rows.jsonl")}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
