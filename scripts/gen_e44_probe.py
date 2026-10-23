"""E44 probe rows (criteria: docs/E44_PROBE_CRITERIA_2026-10-23.md).

    python scripts/gen_e44_probe.py   # -> data/e44_probe/rows.jsonl (images referenced in place)

Conditions per document, image unchanged: O original, L threshold = total/100
(true), H threshold = total*100 (false), T original + the total stated in the
state. SYN = e38a field_threshold dev base EN/ES/DE (first 120); CORD = e41 cord
dev EN (first 60 receipts, both sides). Image paths are relative to the repo root.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_e38c import DOC_QUESTION, fmt_idr, sha256_file  # noqa: E402
from gen_e41 import case_row  # noqa: E402
from mksh_v.gen_e38a import fmt_amount  # noqa: E402

OUT = ROOT / "data/e44_probe"
STATED = {"en": "The total printed on this document is {t}.",
          "es": "El total impreso en este documento es {t}.",
          "de": "Die auf diesem Dokument gedruckte Gesamtsumme beträgt {t}."}


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    from mokka_decisions.contracts import Option

    rows: list[dict] = []

    def emit(base, cond, doc, kind, image, question, state, gold):
        rows.append(case_row(
            id=f"e44p-{kind}-{doc}-{cond}" + (f"-{base['side']}" if "side" in base and cond in ("O", "T") else ""),
            group_id=f"e44p-{kind}-{doc}", source="e44-probe", language=base["language"],
            domain="documents", state=state, question=question,
            options=tuple(Option(id=o["id"], description=o["description"]) for o in base["options"]),
            target_option=gold, label_origin="e44-probe", split="dev",
            extra={"family": "field_threshold", "image": image, "probe_set": kind, "probe_cond": cond,
                   "probe_doc": doc, "src_id": base["id"]}))

    # SYN: one row per document (base), threshold formatted with fmt_amount in the row's language
    syn = [r for r in load(ROOT / "data/e38a/field_threshold/rows.jsonl")
           if r["split"] == "dev" and r["cf_role"] == "base" and r["language"] in ("en", "es", "de")][:120]
    assert len(syn) == 120, len(syn)
    for r in syn:
        doc, lang, img, tot = r["id"], r["language"], f"data/e38a/{r['image']}", r["total"]
        assert (r["total"] >= r["threshold"]) == (r["target_option"] == "true")
        q = DOC_QUESTION["field_threshold"][lang]
        emit(r, "O", doc, "syn", img, r["question"], r["state"], r["target_option"])
        emit(r, "L", doc, "syn", img, q.format(thr=fmt_amount(tot / 100, lang)), r["state"], "true")
        emit(r, "H", doc, "syn", img, q.format(thr=fmt_amount(tot * 100, lang)), r["state"], "false")
        emit(r, "T", doc, "syn", img, r["question"],
             r["state"] + " " + STATED[lang].format(t=fmt_amount(tot, lang)), r["target_option"])

    # CORD: first 60 dev receipts, English, both original sides
    cord = [r for r in load(ROOT / "data/e41/cord/rows.jsonl") if r["split"] == "dev" and r["language"] == "en"]
    order, by = [], {}
    for r in cord:
        if r["group_id"] not in by:
            order.append(r["group_id"])
        by.setdefault(r["group_id"], []).append(r)
    order = order[:60]
    assert len(order) == 60, len(order)
    for gid in order:
        sides = by[gid]
        assert {s["side"] for s in sides} == {"below", "above"}
        b, img, tot = sides[0], f"data/e41/{sides[0]['image']}", sides[0]["gt_total_value"]
        q = DOC_QUESTION["field_threshold"]["en"].replace("EUR", "IDR")
        for s in sides:
            emit(s, "O", gid, "cord", img, s["question"], s["state"], s["target_option"])
            emit(s, "T", gid, "cord", img, s["question"],
                 s["state"] + " " + STATED["en"].format(t=s["gt_total_text"] + " IDR"), s["target_option"])
        emit(b, "L", gid, "cord", img, q.format(thr=fmt_idr(tot / 100)), b["state"], "true")
        emit(b, "H", gid, "cord", img, q.format(thr=fmt_idr(tot * 100)), b["state"], "false")

    OUT.mkdir(parents=True, exist_ok=True)
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert (ROOT / r["image"]).is_file(), r["image"]
    (OUT / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    n = {k: sum(r["probe_set"] == k for r in rows) for k in ("syn", "cord")}
    print("rows", len(rows), n, "sha256", sha256_file(OUT / "rows.jsonl"))
    for r in rows[:4] + [x for x in rows if x["probe_set"] == "cord"][:6]:
        print(" ", r["probe_cond"], r["target_option"], "|", r["question"][:70], "|", r["state"][-60:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
