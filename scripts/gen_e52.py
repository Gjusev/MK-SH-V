"""E52 receipt corpus + bind pool v4 (criteria: docs/E52_CRITERIA_2026-10-29.md).

    .venv/Scripts/python.exe -I scripts/gen_e52.py --dev    # 8-receipt preview
    .venv/Scripts/python.exe -I scripts/gen_e52.py          # full run

Own thermal-receipt renders (Playwright, local, 0 GPU) in three frozen format
strata, riding the e45 threshold-pair and e47 digit-question machinery; then
bind pool v4 = data/e47/bind/rows.jsonl byte-identical + the new receipt kind.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_e38c import fmt_idr, sha256_file, stable_seed  # noqa: E402
from gen_e45 import gap, thresholds  # noqa: E402
from gen_e47 import POS_Q, digits_question  # noqa: E402
from mksh_v.gen_e38a import stable_seed as _unused  # noqa: E402,F401

OUT = ROOT / "data/e52/receipts"
POOL = ROOT / "data/e52/bind"
E47_ROWS = ROOT / "data/e47/bind/rows.jsonl"
E47_SHA = None  # pinned at runtime from the file (asserted stable below)

# ---- frozen design (docs/E52_CRITERIA_2026-10-29.md) ----
STRATA = {"idr": 0.45, "comma": 0.30, "eur": 0.25}   # format families
N_RENDER = {"train_dev": 3000, "fresh": 600}          # 4 rows per render
TOTALS = {"idr": (1e3, 1e8), "comma": (1e1, 1e6), "eur": (1e1, 1e5)}  # log-uniform
UNIT = {"idr": 100.0, "comma": 0.05, "eur": 0.05}     # threshold grid step
CURRENCY = {"idr": "IDR", "comma": "EUR", "eur": "EUR"}
FRESH_SEED_OFFSET = 5_200_000
SHOPS = ("TOKO MAKMUR", "WARUNG SENTOSA", "RITEL NUSANTARA", "CV BERKAH JAYA",
         "TOKO SEDERHANA", "PD MAJU MAPAN", "KOPI KENANGAN", "LAUNDRY KILAT",
         "SAYUR SEGAR", "ELEKTRO ABADI")
ITEMS = ("KOPI SACHET", "GULA 1KG", "BERAS 5KG", "MINYAK 1L", "TELUR 1KG",
         "SABUN MANDI", "MIE INSTAN", "TEH BOX", "SUSU KALENG", "ROKOK 12",
         "AIR MINERAL", "KECAP BOTOL", "SAMPAH PLASTIK", "TISU", "PASTA GIGI",
         "SNACK BOX", "DETERJEN", "SIKAT GIGI", "KERINGAN", "BATERAI AA")
THRESH_Q = {"en": "Is the total on this document at least {t} {u}?",
            "es": "\u00bfEl total de este documento es al menos {t} {u}?",
            "de": "Ist die Gesamtsumme auf diesem Dokument mindestens {t} {u}?"}
STATE = "Scanned document page."


def sample_total(rng: random.Random, stratum: str) -> float:
    lo, hi = TOTALS[stratum]
    v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    if stratum == "idr":
        return float(max(1000, round(v / 100) * 100))
    return round(v, 2)


def fmt_amount(v: float, stratum: str) -> str:
    if stratum == "idr":
        return fmt_idr(v)
    if stratum == "comma":
        s = f"{v:,.2f}"            # 1,234,567.89
        return s.replace(",", "@").replace(".", ",").replace("@", ".")  # 1.234.567,89
    return f"{v:,.2f}"             # 1,234,567.89


def line_items(rng: random.Random, total: float, stratum: str) -> list[tuple[str, float]]:
    n = rng.randint(3, 8)
    if stratum == "idr":
        units = max(1, int(total) // (n * 4))
        w = [max(units, 100) * rng.randint(1, 6) for _ in range(n)]
        s = sum(w)
        w = [round(x * int(total) / s / 100) * 100 for x in w]
        w[-1] += int(total) - sum(w)
        return list(zip(rng.sample(ITEMS, n), w))
    w = [round(total * rng.uniform(0.4, 1.6), 2) for _ in range(n)]
    w = [round(x * total / sum(w), 2) for x in w]
    w[-1] = round(w[-1] + total - sum(w), 2)
    return list(zip(rng.sample(ITEMS, n), w))


def receipt_html(rng: random.Random, shop: str, total: float, stratum: str, seq: int) -> str:
    items = line_items(rng, total, stratum)
    date = f"19{70 + seq % 5}-{1 + seq % 12:02d}-{1 + seq % 28:02d}"
    rows = "".join(
        f"<tr><td>{it}</td><td style='text-align:right'>{fmt_amount(pr, stratum)}</td></tr>"
        for it, pr in items)
    cur = CURRENCY[stratum]
    return (
        "<div id='page' style='width:340px;padding:14px 16px;background:#fff;"
        "font-family:\"DejaVu Sans Mono\",\"DejaVu Sans Condensed\",monospace;color:#111;"
        "font-size:15px;line-height:1.35'>"
        f"<div style='text-align:center;font-size:17px;letter-spacing:2px'>{shop}</div>"
        f"<div style='text-align:center;font-size:12px'>{date} &nbsp; #{seq % 9999:04d}</div>"
        "<div style='border-top:1px dashed #333;margin:8px 0'></div>"
        f"<table style='width:100%;border-collapse:collapse;font-size:13px'>{rows}</table>"
        "<div style='border-top:1px dashed #333;margin:8px 0'></div>"
        "<div style='font-size:19px;font-weight:bold'>TOTAL&nbsp;&nbsp;"
        f"{fmt_amount(total, stratum)} {cur}</div>"
        "<div style='border-top:1px dashed #333;margin:8px 0'></div>"
        "<div style='text-align:center;font-size:11px;color:#333'>TERIMA KASIH</div>"
        "</div>")


def degrade(png: bytes, rng: random.Random) -> bytes:
    """Frozen thermal pipeline: pixelation, blur, contrast, JPEG roundtrip, fade band."""
    from PIL import Image, ImageEnhance, ImageFilter

    im = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = im.size
    f = rng.uniform(0.45, 0.75)
    im = im.resize((max(8, int(w * f)), max(8, int(h * f))), Image.NEAREST)
    im = im.resize((w, h), Image.NEAREST)
    if rng.random() < 0.8:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.2)))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.75, 0.95))
    if rng.random() < 0.30:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=rng.randint(35, 70))
        im = Image.open(buf).convert("RGB")
    if rng.random() < 0.20:
        from PIL import ImageDraw
        y0 = rng.randint(0, max(1, h - 24))
        band = Image.new("RGB", (w, 22), (235, 235, 230))
        im.paste(band, (0, y0))
    return _png_bytes(im)


def _png_bytes(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def digit_row(base_id: str, lang: str, total: float, stratum: str, seq: int):
    """One e47-style digit question on a receipt row (reuses gen_e47 verbatim)."""
    tmp = {"id": base_id, "language": lang, "total": total,
           "img_src": "e41" if stratum == "idr" else "e38a"}
    dq, q, opts, gold = digits_question(tmp)
    if gold not in {o["id"] for o in opts}:  # 8-digit IDR totals exceed e47's n1..n7
        opts = opts + [{"id": gold, "description": gold[1:]}]
    return dq, q, opts, gold


def render_batch(renderer, n: int, seed: int, split_label: str, rows_out: list, stats: dict):
    """Render n receipts, emit 4 rows each (2 threshold + 2 digit)."""
    from mokka_decisions.contracts import Option
    from gen_e41 import case_row

    img_dir = OUT / "receipt"
    img_dir.mkdir(parents=True, exist_ok=True)
    YES = {"en": "Yes", "es": "Si", "de": "Ja"}
    NO = {"en": "No", "es": "No", "de": "Nein"}
    for i in range(n):
        rng = random.Random(stable_seed("e52", seed, i))
        stratum = rng.choices(list(STRATA), weights=list(STRATA.values()))[0]
        lang = ("en", "es", "de")[i % 3]
        total = sample_total(rng, stratum)
        shop = SHOPS[(seed + i) % len(SHOPS)]
        rel = "receipt/r52-%d-%05d.png" % (seed, i)
        if not (OUT / rel).exists():  # deterministic renders: resume skips existing
            body = receipt_html(rng, shop, total, stratum, i)
            png = degrade(renderer.render(body, "DejaVu Sans Mono"), rng)
            (OUT / rel).write_bytes(png)

        pair = "e52-rcp-%s-%d-%05d" % (split_label, seed, i)
        split = "dev" if (i % 10 == 9) else "train"
        if split_label == "fresh":
            split = "test"   # fresh test rows live in their own file (E48 pattern)
        cur, unit = CURRENCY[stratum], UNIT[stratum]
        g = gap(rng)
        lo_t, hi_t = thresholds(total, g, unit)
        stats["strata"][stratum] = stats["strata"].get(stratum, 0) + 4
        nd = len(str(int(total)))
        stats["n_int"].setdefault(stratum, {})
        stats["n_int"][stratum][nd] = stats["n_int"][stratum].get(nd, 0) + 4
        for j, (thr, gold) in enumerate(((lo_t, "true"), (hi_t, "false"))):
            tstr = fmt_amount(thr, stratum)
            q = THRESH_Q[lang].format(t=tstr, u=cur)
            rows_out.append(case_row(
                id="%s-%d" % (pair, j), group_id=pair, source="e52-bind", language=lang,
                domain="documents", state=STATE, question=q,
                options=(Option(id="true", description=YES[lang]),
                         Option(id="false", description=NO[lang])),
                target_option=gold, label_origin="e52-bind", split=split,
                extra={"family": "field_threshold", "bind": "receipt", "pair_id": pair,
                       "img_src": "e52", "image": rel, "total": total,
                       "threshold": thr, "gap": g, "tie": False, "fmt": stratum}))
        for j in range(2):
            dq, q, opts, gold = digit_row("%s-d%d" % (pair, j), lang, total, stratum, i)
            rows_out.append(case_row(
                id="%s-d%d" % (pair, j), group_id=pair, source="e52-bind", language=lang,
                domain="documents", state=STATE, question=q,
                options=tuple(Option(id=o["id"], description=o["description"]) for o in opts),
                target_option=gold, label_origin="e52-bind", split=split,
                extra={"family": "total_digits", "bind": "receipt", "pair_id": pair,
                       "img_src": "e52", "image": rel, "total": total,
                       "digit_q": dq, "fmt": stratum}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="8-receipt preview to artifacts/")
    a = ap.parse_args()

    from gen_e38a import Renderer  # playwright chromium, fonts embedded

    if a.dev:
        out_dir = ROOT / "artifacts/e52-dev"
        out_dir.mkdir(parents=True, exist_ok=True)
        r = Renderer()
        rows = []
        stats = {"strata": {}, "n_int": {}}
        render_batch(r, 8, 991, "dev", rows, stats)
        for row in rows[:8]:
            src = OUT / row["image"]
            (out_dir / Path(row["image"]).name).write_bytes(src.read_bytes())
        r.close()
        print(json.dumps(stats, indent=1))
        return 0

    r = Renderer()
    rows: list[dict] = []
    fresh: list[dict] = []
    stats = {"strata": {}, "n_int": {}}
    render_batch(r, N_RENDER["train_dev"], 5200, "bind", rows, stats)
    render_batch(r, N_RENDER["fresh"], FRESH_SEED_OFFSET, "fresh", fresh, stats)
    r.close()
    (OUT / "fresh_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in fresh), encoding="utf-8")

    # pool v4: E47 rows byte-identical + receipt kind
    e47_sha = sha256_file(E47_ROWS)
    e47_lines = E47_ROWS.read_text(encoding="utf-8").splitlines()
    ids = set()
    for line in e47_lines:
        ids.add(json.loads(line)["id"])
    assert not any(row["id"] in ids for row in rows)
    POOL.mkdir(parents=True, exist_ok=True)
    with (POOL / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for line in e47_lines:
            fh.write(line + "\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "criteria": "docs/E52_CRITERIA_2026-10-29.md",
        "e47_rows_sha256": e47_sha,
        "n_e47_rows": len(e47_lines),
        "n_receipt_rows": len(rows),
        "receipt_share_of_pool": round(len(rows) / (len(e47_lines) + len(rows)), 4),
        "rows_per_render": 4, "renders": N_RENDER,
        "strata_weights": STRATA, "strata_row_counts": stats["strata"],
        "integer_digits_by_stratum": stats["n_int"],
        "totals_range": {k: list(v) for k, v in TOTALS.items()},
        "threshold_units": UNIT, "degradation": "pixelate .45-.75 NEAREST, blur .2-1.2, contrast .75-.95, jpeg q35-70 p=.30, fade band p=.20",
        "receipt_rows_sha256": sha256_file(POOL / "rows.jsonl"),
        "fresh_rows_n": len(fresh), "fresh_rows_sha256": sha256_file(OUT / "fresh_rows.jsonl"),
    }
    (POOL / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
