"""Amber Checkpoint days generator (spec: docs/internal/AMBER_CHECKPOINT_SPEC.md).

    python scripts/gen_amber_checkpoint.py
      -> demo/game/days.json + demo/game/assets/docs/d{day}-{i}.png

4 days x 8 documents rendered by the E38 renderer (Playwright + embedded
fonts, the machinery of scripts/gen_e38a.py) with a NEW worn-paper theme.
Additive: the E38 generators and their outputs are untouched. Discrepancies
are injected programmatically here; gold labels live in days.json and never
reach the model input. Day seeds fixed: a day replays identically.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_e38a import Renderer, git_commit, sha256  # noqa: E402
from mksh_v.gen_e38a import stable_seed  # noqa: E402

OUT = ROOT / "demo/game"
DOCS = OUT / "assets/docs"
FONTS = OUT / "assets/fonts"
SEAL_PNG = ROOT / "tmp/amber-art/seal_entry_48_v3.png"

# The shipped policy (release/inference-policy.json, E48): one constants block,
# copied nowhere else in the game.
POLICY = {"checkpoint": "a1v_e47_s43", "checkpoint_sha256":
          "afcbeaa45a4c033b5e7d3f033b85fdda2642c0fef78397fcecc5bef53cd83c29",
          "T": 1.124, "tau": 0.55, "coverage": 0.9855, "cp95_upper": 0.0406}

MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
SURNAMES = ("HALDER", "VOSS", "EDRIC", "MARLOW", "KESSLER", "BRANDT", "OKTAY", "REZNICK",
            "THALER", "GRIMM", "VANCE", "OSTER", "LINHARD", "MERCER", "KOVAC", "ALBRECHT",
            "STERN", "FALK", "DRAEGER", "JUNOT")
INITIALS = "ABCDEFGHIJKLMNOPRSTVW"

# The campaign week. Day 3 matches demo/game/mock.html (27 OCT 1974, day-1 slip
# pinned 24 OCT, day-2 slips pinned 25 OCT); the 26th was the booth's rest day.
DAYS = {
    1: {"date": "24 OCT 1974", "today": dt.date(1974, 10, 24)},
    2: {"date": "25 OCT 1974", "today": dt.date(1974, 10, 25)},
    3: {"date": "27 OCT 1974", "today": dt.date(1974, 10, 27)},
    4: {"date": "28 OCT 1974", "today": dt.date(1974, 10, 28)},
}

# Rule slips, verbatim from the frozen mock (days 1-3); "· NEW", never an em dash.
SLIPS = {
    1: [("Day 1", "Veldanian passports only. All other issuers: deny at the gate.", "PINNED 24 OCT")],
    2: [("Day 2 · A", "Foreigners must hold a valid work permit. No permit, no entry.", "PINNED 25 OCT"),
        ("Day 2 · B", "Holder must appear on today's manifest. Unlisted names: deny.", "PINNED 25 OCT")],
    3: [("Day 3 · NEW", "Expired documents: deny. Expiry must fall on or after today's date.", "PINNED 27 OCT")],
    4: [("Day 4 · NEW", "Stays over 30 days are not permitted at this checkpoint.", "PINNED 28 OCT"),
        ("Day 4 · NEW", "Declared currency capped at 5,000 VEL. Higher amounts: deny.", "PINNED 28 OCT")],
}

# Rule clauses as the model reads them (the operative rules, seal included from
# day 2 as the mandatory-field rule behind the entry-seal check).
RULE_CLAUSES = {
    1: ["Veldanian passports only; all other issuers are denied"],
    2: ["Veldanian passports only; all other issuers are denied",
        "foreigners must hold a valid work permit",
        "the holder must be on today's manifest",
        "the entry seal must be present on the document"],
    3: ["Veldanian passports only; all other issuers are denied",
        "foreigners must hold a valid work permit",
        "the holder must be on today's manifest",
        "the entry seal must be present on the document",
        "the document must be unexpired (expiry on or after today)"],
    4: ["Veldanian passports only; all other issuers are denied",
        "foreigners must hold a valid work permit",
        "the holder must be on today's manifest",
        "the entry seal must be present on the document",
        "the document must be unexpired (expiry on or after today)",
        "stays over 30 days are not permitted",
        "the declared amount is capped at 5,000 VEL"],
}

# Check definitions. family = the measured capability it plays (spec table).
CHECKS = {
    "type":    {"family": "doc_type",       "q": "Is this a Veldanian passport, issued by Veldania?"},
    "permit":  {"family": "form_complete",  "q": "Does the holder hold a valid work permit?"},
    "seal":    {"family": "form_complete",  "q": "Is the entry seal present on the document?"},
    "manifest": {"family": "source_conflict", "q": "Is the holder on today's manifest?"},
    "expiry":  {"family": "field_threshold", "q": "Is the document unexpired (expiry on or after {date})?"},
    "stay":    {"family": "field_threshold", "q": "Is the declared stay 30 days or fewer?"},
    "amount":  {"family": "field_threshold", "q": "Is the declared amount within the 5,000 VEL cap?"},
}

# Scenario table: (issuer, scenario dict). Explicit and fixed, so a day replays
# identically and every discrepancy type appears across the campaign.
# vel = Veldanian, kor = Korvenian (foreigner).
SCENARIOS = {
    1: [("vel", {}), ("vel", {}), ("kor", {}), ("vel", {}), ("kor", {}), ("vel", {}), ("kor", {}), ("vel", {})],
    2: [("vel", {"seal": False}), ("kor", {"permit": True, "listed": True}),
        ("kor", {"permit": False, "listed": True}), ("vel", {"listed": False}), ("vel", {}),
        ("vel", {}), ("vel", {}), ("vel", {})],
    3: [("vel", {"expiry_delta": -3}), ("vel", {}), ("vel", {"expiry_delta": -210}),
        ("kor", {"permit": True, "listed": True}), ("vel", {"listed": False}), ("vel", {}),
        ("vel", {"seal": False}), ("vel", {})],
    4: [("vel", {"stay": 32}), ("vel", {}), ("vel", {"amount": 6100}),
        ("vel", {"stay": 45}), ("vel", {}), ("vel", {"expiry_delta": -2}), ("vel", {}), ("vel", {})],
}


def _fmt_date(d: dt.date) -> str:
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def _name(rng: random.Random) -> str:
    return f"{rng.choice(SURNAMES)}, {rng.choice(INITIALS)}."


def _issued(rng: random.Random) -> tuple[str, dt.date]:
    d = dt.date(rng.randint(1966, 1973), rng.randint(1, 12), rng.randint(1, 28))
    return _fmt_date(d), d


def _expiry(rng: random.Random, delta: int) -> tuple[str, dt.date, int]:
    """Expired by ``delta`` days, sometimes a bit more (the tense vs blatant mix)."""
    d = dt.date(1974, 10, 24) + dt.timedelta(days=delta - rng.choice((0, 0, 30)))
    return _fmt_date(d), d, (d - dt.date(1974, 10, 24)).days


def make_doc(day: int, i: int, issuer: str, sc: dict) -> dict:
    """One document with injected discrepancies; gold derived from the content."""
    did = f"d{day}-{i}"
    rng = random.Random(stable_seed("amber", did))
    foreign = issuer == "kor"
    listed = sc.get("listed", True)
    stay = sc.get("stay", rng.choice((7, 14, 21, 28)))
    amount = sc.get("amount", rng.choice((300, 900, 1200, 2400, 3800)))
    issued_s, _ = _issued(rng)
    if "expiry_delta" in sc:
        exp_s, exp_d, exp_delta = _expiry(rng, sc["expiry_delta"])
    else:
        base = dt.date(1974, 10, 24) + dt.timedelta(days=rng.choice((14, 60, 180, 400)))
        exp_s, exp_d, exp_delta = _fmt_date(base), base, (base - dt.date(1974, 10, 24)).days
    permit = sc.get("permit") if foreign else None  # None = no permit row printed
    seal = sc.get("seal", True)
    doc = {
        "id": did, "issuer": "REPUBLIC OF " + ("VELDANIA" if issuer == "vel" else "KORVENIA"),
        "nationality": "VELDANIA" if issuer == "vel" else "KORVENIA",
        "holder": _name(rng), "doc_no": f"{'VD' if issuer == 'vel' else 'KV'}-{rng.randint(1000, 9999)}-{rng.randint(10, 99)}",
        "series": rng.choice("ABCD"), "issued": issued_s, "expires": exp_s,
        "expiry_iso": exp_d.isoformat(), "expiry_delta": exp_delta,
        "stay_days": stay, "declared": amount, "work_permit": permit, "permit_no":
            f"WP-{rng.randint(1000, 9999)}" if permit else "",
        "seal": seal, "manifest_listed": listed,
        "image": f"assets/docs/{did}.png",
    }
    # gold: violations of the day's active rules (never sent to the model)
    v: list[str] = []
    if issuer != "vel":
        v.append("issuer")
    if day >= 2:
        if foreign and not permit:
            v.append("permit")
        if not seal:
            v.append("seal")
        if not listed:
            v.append("manifest")
    if day >= 3 and exp_d < DAYS[day]["today"]:
        v.append("expiry")
    if day >= 4:
        if stay > 30:
            v.append("stay")
        if amount > 5000:
            v.append("amount")
    doc["violations"] = v
    doc["verdict"] = "deny" if v else "approve"
    return doc


def gold_of(kind: str, doc: dict, day: int) -> str:
    return {
        "type": "yes" if doc["issuer"].endswith("VELDANIA") else "no",
        "permit": "yes" if doc["work_permit"] else "no",
        "seal": "yes" if doc["seal"] else "no",
        "manifest": "yes" if doc["manifest_listed"] else "no",
        "expiry": "yes" if dt.date.fromisoformat(doc["expiry_iso"]) >= DAYS[day]["today"] else "no",
        "stay": "yes" if doc["stay_days"] <= 30 else "no",
        "amount": "yes" if doc["declared"] <= 5000 else "no",
    }[kind]


def checks_for(day: int, doc: dict, state: str) -> list[dict]:
    kinds = ["type"]
    if day >= 2:
        kinds.append("manifest")
        kinds.append("permit" if doc["nationality"] == "KORVENIA" else "seal")
    if day >= 3:
        kinds.append("expiry")
    if day >= 4:
        kinds += ["stay", "amount"]
    return [{"id": f"{doc['id']}-c{k}", "kind": k, **CHECKS[k],
             "question": CHECKS[k]["q"].format(date=DAYS[day]["date"]),
             "state": state, "gold": gold_of(k, doc, day)} for k in kinds]


def plan() -> dict:
    """The full campaign, deterministic (no rendering here)."""
    days = []
    for day in (1, 2, 3, 4):
        rng = random.Random(stable_seed("amber", "day", day))
        docs = [make_doc(day, i + 1, iss, sc) for i, (iss, sc) in enumerate(SCENARIOS[day])]
        listed = [d["holder"] for d in docs if d["manifest_listed"]]
        pool = [n for n in (_name(rng) for _ in range(64)) if n not in {d["holder"] for d in docs}]
        manifest = listed + [n for n in pool if n not in listed][: 8 - len(listed)]
        rng.shuffle(manifest)
        state = (f"Day {day} at Amber Checkpoint, {DAYS[day]['date']}. Rules of entry: "
                 + "; ".join(RULE_CLAUSES[day]) + ". "
                 + f"Manifest ({len(manifest)} names): " + "; ".join(manifest) + ".")
        slips = [s for d in range(1, day + 1) for s in SLIPS[d]]
        for d in docs:
            d["checks"] = checks_for(day, d, state)
        days.append({"day": day, "date": DAYS[day]["date"], "today": DAYS[day]["today"].isoformat(),
                     "slips": [{"title": t, "text": x, "pinned": p} for t, x, p in slips],
                     "manifest": manifest, "state": state, "docs": docs})
    return {"policy": POLICY, "days": days}


# ------------------------------------------------------------------ render ---
def font_css() -> str:
    css = []
    for f in sorted(FONTS.glob("*.ttf")):
        fam = ("'Special Elite'" if "Special" in f.name else
               "'IBM Plex Sans'" if "PlexSans" in f.name else "'IBM Plex Mono'")
        weight = 600 if "SemiBold" in f.name else 400
        css.append(f"@font-face{{font-family:{fam};font-weight:{weight};"
                   f"src:url(data:font/ttf;base64,{base64.b64encode(f.read_bytes()).decode()})}}")
    return "<style>" + "".join(css) + "</style>"


GRAIN = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E"
         "%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E"
         "%3C/filter%3E%3Crect width='120' height='120' filter='url(%23n)' opacity='0.5'/%3E%3C/svg%3E")


def worn_theme(seed: int, issuer: str) -> dict:
    """The new worn-paper theme: Veldania's olive cream, Korvenia's grey, per-doc wear."""
    r = random.Random(seed)
    return {"paper": "#EFE7D2" if issuer == "vel" else "#E0DED4",
            "ink": "#23251D", "accent": "#6B6A45" if issuer == "vel" else "#5C5C66",
            "pad": r.randint(20, 26), "dx": r.randint(-3, 3), "dy": r.randint(-2, 2),
            "grain": round(r.uniform(0.035, 0.05), 3), "bust": r.randrange(3)}


def permit_html(doc: dict, seal_b64: str | None) -> str:
    t = worn_theme(stable_seed("amber", doc["id"], "wear"), "vel" if doc["nationality"] == "VELDANIA" else "kor")
    fields = [("HOLDER", doc["holder"], "holder"), ("NATIONALITY", doc["nationality"], "nationality"),
              ("ISSUED", doc["issued"], "issued"), ("EXPIRES", doc["expires"], "expires"),
              ("STAY", f"{doc['stay_days']} DAYS", "stay"),
              ("DECLARED", f"{doc['declared']:,} VEL", "declared")]
    if doc["nationality"] == "KORVENIA":
        fields.append(("WORK PERMIT", doc["permit_no"] or "NONE", "permit"))
    fhtml = "".join(f'<div class="f"><label>{k}</label><span data-k="{dk}">{v}</span></div>' for k, v, dk in fields)
    seal = (f'<img class="seal" src="data:image/png;base64,{seal_b64}" alt="entry seal">' if doc["seal"]
            else '<div class="noseal" data-k="seal-missing"></div>')
    return f"""{font_css()}<style>
#page{{width:460px;box-sizing:border-box;padding:{t['pad'] + t['dy']}px {t['pad'] - t['dx']}px {t['pad'] - t['dy']}px {t['pad'] + t['dx']}px;
 background:{t['paper']};color:{t['ink']};font-family:'IBM Plex Sans';position:relative}}
#page::after{{content:"";position:absolute;inset:0;background-image:url("{GRAIN}");opacity:{t['grain']};pointer-events:none}}
#page header{{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:2px solid {t['ink']};padding-bottom:10px;margin-bottom:14px}}
#page .rep{{font-family:'Special Elite';font-size:19px;line-height:1.15}}
#page .rep small{{display:block;font-family:'IBM Plex Sans';font-weight:600;font-size:11.5px;letter-spacing:1.6px;color:#4A4534;margin-top:3px}}
#page .no{{font-family:'IBM Plex Mono';font-size:11px;color:#4A4534;text-align:right;line-height:1.5}}
#page .body{{display:flex;gap:18px}}
#page .portrait{{width:56px;height:70px;border:2px solid {t['ink']};background:#E6D8B9;flex:none;position:relative;overflow:hidden}}
#page .portrait i,#page .portrait b{{background:#857C5F;
 -webkit-mask-image:repeating-linear-gradient(45deg,#000 0 1px,transparent 1px 2px);mask-image:repeating-linear-gradient(45deg,#000 0 1px,transparent 1px 2px)}}
#page .portrait i{{position:absolute;left:{19 + t['bust']}px;top:13px;width:18px;height:18px;border-radius:50%}}
#page .portrait b{{position:absolute;left:{7 - t['dx'] // 2}px;top:35px;width:42px;height:28px;border-radius:13px 13px 0 0}}
#page .portrait::before{{content:"PHOTO";position:absolute;left:-7px;top:30px;width:74px;text-align:center;background:{t['ink']};
 color:{t['paper']};font-family:'IBM Plex Mono';font-size:9px;letter-spacing:3px;padding:2px 0;transform:rotate(-18deg);z-index:2}}
#page .fields{{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:9px 14px}}
#page .f label{{display:block;font-size:9.5px;font-weight:600;letter-spacing:1.4px;color:#4A4534}}
#page .f span{{font-family:'IBM Plex Mono';font-weight:600;font-size:13.5px}}
#page .manifest{{margin-top:14px;border-top:1px solid rgba(35,37,29,.35);padding-top:9px;font-family:'IBM Plex Mono';
 font-size:11px;color:#4A4534;letter-spacing:.3px}}
#page .foot{{display:flex;justify-content:space-between;align-items:flex-end;margin-top:16px}}
#page .seal{{width:96px;height:96px;transform:rotate(-8deg)}}
#page .noseal{{width:96px;height:96px;border:2px dashed rgba(35,37,29,.25)}}
#page .serial{{font-family:'IBM Plex Mono';font-size:10px;color:#4A4534;letter-spacing:1px;text-align:right}}
</style>
<div id="page" data-k="permit" data-issuer="{doc['issuer']}">
 <header><div class="rep">{doc['issuer']}<small>TEMPORARY ENTRY PERMIT</small></div>
  <div class="no">NO. {doc['doc_no']}<br>SERIES {doc['series']}</div></header>
 <div class="body"><div class="portrait"><i></i><b></b></div><div class="fields">{fhtml}</div></div>
 <div class="manifest">MANIFEST REF {doc['id'].split('-')[0].upper()} · BOOTH 04 · RETAIN FOR INSPECTION</div>
 <div class="foot">{seal}<div class="serial">PRINTED BY THE CHANCELLERY<br>OF {doc['issuer']}</div></div>
</div>"""


def render_all(campaign: dict) -> dict:
    """Render every document PNG with the E38 renderer; returns sha256 per image."""
    seal_b64 = base64.b64encode(SEAL_PNG.read_bytes()).decode() if SEAL_PNG.exists() else None
    r = Renderer()
    shas = {}
    try:
        first = campaign["days"][0]["docs"][0]
        r.render(permit_html(first, seal_b64), "IBM Plex Mono")  # warm the embedded fonts once
        r.page.evaluate("() => document.fonts.ready.then(() => true)")
        for day in campaign["days"]:
            for doc in day["docs"]:
                png = r.render(permit_html(doc, seal_b64), "IBM Plex Mono")
                p = DOCS / f"{doc['id']}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(png)
                shas[doc["image"]] = sha256(png)
        print(f"rendered {len(shas)} documents -> {DOCS}")
    finally:
        r.close()
    return shas


def main() -> int:
    campaign = plan()
    shas = render_all(campaign)
    campaign["generator"] = {"commit": git_commit(), "fonts_sha256":
                             {f.name: sha256(f.read_bytes()) for f in sorted(FONTS.glob("*.ttf"))},
                             "images_sha256": shas}
    (OUT / "days.json").write_text(json.dumps(campaign, indent=1, ensure_ascii=False), encoding="utf-8")
    n_checks = sum(len(d["checks"]) for day in campaign["days"] for d in day["docs"])
    print(json.dumps({"days": len(campaign["days"]), "docs": sum(len(d["docs"]) for d in campaign["days"]),
                      "checks": n_checks,
                      "verdicts": {f"d{day['day']}": [d["verdict"][0] for d in day["docs"]] for day in campaign["days"]}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
