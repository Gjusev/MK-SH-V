"""Amber Checkpoint days generator v2 (spec: docs/internal/AMBER_CHECKPOINT_SPEC.md
+ dated correction; pivot A, orchestrator GO 2026-10-27).

    python scripts/gen_amber_checkpoint.py
      -> demo/game/days.json + demo/game/assets/docs/d{day}-{i}.png

4 days x 8 documents rendered by the E38 renderer with the NEW worn-paper
theme (additive: E38 generators and outputs untouched). After the pivot the
booth is a customs desk and every check is a measured capability, played in
its measured framing: doc_type (6-way, verbatim question and options),
form_complete (E38 forms, verbatim), source_conflict (sender name read from
the paper, goal-match style options), field_threshold (total vs the
5,000.00 VEL cap, decimals). No date check: date arithmetic was never
measured. Gold labels live here and never reach the model input. Day seeds
fixed: a day replays identically.
"""
from __future__ import annotations

import base64
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_e38a import Renderer, git_commit, sha256  # noqa: E402
from mksh_v.gen_e38a import (COMPANIES, OPTION_TEXT, amount_doc, doc_content, form_content,  # noqa: E402
                             form_label, render_doc, render_form, reprice, stable_seed)
from dataclasses import replace  # noqa: E402

OUT = ROOT / "demo/game"
DOCS = OUT / "assets/docs"
FONTS = OUT / "assets/fonts"

# The shipped policy (release/inference-policy.json, E48): one constants block.
POLICY = {"checkpoint": "a1v_e47_s43",
          "T": 1.124, "tau": 0.55, "coverage": 0.9855, "cp95_upper": 0.0406}

DAYS = {1: "24 OCT 1974", 2: "25 OCT 1974", 3: "27 OCT 1974", 4: "28 OCT 1974"}

SD = "Scanned document page."  # the measured state of all four families

# Rule slips, English (mock layout; days 1-3 wording follows the frozen mock's
# register). Day 4 is a spot-inspection notice, not a new rule: difficulty
# comes from finer threshold gaps, which is the measured hard mode.
SLIPS = {
    1: [("Day 1", "Official declarations only. Private receipts, letters and notes: deny at the gate.", "PINNED 24 OCT")],
    2: [("Day 2 · A", "Every declaration form must be complete. Required fields are marked with *.", "PINNED 25 OCT"),
        ("Day 2 · B", "The trading company must appear on today's manifest. Unlisted companies: deny.", "PINNED 25 OCT")],
    3: [("Day 3 · NEW", "Declared value capped at 5,000.00 VEL. Totals at or above the cap: deny.", "PINNED 27 OCT")],
    4: [("Day 4 · NEW", "Spot inspection. Margins near the cap will be read to the cent.", "PINNED 28 OCT")],
}

CAP = 5000.00
# totals per scenario class: within/over the cap, wide (D3) and fine (D4) gaps
WIDE_IN = (312.50, 812.50, 1850.00, 3420.00, 4100.00)
WIDE_OVER = (6400.00, 8250.00)
FINE_IN = (4650.00, 4895.00, 4995.00)
FINE_OVER = (5015.00, 5180.00, 5250.00)

DOC_TYPES = ("invoice", "receipt", "contract", "form", "letter", "other")
TYPE_OPTIONS = [{"id": i, "description": d} for i, d in zip(DOC_TYPES, OPTION_TEXT["doc_type"]["en"])]
FORM_OPTIONS = [{"id": i, "description": d} for i, d in
                zip(("complete", "missing_required", "illegible"), OPTION_TEXT["form_complete"]["en"])]
TF_OPTIONS = [{"id": "false", "description": "No"}, {"id": "true", "description": "Yes"}]

# (kind, scenario) per day. kind: inv=invoice declaration, form=declaration
# form, wrong=private paper (receipt/letter). listed: sender on the manifest.
SCENARIOS = {
    1: [("inv", {}), ("inv", {}), ("wrong",), ("inv", {}), ("wrong",), ("inv", {}),
        ("wrong",), ("inv", {})],
    2: [("inv", {"listed": True}), ("form", "complete"), ("inv", {"listed": False}),
        ("form", "missing_required"), ("inv", {"listed": True}), ("wrong",),
        ("inv", {"listed": True}), ("inv", {"listed": True})],
    3: [("inv", {"total": "wide_in", "listed": True}), ("inv", {"total": "wide_over", "listed": True}),
        ("inv", {"total": "wide_in", "listed": True}), ("form", "complete"),
        ("inv", {"total": "wide_in", "listed": False}), ("inv", {"total": 5250.00, "listed": True}),
        ("inv", {"total": "wide_in", "listed": True}), ("form", "missing_required")],
    4: [("inv", {"total": "fine_in", "listed": True}), ("inv", {"total": "fine_over", "listed": True, "distract": False}),
        ("inv", {"total": "fine_in", "listed": True}), ("form", "complete"),
        ("inv", {"total": 5115.00, "listed": True}), ("inv", {"total": "fine_in", "listed": False}),
        ("form", "missing_required"), ("wrong",)],
}

TOTALS = {"wide_in": WIDE_IN, "wide_over": WIDE_OVER, "fine_in": FINE_IN, "fine_over": FINE_OVER}


def make_doc(day: int, i: int, scen: tuple, sender: str | None = None) -> dict:
    did = f"d{day}-{i}"
    rng = random.Random(stable_seed("amber", "v2", did))
    kind = scen[0]
    sc = scen[1] if len(scen) > 1 else {}
    doc = {"id": did, "kind": kind, "image": f"assets/docs/{did}.png"}
    if kind == "inv":
        # "total" is a TOTALS pool name, or a literal fine-gap value chosen by
        # measurement (level design inside the validated distribution)
        total = (sc["total"] if isinstance(sc.get("total"), float)
                 else rng.choice(TOTALS[sc["total"]]) if "total" in sc
                 else rng.choice((312.50, 812.50, 1850.00)))
        # the measured hard mode (E38a decisive counterfactuals): a competing
        # amount printed on the same page, near the cap on fine gaps. The draw
        # is always consumed (so frozen docs keep their byte-identical renders)
        # but only used where the scenario wants the competing amount.
        gap = "fine" if (isinstance(sc.get("total"), float) or str(sc.get("total", "")).startswith("fine")) else "wide"
        drawn = round(total + rng.choice((-1, 1)) * rng.uniform(40, 160), 2) if gap == "fine" else None
        distractor = drawn if (gap == "fine" and sc.get("distract", True)) else None
        content = amount_doc("invoice", total, rng, "en", distractor=distractor)
        if sender is not None:  # assigned by the day plan: unique companies per day
            content = replace(content, sender=sender)
            content = reprice(content, content.total_value, rng)
        doc.update(doc_type="invoice", total=content.total_value, sender=content.sender,
                   distractor=distractor,
                   listed=sc.get("listed", True), template=f"acp-h0b0f{rng.randrange(2)}",
                   issuer="vel" if rng.random() < 0.7 else "kor")
    elif kind == "form":
        label = sc
        title, fields = form_content(label, rng, "en")
        assert form_label(fields) == label
        doc.update(doc_type="form", form_label=label, form_title=title,
                   template=f"acp-h{rng.randrange(3)}b1f0", issuer="vel")
    else:
        t = rng.choice(("receipt", "letter"))
        content = doc_content(t, rng, "en")
        doc.update(doc_type=t, template=f"acp-h{rng.randrange(4)}b0f0",
                   issuer="kor" if rng.random() < 0.5 else "vel")
    doc["violations"] = violations_of(day, doc)
    doc["verdict"] = "deny" if doc["violations"] else "approve"
    return doc


def violations_of(day: int, doc: dict) -> list[str]:
    v: list[str] = []
    if doc["kind"] == "wrong":
        v.append("paper type")
    if doc["kind"] == "form" and day >= 2 and doc["form_label"] != "complete":
        v.append("incomplete form")
    if doc["kind"] == "inv":
        if day >= 2 and not doc["listed"]:
            v.append("not on manifest")
        if day >= 3 and doc["total"] >= CAP:
            v.append("over the cap")
    return v


def checks_for(day: int, doc: dict, manifest: list[str]) -> list[dict]:
    """Every check in its measured framing; gold never enters the model input."""
    out = [{"id": f"{doc['id']}-c1", "kind": "type", "family": "doc_type", "state": SD,
            "question": "Which document type is this?", "options": TYPE_OPTIONS,
            "gold": doc["doc_type"], "display": "document type"}]
    if doc["kind"] == "inv" and day >= 2:
        rng = random.Random(stable_seed("amber", "v2", doc["id"], "names"))
        others = [c for c in dict.fromkeys(manifest + list(COMPANIES)) if c != doc["sender"]][:4]
        names = [doc["sender"], *others]
        rng.shuffle(names)
        out.append({"id": f"{doc['id']}-c2", "kind": "manifest", "family": "source_conflict",
                    "state": SD,
                    "question": "Which company is named as the sender on this document?",
                    "options": [{"id": f"g{n}", "description": c} for n, c in enumerate(names)],
                    "gold": f"g{names.index(doc['sender'])}", "display": "sender on manifest"})
    if doc["kind"] == "inv" and day >= 3:
        out.append({"id": f"{doc['id']}-c3", "kind": "amount", "family": "field_threshold",
                    "state": SD,
                    "question": "Is the total on this document at least 5,000.00 VEL?",
                    "options": TF_OPTIONS, "gold": "true" if doc["total"] >= CAP else "false",
                    "display": "within the 5,000.00 VEL cap"})
    if doc["kind"] == "form":
        out.append({"id": f"{doc['id']}-c2", "kind": "form", "family": "form_complete",
                    "state": SD,
                    "question": "Is this form complete? Required fields are marked with *.",
                    "options": FORM_OPTIONS, "gold": doc["form_label"], "display": "form complete"})
    return out


def plan() -> dict:
    days = []
    for day in (1, 2, 3, 4):
        rng = random.Random(stable_seed("amber", "v2", "day", day))
        # every invoice doc gets a distinct company; manifest fillers come only
        # from companies no doc used, so an unlisted sender can never be listed
        invs = sum(1 for sc in SCENARIOS[day] if sc[0] == "inv")
        senders = rng.sample(COMPANIES, invs)
        it = iter(senders)
        docs = [make_doc(day, i + 1, sc, sender=next(it) if sc[0] == "inv" else None)
                for i, sc in enumerate(SCENARIOS[day])]
        listed = [d["sender"] for d in docs if d["kind"] == "inv" and d["listed"]]
        fillers = [c for c in COMPANIES if c not in set(senders)]
        manifest = list(dict.fromkeys(listed + fillers))[:6]
        for d in docs:
            d["checks"] = checks_for(day, d, manifest)
        days.append({"day": day, "date": DAYS[day],
                     "slips": [{"title": t, "text": x, "pinned": p}
                               for dd in range(1, day + 1) for t, x, p in SLIPS[dd]],
                     "manifest": manifest, "docs": docs})
    return {"policy": POLICY, "days": days}


# ------------------------------------------------------------------ render ---
def worn_theme(seed: int, issuer: str):
    """The new worn-paper theme (an E38 Theme): Veldania's olive cream paper,
    Korvenia's grey, per-doc wear in pad/dx/dy. Additive: E38 outputs untouched."""
    from mksh_v.gen_e38a import Theme

    r = random.Random(seed)
    return Theme(font="DejaVu Serif", base_px=15,
                 accent="#6B6A45" if issuer == "vel" else "#5C5C66",
                 ink="#23251D",
                 paper="#EFE7D2" if issuer == "vel" else "#E0DED4",
                 pad=r.randint(20, 26), dx=r.randint(-3, 3), dy=r.randint(-2, 2))


def render_all(campaign: dict) -> dict:
    r = Renderer()
    shas = {}
    try:
        for day in campaign["days"]:
            for doc in day["docs"]:
                rng = random.Random(stable_seed("amber", "v2", doc["id"], "content"))
                theme = worn_theme(stable_seed("amber", "v2", doc["id"], "wear"), doc["issuer"])
                if doc["kind"] == "inv":
                    # rebuild, then force the semantic fields the checks read:
                    # the printed sender and total must equal the plan's gold.
                    # the near-cap competing amount (doc["distractor"]) is part
                    # of the measured hard-mode design: it must reach the pixels.
                    content = amount_doc("invoice", doc["total"], rng, "en",
                                         distractor=doc.get("distractor"))
                    content = replace(content, sender=doc["sender"])
                    content = reprice(content, doc["total"], rng)
                    html = render_doc(content, doc["template"], theme)
                elif doc["kind"] == "form":
                    _, fields = form_content(doc["form_label"], rng, "en")
                    html = render_form(doc["form_title"], fields, "en", doc["template"], theme)
                else:
                    html = render_doc(doc_content(doc["doc_type"], rng, "en"), doc["template"], theme)
                png = r.render(html, theme.font)
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
    campaign["generator"] = {"commit": git_commit(),
                             "fonts_sha256": {f.name: sha256(f.read_bytes()) for f in sorted(FONTS.glob("*.ttf"))},
                             "images_sha256": shas}
    (OUT / "days.json").write_text(json.dumps(campaign, indent=1, ensure_ascii=False), encoding="utf-8")
    n_checks = sum(len(d["checks"]) for day in campaign["days"] for d in day["docs"])
    print(json.dumps({"days": 4, "docs": 32, "checks": n_checks,
                      "verdicts": {f"d{day['day']}": [d["verdict"][0] for d in day["docs"]]
                                   for day in campaign["days"]}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
