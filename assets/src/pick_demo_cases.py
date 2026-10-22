"""Pick the demo's case set (v2, informed by the probe: GUI train is 40/40
favourable across en/es/de and neg/fin; CORD train is refused ~100 %; our
synthetic train rows are at ceiling in most families).

Selection: 9 favourable (3 GUI in en/es/de with mixed polarity, 6 synthetic,
one per family with spread languages) + 4 refusals (the three existing CORD
cases + a fresh es one). Every case is run through the released weights and
policy and verified against gold. Train splits only; the eval sets are never
touched.

    MKSHV_MODEL_DIR=release python assets/src/pick_demo_cases.py
    python assets/src/materialize_cases.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))
from PIL import Image  # noqa: E402

EX = ROOT / "demo/examples"

# the curated cases already shipped (kept: the README's case table links to them)
EXISTING = [
    dict(kind="gui", src=str(EX / "gui-notes-fin.png"), family="task_done", language="en",
         state="Goal: Listen to a lecture on Artificial Intelligence and then jot down the name of the course in the notes app.",
         question="Does the screenshot show the goal as completed?",
         options=[("false", "No"), ("true", "Yes")], verdict="answer", note="kept: README case table"),
    dict(kind="cord", src=str(EX / "cord-receipt-1.png"), family="field_threshold", language="en",
         state="Scanned document page.",
         question="Is the total on this document at least 45.662 IDR?",
         options=[("false", "No"), ("true", "Yes")], verdict="abstain",
         note="kept: README case table; the model's top pick (false) would have been wrong (total 302.016 IDR)"),
    dict(kind="cord", src=str(EX / "cord-receipt-2.png"), family="field_threshold", language="de",
         state="Gescannte Dokumentseite.",
         question="Beträgt die Gesamtsumme auf diesem Dokument mindestens 120.000 IDR?",
         options=[("false", "Nein"), ("true", "Ja")], verdict="abstain", note="kept: README case table"),
    dict(kind="cord", src=str(EX / "cord-receipt-1.png"), family="field_threshold", language="en",
         state="Scanned document page.",
         question="Which range does the total on this document fall into?",
         options=[("low", "below 100.000 IDR"), ("mid", "100.000 to 200.000 IDR"), ("high", "above 200.000 IDR")],
         verdict="abstain", note="kept: 3-option range question"),
]

WANT_SYN = [("doc_type", "en"), ("form_complete", "es"), ("field_threshold", "de"),
            ("task_done", "en"), ("urgency", "en"), ("screen_state", "de")]
WANT_GUI = [("es", "fin"), ("de", "neg"), ("en", "neg")]
WANT_CORD_REFUSAL = ["es"]


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def decide_on(app, img_path, r):
    bars, verdict, abstain, _ = app.decide(
        Image.open(img_path), r["state"], r["question"],
        "\n".join(f"{o['id']}: {o['description']}" for o in r["options"]))
    nums = [float(x.strip(" %")) for x in re.findall(r"prob-num'>([^<]*)<", bars)]
    ids = re.findall(r"prob-id'>([^<]*)<", bars)
    pmax = max(nums) / 100
    return pmax, (ids[nums.index(max(nums))] if pmax >= 0.55 else None)


def entry(kind, src, r, pmax, top):
    return dict(kind=kind, src=str(src), family=r["family"], language=r["language"],
                state=r["state"], question=r["question"],
                options=[(o["id"], o["description"]) for o in r["options"]],
                gold=r["target_option"], top=top, pmax=round(pmax, 3),
                verdict="answer" if pmax >= 0.55 else "abstain",
                **({"gt_total_text": r["gt_total_text"]} if "gt_total_text" in r else {}))


def main() -> None:
    import app
    picked = list(EXISTING)
    used = {c["question"] + c["language"] for c in picked}
    # synthetic: one per family at the wanted language, favourable only
    for fam, lang in WANT_SYN:
        src_dir = "e38a" if fam in ("doc_type", "form_complete", "field_threshold") else "e38b"
        for r in rows(ROOT / f"data/{src_dir}/{fam}/rows.jsonl"):
            if r["split"] != "train" or r["language"] != lang or r.get("cf_role") not in (None, "base"):
                continue
            if r["question"] + r["language"] in used:
                continue
            src = ROOT / f"data/{src_dir}/{r['image']}"
            pmax, top = decide_on(app, src, r)
            if pmax >= 0.55 and top == r["target_option"]:
                picked.append(entry("syn", src, r, pmax, top)); used.add(r["question"] + r["language"])
                print("keep syn", fam, lang, f"p={pmax:.2f}", flush=True)
                break
    # gui: favourable at the wanted language/polarity
    for lang, pol in WANT_GUI:
        for r in rows(ROOT / "data/e41/gui_odyssey/rows.jsonl"):
            if r["split"] != "train" or r["language"] != lang or r.get("polarity") != pol:
                continue
            if r["question"] + r["language"] in used:
                continue
            pmax, top = decide_on(app, ROOT / "data/e41" / r["image"], r)
            if pmax >= 0.55 and top == r["target_option"]:
                picked.append(entry("gui", ROOT / "data/e41" / r["image"], r, pmax, top))
                used.add(r["question"] + r["language"])
                print("keep gui", lang, pol, f"p={pmax:.2f}", flush=True)
                break
    # cord: one fresh refusal in spanish
    for r in rows(ROOT / "data/e41/cord/rows.jsonl"):
        if r["split"] != "train" or r["language"] != "es" or r["question"] + r["language"] in used:
            continue
        pmax, top = decide_on(app, ROOT / "data/e41" / r["image"], r)
        if pmax < 0.55:
            picked.append(entry("cord", ROOT / "data/e41" / r["image"], r, pmax, top))
            print("keep-refusal cord es", f"p={pmax:.2f}", flush=True)
            break
    json.dump(picked, open(ROOT / "demo/cases.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(len(picked), "cases ->", sum(c["verdict"] == "answer" for c in picked), "answers,",
          sum(c["verdict"] == "abstain" for c in picked), "refusals")


if __name__ == "__main__":
    main()
