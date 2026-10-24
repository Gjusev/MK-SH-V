"""E45 bind pool v2 (criteria: docs/E45_CRITERIA_2026-10-24.md).

    python scripts/gen_e45.py   # -> data/e45/bind/rows.jsonl (+ manifest)

Same images as E44 (probe documents and their counterfactual siblings
excluded), new rows: fine-gap threshold pairs, text-only comparison pairs,
digit-reading questions, the E44 RANGE family, unfiltered goal pairs and a
four-way goal-match family. `img_src` None = text-only row.
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

from gen_e38c import (DOC_QUESTION, DOC_STATE, LANGS, SCR_QUESTION, SCR_STATE,  # noqa: E402
                      _ft_options, _td_options, fmt_idr, sha256_file)
from gen_e41 import case_row  # noqa: E402
from gen_e44 import RANGE_OPT, RANGE_Q, fmt_bin, load  # noqa: E402
from gen_e44_probe import STATED  # noqa: E402
from mksh_v.gen_e38a import fmt_amount, stable_seed  # noqa: E402

OUT = ROOT / "data/e45/bind"
WEIGHTS = {"num_syn": 0.20, "num_cord": 0.10, "cmp_text": 0.15, "digits": 0.20,
           "range": 0.05, "goal": 0.15, "goal_match": 0.15}
DIGIT_Q = {
    "first": {"en": "What is the first digit of the total on this document?",
              "es": "¿Cuál es la primera cifra del total de este documento?",
              "de": "Was ist die erste Ziffer der Gesamtsumme auf diesem Dokument?"},
    "second": {"en": "What is the second digit of the total on this document?",
               "es": "¿Cuál es la segunda cifra del total de este documento?",
               "de": "Was ist die zweite Ziffer der Gesamtsumme auf diesem Dokument?"},
    "ndigits": {"en": "How many digits does the whole-number part of the total on this document have?",
                "es": "¿Cuántas cifras tiene la parte entera del total de este documento?",
                "de": "Wie viele Ziffern hat der ganzzahlige Teil der Gesamtsumme auf diesem Dokument?"},
}
MATCH_STATE = {"en": "Screenshot of a mobile app.", "es": "Captura de una app móvil.",
               "de": "Screenshot einer mobilen App."}
MATCH_Q = {"en": "Which goal does this screenshot show as completed?",
           "es": "¿Qué objetivo muestra la captura como completado?",
           "de": "Welches Ziel zeigt der Screenshot als erledigt?"}


def gap(rng: random.Random) -> float:
    lo, hi = (1.002, 1.10) if rng.random() < 0.70 else (1.10, 1000.0)
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def thresholds(total: float, g: float, unit: float) -> tuple[float, float]:
    """(true threshold <= total, false threshold > total), on the printed unit grid."""
    lo = math.floor(total / g / unit) * unit
    hi = max(math.ceil(total * g / unit) * unit, math.floor(total / unit) * unit + unit)
    return round(lo, 2), round(hi, 2)


def sig_digits(total: float, idr: bool) -> str:
    """Digits of the printed total without separators (EUR with 2 decimals)."""
    return str(round(total)) if idr else f"{total:.2f}".replace(".", "")


def main() -> int:
    from mokka_decisions.contracts import Option

    rows: list[dict] = []
    count: dict[str, int] = {}

    def emit(pair, kind, split, lang, domain, src, image, state, question, options, gold, **extra):
        count[pair] = count.get(pair, 0) + 1
        fam = {"goal": "task_done", "goal_match": "goal_match", "range": "total_range",
               "digits": "total_digits"}.get(kind, "field_threshold")
        rows.append(case_row(
            id=f"e45-{pair}-{count[pair] - 1}", group_id=pair, source="e45-bind", language=lang,
            domain=domain, state=state, question=question,
            options=tuple(Option(id=i, description=d) for i, d in options), target_option=gold,
            label_origin="e45-bind", split=split,
            extra={"family": fam, "bind": kind, "pair_id": pair, "img_src": src, "image": image, **extra}))

    def opts(o):
        return [(x.id, x.description) for x in o]

    def threshold_pair(pair, kind, split, lang, src, image, total, rng, idr=False, stated=False, tie=False):
        g = gap(rng)
        unit = 1 if idr else 0.01
        lo, hi = thresholds(total, g, unit)
        if tie:  # threshold equal to the total (true) vs one unit above (false)
            lo, hi = round(total, 2), round(total + unit, 2)
        state = DOC_STATE[lang]
        if stated:
            state += " " + STATED[lang].format(t=fmt_idr(total) + " IDR" if idr else fmt_amount(total, lang))
        for thr, gold in ((lo, "true"), (hi, "false")):
            if idr:
                q = DOC_QUESTION["field_threshold"][lang].replace("EUR", "IDR").format(thr=fmt_idr(thr))
            else:
                q = DOC_QUESTION["field_threshold"][lang].format(thr=fmt_amount(thr, lang))
            assert (total >= thr) == (gold == "true"), (total, thr, gold)
            emit(pair, kind, split, lang, "documents", src, image, state, q, opts(_ft_options(lang)),
                 gold, total=total, threshold=thr, gap=round(g, 6), tie=tie)

    singles: dict[tuple[str, str], list] = {}  # (kind, split) -> [(emit args)], grouped two per pair

    def single(kind, split, args):
        singles.setdefault((kind, split), []).append(args)

    # ---- synthetic images (E38a field_threshold) ----
    probe = load(ROOT / "data/e44_probe/rows.jsonl")
    probe_syn = {r["src_id"] for r in probe if r["probe_set"] == "syn"}
    probe_cord = {r["probe_doc"] for r in probe if r["probe_set"] == "cord"}
    e38a = load(ROOT / "data/e38a/field_threshold/rows.jsonl")
    probe_groups = {r["group_id"] for r in e38a if r["id"] in probe_syn}
    assert len(probe_groups) == len(probe_syn) == 120
    syn = [r for r in e38a if r["language"] in LANGS and r["group_id"] not in probe_groups
           and r["split"] in ("train", "dev")]
    for r in syn:
        rng = random.Random(stable_seed("e45-syn", r["id"]))
        split, lang, total = r["split"], r["language"], r["total"]
        for k in range(2 if split == "train" else 1):
            threshold_pair(f"syn-{r['id']}-{k}", "num_syn", split, lang, "e38a", r["image"], total, rng)
        which = ("first", "second", "ndigits")[rng.randrange(3)]
        single("digits", split, (r, which, False, "e38a"))
        single("range", split, (r, rng.randrange(4)))

    # ---- CORD receipts (E41) ----
    cord: dict[str, dict] = {}
    for r in load(ROOT / "data/e41/cord/rows.jsonl"):
        cord.setdefault(r["group_id"], r)
    for i, (gid, r) in enumerate(sorted(cord.items())):
        if gid in probe_cord:
            continue
        rng = random.Random(stable_seed("e45-cord", gid))
        lang = LANGS[i % len(LANGS)]
        for k in range(2 if r["split"] == "train" else 1):
            threshold_pair(f"cord-{gid}-{k}", "num_cord", r["split"], lang, "e41", r["image"],
                           r["gt_total_value"], rng, idr=True)
        which = ("first", "second", "ndigits")[rng.randrange(3)]
        single("digits", r["split"], (dict(r, language=lang, total=r["gt_total_value"]), which, True, "e41"))

    # ---- text-only comparison (no image) ----
    for split, n in (("train", 3000), ("dev", 150)):
        for i in range(n):
            rng = random.Random(stable_seed("e45-cmp", split, i))
            lang = LANGS[i % len(LANGS)]
            idr = rng.random() < 0.3
            total = (round(math.exp(rng.uniform(math.log(1000), math.log(5_000_000)))) if idr
                     else round(math.exp(rng.uniform(0, math.log(100_000))), 2))
            threshold_pair(f"cmp-{split}-{i}", "cmp_text", split, lang, None, None, max(total, 1.0),
                           rng, idr=idr, stated=True, tie=rng.random() < 0.05)

    # ---- GUI final screens (E41) ----
    fin: dict[str, dict] = {}
    for r in load(ROOT / "data/e41/gui_odyssey/rows.jsonl"):
        if r["polarity"] == "fin":
            fin.setdefault(r["group_id"], r)
    eps = sorted(fin)
    for i, eid in enumerate(eps):
        r = fin[eid]
        rng = random.Random(stable_seed("e45-goal", eid))
        own = r["goal"].strip()
        pool = sorted({fin[e]["goal"].strip() for e in eps
                       if e != eid and fin[e]["split"] == r["split"]} - {own}, key=str.lower)
        pool = [g for g in pool if g.lower() != own.lower()]
        lang = LANGS[i % len(LANGS)]
        other = rng.choice(pool)
        for goal, gold in ((own, "true"), (other, "false")):
            emit(f"goal-{eid}", "goal", r["split"], lang, "screens", "e41", r["image"],
                 SCR_STATE["task_done"][lang].format(goal=goal), SCR_QUESTION["task_done"][lang],
                 opts(_td_options(lang)), gold, own_goal=own, goal=goal)
        single("goal_match", r["split"], (r, own, rng.sample(pool, 3), rng.randrange(4), lang))

    # ---- singletons, two per pair slot (same kind and split) ----
    for (kind, split), items in sorted(singles.items()):
        for j in range(0, len(items) - 1, 2):
            pair = f"{kind}-{split}-{j // 2}"
            for it in items[j:j + 2]:
                if kind == "digits":
                    r, which, idr, src = it
                    lang, d = r["language"], sig_digits(r["total"], idr)
                    n_int = len(str(int(math.floor(r["total"]))))
                    if which == "first":
                        options, gold = [(f"d{k}", str(k)) for k in range(1, 10)], f"d{d[0]}"
                    elif which == "second":
                        options, gold = [(f"d{k}", str(k)) for k in range(10)], f"d{d[1]}"
                    else:
                        options, gold = [(f"n{k}", str(k)) for k in range(1, 8)], f"n{n_int}"
                    assert gold in {o[0] for o in options}, (r["total"], which)
                    emit(pair, kind, split, lang, "documents", src, r["image"], DOC_STATE[lang],
                         DIGIT_Q[which][lang], options, gold, total=r["total"], digit_q=which)
                elif kind == "range":
                    r, pos = it
                    lang, total = r["language"], r["total"]
                    b = math.floor(math.log10(total))
                    options = [(f"r{k}", RANGE_OPT[lang].format(lo=fmt_bin(b - pos + k, lang),
                                                                hi=fmt_bin(b - pos + k + 1, lang), cur="EUR"))
                               for k in range(4)]
                    emit(pair, kind, split, lang, "documents", "e38a", r["image"], DOC_STATE[lang],
                         RANGE_Q[lang], options, f"r{pos}", total=total)
                else:
                    r, own, others, pos, lang = it
                    goals = others[:pos] + [own] + others[pos:]
                    emit(pair, kind, split, lang, "screens", "e41", r["image"], MATCH_STATE[lang],
                         MATCH_Q[lang], [(f"g{k}", g) for k, g in enumerate(goals)], f"g{pos}", own_goal=own)

    # ---- checks ----
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)
    by_pair: dict[str, list[dict]] = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    for p, rs in by_pair.items():
        assert len(rs) == 2 and len({x["split"] for x in rs}) == 1 and len({x["bind"] for x in rs}) == 1, p
        if rs[0]["bind"] in ("num_syn", "num_cord", "cmp_text", "goal"):
            assert rs[0]["image"] == rs[1]["image"] and {x["target_option"] for x in rs} == {"true", "false"}
    roots = {"e38a": ROOT / "data/e38a", "e41": ROOT / "data/e41"}
    for r in rows:
        assert r["img_src"] is None or (roots[r["img_src"]] / r["image"]).is_file(), r["image"]
    probe_imgs = {r["image"].removeprefix("data/e38a/").removeprefix("data/e41/") for r in probe}
    assert not {r["image"] for r in rows if r["image"]} & probe_imgs

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                    encoding="utf-8")
    n = {sp: {k: sum(r["split"] == sp and r["bind"] == k for r in rows) for k in WEIGHTS}
         for sp in ("train", "dev")}
    manifest = {"criteria": "docs/E45_CRITERIA_2026-10-24.md", "weights": WEIGHTS, "n_rows": n,
                "rows_sha256": sha256_file(OUT / "rows.jsonl")}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    for k in WEIGHTS:
        r = next(x for x in rows if x["bind"] == k)
        print(" ", k, r["target_option"], "|", r["state"][-60:], "|", r["question"][:60], "|",
              [o["description"][:20] for o in r["options"]][:4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
