"""E44 bind pool (criteria: docs/E44_CRITERIA_2026-10-23.md).

    python scripts/gen_e44.py   # -> data/e44/bind/rows.jsonl (+ manifest)

Balanced pairs in which only the text changes, on images that already exist:
NUM-SYN / NUM-TEXT / RANGE on E38a field_threshold images, NUM-CORD on E41
CORD receipts, GOAL on E41 GUI-Odyssey final screens. Train rows come from
train splits; dev rows (selection only) from dev images not in the E44 probe.
`img_src` names the dataset root the `image` path is relative to.
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
from gen_e44_probe import STATED  # noqa: E402
from mksh_v.gen_e38a import fmt_amount, stable_seed  # noqa: E402

OUT = ROOT / "data/e44/bind"
WEIGHTS = {"num_syn": 0.35, "num_cord": 0.20, "num_text": 0.10, "range": 0.05, "goal": 0.30}
RANGE_Q = {"en": "Which range does the total on this document fall into?",
           "es": "¿En qué rango está el total de este documento?",
           "de": "In welchem Bereich liegt die Gesamtsumme auf diesem Dokument?"}
RANGE_OPT = {"en": "from {lo} to under {hi} {cur}", "es": "de {lo} a menos de {hi} {cur}",
             "de": "von {lo} bis unter {hi} {cur}"}


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def gap(rng: random.Random, fine: bool) -> float:
    lo, hi = (1.005, 1.5) if fine else (1.5, 1000.0)
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def fmt_int(n: int, lang: str) -> str:
    s = f"{n:,}"
    return s if lang == "en" else s.replace(",", ".")


def fmt_bin(e: int, lang: str) -> str:
    """10**e as printed in the row's language (0.001 ... 10.000.000)."""
    if e >= 0:
        return fmt_int(10 ** e, lang)
    s = f"{10.0 ** e:.{-e}f}"
    return s if lang == "en" else s.replace(".", ",")


def jaccard(a: str, b: str) -> float:
    x, y = set(a.lower().split()), set(b.lower().split())
    return len(x & y) / max(len(x | y), 1)


def main() -> int:
    from mokka_decisions.contracts import Option

    rows: list[dict] = []
    count: dict[str, int] = {}

    def emit(pair, kind, split, lang, domain, src, image, state, question, options, gold, **extra):
        count[pair] = count.get(pair, 0) + 1
        rows.append(case_row(
            id=f"e44-{pair}-{count[pair] - 1}", group_id=pair,
            source="e44-bind", language=lang, domain=domain, state=state, question=question,
            options=tuple(Option(id=i, description=d) for i, d in options), target_option=gold,
            label_origin="e44-bind", split=split,
            extra={"family": "field_threshold" if kind.startswith("num") else
                   ("total_range" if kind == "range" else "task_done"),
                   "bind": kind, "pair_id": pair, "img_src": src, "image": image, **extra}))

    def opts(o):
        return [(x.id, x.description) for x in o]

    def threshold_pair(pair, kind, split, lang, src, image, total, rng, fine, idr=False, stated=False):
        g = gap(rng, fine)
        state = DOC_STATE[lang]
        if stated:
            state += " " + STATED[lang].format(t=fmt_idr(total) + " IDR" if idr else fmt_amount(total, lang))
        for thr, gold in ((total / g, "true"), (total * g, "false")):
            if idr:
                thr = round(thr)
                q = DOC_QUESTION["field_threshold"][lang].replace("EUR", "IDR").format(thr=fmt_idr(thr))
            else:
                thr = round(thr, 2)
                q = DOC_QUESTION["field_threshold"][lang].format(thr=fmt_amount(thr, lang))
            assert (total >= thr) == (gold == "true"), (total, thr, gold)
            emit(pair, kind, split, lang, "documents", src, image, state, q, opts(_ft_options(lang)),
                 gold, total=total, threshold=thr, gap=round(g, 6))

    # ---- synthetic images (E38a field_threshold) ----
    probe = load(ROOT / "data/e44_probe/rows.jsonl")
    probe_syn = {r["src_id"] for r in probe if r["probe_set"] == "syn"}
    probe_cord = {r["probe_doc"] for r in probe if r["probe_set"] == "cord"}
    e38a = load(ROOT / "data/e38a/field_threshold/rows.jsonl")
    probe_groups = {r["group_id"] for r in e38a if r["id"] in probe_syn}  # counterfactual siblings too
    assert len(probe_groups) == len(probe_syn) == 120
    syn = [r for r in e38a if r["language"] in LANGS and r["group_id"] not in probe_groups
           and r["split"] in ("train", "dev")]
    ranges: list[tuple] = []
    for r in syn:
        rng = random.Random(stable_seed("e44-syn", r["id"]))
        split, lang, total = r["split"], r["language"], r["total"]
        n_pairs = 2 if split == "train" else 1
        for k in range(n_pairs):
            threshold_pair(f"syn-{r['id']}-{k}", "num_syn", split, lang, "e38a", r["image"],
                           total, rng, fine=(k == 0) if split == "train" else rng.random() < 0.5)
        if split == "train" and rng.random() < 0.5:
            threshold_pair(f"txt-{r['id']}", "num_text", split, lang, "e38a", r["image"], total,
                           rng, fine=rng.random() < 0.5, stated=True)
        ranges.append((r, rng.randrange(4)))
    ranges.sort(key=lambda x: x[0]["split"])  # stable: pairs never straddle splits
    for i in range(0, len(ranges) - 1, 2):
        if ranges[i][0]["split"] != ranges[i + 1][0]["split"]:
            continue  # RANGE singletons, two per pair slot
        for r, pos in ranges[i:i + 2]:
            lang, total = r["language"], r["total"]
            b = math.floor(math.log10(total))
            s = b - pos
            options = [(f"r{j}", RANGE_OPT[lang].format(lo=fmt_bin(s + j, lang),
                                                        hi=fmt_bin(s + j + 1, lang), cur="EUR"))
                       for j in range(4)]
            gold = f"r{pos}"
            assert 10.0 ** b <= total < 10.0 ** (b + 1)
            emit(f"rng-{ranges[i][0]['id']}", "range", r["split"], lang, "documents", "e38a",
                 r["image"], DOC_STATE[lang], RANGE_Q[lang], options, gold, total=total)

    # ---- CORD receipts (E41) ----
    cord: dict[str, dict] = {}
    for r in load(ROOT / "data/e41/cord/rows.jsonl"):
        cord.setdefault(r["group_id"], r)
    for i, (gid, r) in enumerate(sorted(cord.items())):
        if gid in probe_cord:
            continue
        rng = random.Random(stable_seed("e44-cord", gid))
        lang = LANGS[i % len(LANGS)]
        n_pairs = 2 if r["split"] == "train" else 1
        for k in range(n_pairs):
            threshold_pair(f"cord-{gid}-{k}", "num_cord", r["split"], lang, "e41", r["image"],
                           r["gt_total_value"], rng, fine=(k == 0) if n_pairs == 2 else rng.random() < 0.5,
                           idr=True)

    # ---- GUI final screens (E41), own vs another episode's goal ----
    fin: dict[str, dict] = {}
    for r in load(ROOT / "data/e41/gui_odyssey/rows.jsonl"):
        if r["polarity"] == "fin":
            fin.setdefault(r["group_id"], r)
    eps = sorted(fin)
    for i, eid in enumerate(eps):
        r = fin[eid]
        rng = random.Random(stable_seed("e44-goal", eid))
        pool = [e for e in eps if e != eid and fin[e]["split"] == r["split"]]
        for _ in range(200):
            other = fin[rng.choice(pool)]["goal"]
            if jaccard(other, r["goal"]) < 0.5:
                break
        else:
            raise SystemExit(f"no swap goal for {eid}")
        lang = LANGS[i % len(LANGS)]
        for goal, gold in ((r["goal"], "true"), (other, "false")):
            emit(f"goal-{eid}", "goal", r["split"], lang, "screens", "e41", r["image"],
                 SCR_STATE["task_done"][lang].format(goal=goal), SCR_QUESTION["task_done"][lang],
                 opts(_td_options(lang)), gold, own_goal=r["goal"], goal=goal)

    # ---- checks ----
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)
    by_pair: dict[str, list[dict]] = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    for p, rs in by_pair.items():
        assert len(rs) == 2, p
        assert len({x["split"] for x in rs}) == 1
        if rs[0]["bind"] != "range":
            assert rs[0]["image"] == rs[1]["image"] and {x["target_option"] for x in rs} == {"true", "false"}
    roots = {"e38a": ROOT / "data/e38a", "e41": ROOT / "data/e41"}
    for r in rows:
        assert (roots[r["img_src"]] / r["image"]).is_file(), r["image"]
    assert not {r["image"] for r in rows} & {r["image"] for r in probe if r["probe_set"] == "cord"}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                    encoding="utf-8")
    n = {sp: {k: sum(r["split"] == sp and r["bind"] == k for r in rows) for k in WEIGHTS}
         for sp in ("train", "dev")}
    manifest = {"criteria": "docs/E44_CRITERIA_2026-10-23.md", "weights": WEIGHTS, "n_rows": n,
                "rows_sha256": sha256_file(OUT / "rows.jsonl")}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    for k in WEIGHTS:
        r = next(x for x in rows if x["bind"] == k)
        print(" ", k, r["target_option"], "|", r["state"][-70:], "|", r["question"][:70])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
