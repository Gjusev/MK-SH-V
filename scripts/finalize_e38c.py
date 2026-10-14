"""E38c finalize (criteria: docs/E38C_CRITERIA_2026-10-14.md + -r2).

    python scripts/finalize_e38c.py

Merges operator visual reviews (inline + agent verdicts), keeps task_done pairs
whose positive is "yes", patches label_origin, builds extension rows for kept
new episodes, deletes unreferenced shots, regenerates the manifest, writes the
50-positive coordinator subsample and the negative-audit summary (CP bound).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
DATA = ROOT / "data/e38c"
GUI = DATA / "gui_odyssey"
AUD = ROOT / "runs/e38c-audit"
LANGS = ("en", "es", "de")
LABEL_ORIGIN = "annotation+visual-verification"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_verdicts() -> tuple[dict, dict]:
    pos, neg = {}, {}
    for l in (AUD / "positive_review.jsonl").read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        pos.setdefault(r["episode_id"], (r["operator_verdict"], r.get("reason", "")))
    for p in sorted((AUD / "verdicts").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for v in d["verdicts"]:
            if d["kind"] == "positive":
                pos.setdefault(v["episode_id"], (v["operator_verdict"], v.get("reason", "")))
            else:
                neg.setdefault(v["episode_id"], (bool(v["valid"]), v.get("reason", "")))
    return pos, neg


def cp_upper(k: int, n: int) -> float:
    try:
        from scipy.stats import beta
        return float(beta.ppf(0.95, k + 1, n - k))
    except ImportError:
        def pdf(x, a, b):
            from math import exp, lgamma
            return exp((a - 1) * (x and __import__("math").log(x)) +
                       (b - 1) * __import__("math").log(1 - x) +
                       lgamma(a + b) - lgamma(a) - lgamma(b)) if 0 < x < 1 else 0.0
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            steps = 400
            cdf = sum(pdf(i / steps, k + 1, n - k) / steps for i in range(steps)
                      if i / steps <= mid)
            if cdf < 0.95:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


def build_extension_rows(pool_entry: dict) -> list[dict]:
    from mokka_decisions.contracts import DecisionCase, Option
    from mksh_v.gen_e38b import OPTION_TEXT, QUESTION as SCR_QUESTION, STATE as SCR_STATE

    eid = pool_entry["episode_id"]
    k, n = pool_entry["k"], pool_entry["n_steps"]
    steps = pool_entry["steps"]
    opts = {lang: tuple(Option(id=i, description=d) for i, d in
                        zip(("false", "true"), OPTION_TEXT["task_done"][lang]))
            for lang in LANGS}
    rows = []
    for lang in LANGS:
        for polarity, rel, label in (("neg", f"shots/{eid}_neg.png", "false"),
                                     ("fin", f"shots/{eid}_fin.png", "true")):
            rows.append(DecisionCase(
                id=f"e38c-gui-{eid}-{polarity}-{lang}", group_id=eid,
                source="e38c-gui", language=lang, domain="screens",
                state=SCR_STATE["task_done"][lang].format(goal=pool_entry["goal"]),
                question=SCR_QUESTION["task_done"][lang],
                options=opts[lang], target_kind="single", target_option=label,
                label_origin=LABEL_ORIGIN, split="test").to_dict()
                | {"family": "task_done", "image": rel, "polarity": polarity,
                   "k": k, "n_steps": n, "steps": steps, "final_action": "COMPLETE",
                   "neg_ref": pool_entry["neg_ref"], "fin_ref": pool_entry["fin_ref"],
                   "goal": pool_entry["goal"]})
    return rows


def main() -> int:
    pos, neg = load_verdicts()
    orig_ids = [r["group_id"] for r in json.loads(
        (AUD / "positives_to_review.json").read_text(encoding="utf-8"))]
    missing_orig = [e for e in orig_ids if e not in pos]
    pool = load_jsonl(GUI / "extend_pool.jsonl")
    clean_pool = [r for r in pool if r.get("clean")]
    missing_ext = [r["episode_id"] for r in clean_pool if r["episode_id"] not in pos]
    print(f"verdicts: pos={len(pos)} neg={len(neg)}; missing orig={len(missing_orig)} "
          f"ext={len(missing_ext)}", flush=True)
    if missing_orig or missing_ext:
        print("INCOMPLETE reviews; write missing verdicts first. Sample:",
              (missing_orig or missing_ext)[:5], flush=True)
        return 1

    keep = sorted(e for e, (v, _) in pos.items() if v == "yes")
    tally = Counter(v for v, _ in pos.values())
    print("positive tally:", dict(tally), "| keep:", len(keep), flush=True)
    keep_json = AUD / "keep_episodes.json"
    keep_json.write_text(json.dumps(keep), encoding="utf-8")

    pool_by_id_pre = {r["episode_id"]: r for r in clean_pool}
    missing_negs = [e for e in keep if e in pool_by_id_pre
                    and not (GUI / f"shots/{e}_neg.png").exists()]
    if missing_negs:
        print(f"keep={len(keep)}; extension negatives missing: {len(missing_negs)} "
              f"-> run: python scripts/extend_e38c.py negs runs/e38c-audit/keep_episodes.json, "
              f"then re-run finalize", flush=True)
        return 2

    # negative audit stats (100 fixed by base criteria)
    n_neg = len(neg)
    k_bad = sum(1 for ok, _ in neg.values() if not ok)
    noise = k_bad / max(n_neg, 1)
    ub = cp_upper(k_bad, n_neg)

    orig_rows = load_jsonl(GUI / "rows.jsonl")
    kept_orig = [dict(r, label_origin=LABEL_ORIGIN) if r["source"] == "e38c-gui" else r
                 for r in orig_rows if r["group_id"] in set(keep)]
    pool_by_id = {r["episode_id"]: r for r in clean_pool}
    ext_rows = [row for e in keep if e in pool_by_id for row in build_extension_rows(pool_by_id[e])]
    final_rows = kept_orig + ext_rows
    print(f"rows: kept_orig={len(kept_orig)} ext={len(ext_rows)} total={len(final_rows)}", flush=True)

    (GUI / "rows.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in final_rows), encoding="utf-8")

    # delete shots not referenced by any kept row
    referenced = {r["image"] for r in final_rows}
    removed = 0
    for p in (GUI / "shots").iterdir():
        if f"shots/{p.name}" not in referenced:
            p.unlink()
            removed += 1
    print(f"shots removed: {removed}, kept: {len(list((GUI / 'shots').iterdir()))}", flush=True)

    # manifest over all remaining files
    files = {}
    for sub in ("cord", "gui_odyssey"):
        for p in (DATA / sub).rglob("*"):
            if p.is_file() and p.name != "manifest.json" and p.name != "extend_pool.jsonl":
                files[p.relative_to(DATA).as_posix()] = sha256_file(p)
    manifest = {"criteria": "docs/E38C_CRITERIA_2026-10-14.md+r2",
                "n_rows": {"cord": 576, "gui_odyssey": len(final_rows)},
                "stats": {"positive_review": {"reviewed": len(pos), "tally": dict(tally),
                                              "kept_episodes": len(keep),
                                              "discard_rate": round(1 - len(keep) / len(pos), 4)},
                          "negative_audit": {"n": n_neg, "invalid": k_bad,
                                             "noise_rate": round(noise, 4),
                                             "cp_upper_95": round(ub, 4)}},
                "files_sha256": files}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # coordinator subsample: 50 kept positives by ascending sha256
    subsample = sorted(keep, key=lambda e: hashlib.sha256(e.encode()).hexdigest())[:50]
    (AUD / "positive_subsample_50.json").write_text(json.dumps(
        [{"episode_id": e, "image": f"shots/{e}_fin.png",
          "operator_verdict": pos[e][0], "reason": pos[e][1]} for e in subsample],
        ensure_ascii=False, indent=2), encoding="utf-8")

    with (AUD / "negative_review.jsonl").open("w", encoding="utf-8") as f:
        for e, (ok, reason) in sorted(neg.items()):
            f.write(json.dumps({"episode_id": e, "valid": ok, "reason": reason},
                               ensure_ascii=False) + "\n")
    print(f"noise {k_bad}/{n_neg} = {noise:.2%}, CP95 upper {ub:.2%} "
          f"({'<=15% clean' if ub <= 0.15 else '>15% -> report as eval with label noise'})",
          flush=True)
    print("subsample written:", len(subsample), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
