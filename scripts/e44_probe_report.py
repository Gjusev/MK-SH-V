"""E44 probe report (criteria: docs/E44_PROBE_CRITERIA_2026-10-23.md).

    python scripts/e44_probe_report.py   # -> runs/e44_probe/e44_probe_report.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43)


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def p_true(case: dict, row: dict) -> float:
    ids = [o["id"] for o in row["options"]]
    return case["probs"][ids.index("true")]


def main() -> int:
    rows = {r["id"]: r for r in load(ROOT / "data/e44_probe/rows.jsonl")}
    out: dict = {"criteria": "docs/E44_PROBE_CRITERIA_2026-10-23.md", "seeds": {}}
    for seed in SEEDS:
        cases = load(ROOT / f"runs/e44_probe/percase_s{seed}.jsonl")
        per: dict = {}
        for kind in ("syn", "cord"):
            mine = [(c, rows[c["id"]]) for c in cases if rows[c["id"]]["probe_set"] == kind]
            acc = {cond: [c["pred"] == c["gold"] for c, r in mine if r["probe_cond"] == cond] for cond in "OLHT"}
            pl = {r["probe_doc"]: p_true(c, r) for c, r in mine if r["probe_cond"] == "L"}
            ph = {r["probe_doc"]: p_true(c, r) for c, r in mine if r["probe_cond"] == "H"}
            s = sum(pl[d] - ph[d] for d in pl) / len(pl)
            per[kind] = {"n": {k: len(v) for k, v in acc.items()},
                         "acc": {k: round(sum(v) / len(v), 4) for k, v in acc.items()},
                         "S": round(s, 4),
                         "mean_p_true": {"L": round(sum(pl.values()) / len(pl), 4),
                                         "H": round(sum(ph.values()) / len(ph), 4)}}
        out["seeds"][str(seed)] = per

    readings = {}
    for kind in ("syn", "cord"):
        v = [out["seeds"][str(s)][kind] for s in SEEDS]
        lh = [min(x["acc"]["L"], x["acc"]["H"]) for x in v]
        if all(x >= 0.90 for x in lh):
            readings[kind] = "R1 magnitude read; failure is fine-grained -> E44 targets fine digit reading"
        elif all(x["S"] < 0.10 for x in v):
            t = [x["acc"]["T"] for x in v]
            if all(x >= 0.90 for x in t):
                readings[kind] = "R2 question number ignored; stated totals compared -> E44 targets perception"
            elif all(x < 0.70 for x in t):
                readings[kind] = "R2 question number ignored; even stated totals not compared -> E44 targets text-number binding"
            else:
                readings[kind] = "R2 question number ignored; T intermediate (see numbers)"
        else:
            readings[kind] = "intermediate (see numbers)"
    out["readings"] = readings
    (ROOT / "runs/e44_probe/e44_probe_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for s in SEEDS:
        for kind in ("syn", "cord"):
            x = out["seeds"][str(s)][kind]
            print(f"s{s} {kind:4}: acc O={x['acc']['O']} L={x['acc']['L']} H={x['acc']['H']} T={x['acc']['T']} | "
                  f"S={x['S']} (p_true L {x['mean_p_true']['L']}, H {x['mean_p_true']['H']})")
    for k, v in readings.items():
        print(f"reading {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
