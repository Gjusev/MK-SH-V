"""E44 report (criteria: docs/E44_CRITERIA_2026-10-23.md).

    python scripts/e44_report.py --prep   # -> runs/e44/m3_rows.jsonl (E43 swap + own goal, 800 rows)
    # M1/M3 per seed, CPU:
    #   python scripts/e43_infer.py --seed S --ckpt runs/e44/runs/a1v_e44_sS/best.pt \
    #       --rows data/e44_probe/rows.jsonl --root . --out runs/e44/probe_sS.jsonl
    #   python scripts/e43_infer.py --seed S --ckpt ... --rows runs/e44/m3_rows.jsonl \
    #       --root data/e43/gui_cal --out runs/e44/m3_sS.jsonl
    python scripts/e44_report.py          # -> runs/e44/e44_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e44_probe_report import load, p_true  # noqa: E402

SEEDS = (42, 43)
OUT = ROOT / "runs/e44"
E41 = {42: {"test_nonfr": 0.9083, "task_done": 0.826, "testC": 1.0, "task": 0.81},
       43: {"test_nonfr": 0.9088, "task_done": 0.783, "testC": 1.0, "task": 0.83}}


def prep() -> None:
    rows = load(ROOT / "data/e43/gui_cal/rows.jsonl")
    swap = load(ROOT / "data/e43/gui_cal/swap.jsonl")
    own = {r["group_id"]: r for r in rows if r.get("polarity") == "fin" and r["language"] == "en"}
    m3 = [dict(r, m3="swap") for r in swap] + [dict(own[r["group_id"]], m3="own") for r in swap]
    assert len(m3) == 800 and all(r["image"] for r in m3)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m3_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in m3),
                                       encoding="utf-8")
    print("m3 rows", len(m3))


def acc(rows: list[dict]) -> float:
    return round(sum(r["correct"] for r in rows) / len(rows), 4)


def seed_report(seed: int) -> dict:
    run = OUT / f"runs/a1v_e44_s{seed}"
    ev = {name: load(run / f"eval_{name}.jsonl") for name in ("test_nonfr", "real", "testC", "task")}
    fam = defaultdict(list)
    for r in ev["test_nonfr"]:
        fam[r["family"]].append(r)
    real_fam = defaultdict(list)
    for r in ev["real"]:
        real_fam[r["family"]].append(r)
    # M1: probe
    prow = {r["id"]: r for r in load(ROOT / "data/e44_probe/rows.jsonl")}
    probe = {}
    for kind in ("syn", "cord"):
        mine = [(c, prow[c["id"]]) for c in load(OUT / f"probe_s{seed}.jsonl") if prow[c["id"]]["probe_set"] == kind]
        a = {k: round(sum(c["pred"] == c["gold"] for c, r in mine if r["probe_cond"] == k)
                      / sum(r["probe_cond"] == k for _, r in mine), 4) for k in "OLHT"}
        pl = {r["probe_doc"]: p_true(c, r) for c, r in mine if r["probe_cond"] == "L"}
        ph = {r["probe_doc"]: p_true(c, r) for c, r in mine if r["probe_cond"] == "H"}
        probe[kind] = {"acc": a, "S": round(sum(pl[d] - ph[d] for d in pl) / len(pl), 4)}
    # M3: goal swap / own goal
    mrow = {r["id"]: r for r in load(OUT / "m3_rows.jsonl")}
    m3 = defaultdict(list)
    for c in load(OUT / f"m3_s{seed}.jsonl"):
        m3[mrow[c["id"]]["m3"]].append(c["pred"] == c["gold"])
    m3 = {k: round(sum(v) / len(v), 4) for k, v in m3.items()}
    return {"M1": probe, "M2_field_threshold_test": acc(fam["field_threshold"]),
            "M3": m3, "M4_cord_real": acc(real_fam["field_threshold"]),
            "N1_test_nonfr": acc(ev["test_nonfr"]), "N2_real_gui": acc(real_fam["task_done"]),
            "N3": {"testC": acc(ev["testC"]), "task": acc(ev["task"])},
            "per_family_test": {k: acc(v) for k, v in sorted(fam.items())}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    if ap.parse_args().prep:
        prep()
        return 0
    seeds = {s: seed_report(s) for s in SEEDS}
    g1 = all(x["M1"]["syn"]["S"] >= 0.50 and min(x["M1"]["syn"]["acc"]["L"], x["M1"]["syn"]["acc"]["H"]) >= 0.90
             and x["M2_field_threshold_test"] >= 0.70 for x in seeds.values())
    g2 = all(x["M3"]["swap"] >= 0.80 and x["M3"]["own"] >= 0.80 for x in seeds.values())
    g3 = {s: {"N1": E41[s]["test_nonfr"] - x["N1_test_nonfr"] <= 0.020,
              "N2": E41[s]["task_done"] - x["N2_real_gui"] <= 0.030,
              "N3": all(E41[s][k] - x["N3"][k] <= 0.030 for k in ("testC", "task"))}
          for s, x in seeds.items()}
    g3_ok = all(all(v.values()) for v in g3.values())
    out = {"criteria": "docs/E44_CRITERIA_2026-10-23.md", "seeds": seeds,
           "gates": {"G1_numbers": g1, "G2_goals": g2, "G3_no_regression": g3, "G3_ok": g3_ok,
                     "numbers_bound": g1 and g3_ok, "goals_bound": g2 and g3_ok}}
    (OUT / "e44_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
