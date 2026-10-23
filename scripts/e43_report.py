"""E43 report (criteria: docs/E43_CRITERIA_2026-10-23.md).

    python scripts/e43_report.py      # -> runs/e43/e43_report.json

Reads only: data/e43/gui_cal/rows.jsonl (split map), runs/e43/percase_s{42,43}__{gui_cal,swap}.jsonl,
runs/e42-calib/percase_s{42,43}__gui.jsonl (held-out E38c GUI) and runs/e42-calib/e42_summary.json
(baseline operating point). The math (fit_T, Clopper-Pearson, risk_at, ece15) is E42's, imported.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
import e42_calib as e42  # noqa: E402

GRID = e42.TAU_GRID + [0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
BUDGET = e42.RISK_BUDGET
SEEDS = (42, 43)


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def pick_tau_ext(cases: list[dict], t: float) -> dict:
    """Frozen E43 rule: smallest tau on the extended grid whose cal_policy CP95
    upper bound is <= budget; none qualifies -> no feasible gate (abstain all)."""
    for tau in GRID:
        r = e42.risk_at(cases, tau, t)
        if r["cp95"] is not None and r["cp95"][1] <= BUDGET:
            return {"threshold": tau, "feasible": True, "cal": r}
    return {"threshold": None, "feasible": False}


def pmax(c: dict, t: float) -> float:
    return max(e42.temp_probs(c["probs"], t))


def auroc(cases: list[dict], t: float) -> float | None:
    """P(max-prob of a correct case > that of a wrong one), ties half."""
    pos = [pmax(c, t) for c in cases if c["pred"] == c["gold"]]
    neg = [pmax(c, t) for c in cases if c["pred"] != c["gold"]]
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    while i < len(allv):  # average ranks for ties
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for k in range(i, j):
            ranks.setdefault(k, (i + j + 1) / 2)
        i = j
    r_pos = sum(ranks[k] for k, (_, lab) in enumerate(allv) if lab == 1)
    return round((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), 4)


def cluster_bootstrap(cases: list[dict], tau: float, t: float, n: int = 1000, seed: int = 42) -> list[float] | None:
    by: dict[str, list[dict]] = {}
    for c in cases:
        by.setdefault(c["group_id"], []).append(c)
    groups, rng, risks = list(by.values()), random.Random(seed), []
    for _ in range(n):
        acc = [c for g in (rng.choice(groups) for _ in groups) for c in g if pmax(c, t) >= tau]
        if acc:
            risks.append(sum(c["pred"] != c["gold"] for c in acc) / len(acc))
    if not risks:
        return None
    risks.sort()
    return [round(risks[int(0.025 * len(risks))], 4), round(risks[int(0.975 * len(risks)) - 1], 4)]


def report_seed(seed: int, split_of: dict[str, str]) -> dict:
    cal = load(ROOT / f"runs/e43/percase_s{seed}__gui_cal.jsonl")
    cal_t = [c for c in cal if split_of[c["id"]] == "cal_temperature"]
    cal_p = [c for c in cal if split_of[c["id"]] == "cal_policy"]
    swap = load(ROOT / f"runs/e43/percase_s{seed}__swap.jsonl")
    held = load(ROOT / f"runs/e42-calib/percase_s{seed}__gui.jsonl")
    base = json.loads((ROOT / "runs/e42-calib/e42_summary.json").read_text(encoding="utf-8"))["seeds"][str(seed)]

    t = e42.fit_T(cal_t)
    tau = pick_tau_ext(cal_p, t)
    out = {
        "n": {"cal_temperature": len(cal_t), "cal_policy": len(cal_p), "swap": len(swap), "held_out_gui": len(held)},
        "T_screens": t, "tau_screens": tau,
        "baseline_e42": {"T": base["T"], "tau": base["tau"]["threshold"],
                         "held_out": e42.risk_at(held, base["tau"]["threshold"], base["T"])},
        "raw_acc": {"cal_policy": round(sum(c["pred"] == c["gold"] for c in cal_p) / len(cal_p), 4),
                    "held_out_gui": round(sum(c["pred"] == c["gold"] for c in held) / len(held), 4)},
        "ece15": {"cal_policy_raw": e42.ece15(cal_p, 1.0), "cal_policy_T": e42.ece15(cal_p, t),
                  "held_out_raw": e42.ece15(held, 1.0), "held_out_T": e42.ece15(held, t)},
        "auroc_cal_policy": auroc(cal_p, t),
        "cal_policy_curve": [{"tau": g, **e42.risk_at(cal_p, g, t)} for g in GRID],
        "swap": {"accuracy": round(sum(c["pred"] == c["gold"] for c in swap) / len(swap), 4),
                 "answered_true_at_tau": (round(sum(1 for c in swap if c["pred"] != c["gold"]
                                                    and pmax(c, t) >= tau["threshold"]) / len(swap), 4)
                                          if tau["feasible"] else None)},
    }
    if tau["feasible"]:
        h = e42.risk_at(held, tau["threshold"], t)
        out["held_out"] = h
        out["held_out_cluster_boot95"] = cluster_bootstrap(held, tau["threshold"], t)
        out["claim_ok"] = bool(h["cp95"] is not None and h["cp95"][1] <= BUDGET)
    else:
        out["held_out"] = {"coverage": 0.0, "note": "no feasible gate: screens policy abstains on every case"}
        out["claim_ok"] = False
    return out


def main() -> int:
    split_of = {r["id"]: r["split"] for r in load(ROOT / "data/e43/gui_cal/rows.jsonl")}
    seeds = {str(s): report_seed(s, split_of) for s in SEEDS}
    claim = all(v["claim_ok"] for v in seeds.values())
    cov = [v["held_out"].get("coverage", 0.0) for v in seeds.values()]
    swap_low = all(v["swap"]["accuracy"] < 0.5 for v in seeds.values())
    verdict = ("claimable" if claim and min(cov) >= 0.10 else
               "claimable, not useful (coverage < 10 %)" if claim else "not claimable")
    rep = {"criteria": "docs/E43_CRITERIA_2026-10-23.md", "grid": GRID, "budget": BUDGET,
           "seeds": seeds, "held_out_gui_claimable": claim, "verdict": verdict,
           "goal_grounding_reading": ("narrow the claim: model does not use the goal (swap accuracy < 0.5, both seeds)"
                                      if swap_low else "swap accuracy >= 0.5 on at least one seed")}
    out = ROOT / "runs/e43/e43_report.json"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    for s, v in seeds.items():
        h = v["held_out"]
        print(f"s{s}: T={v['T_screens']} tau={v['tau_screens']['threshold']} | held-out cov={h.get('coverage')} "
              f"risk={h.get('risk')} cp95={h.get('cp95')} | swap acc={v['swap']['accuracy']} | auroc={v['auroc_cal_policy']}")
    print("verdict:", verdict, "|", rep["goal_grounding_reading"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
