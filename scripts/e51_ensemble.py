"""E51 Arm A: two-seed ensemble policy on the E48 per-case files (offline, 0 GPU).

    python -I scripts/e51_ensemble.py

Criteria docs/E51_CRITERIA_2026-10-29.md. Fits T_ens/tau_ens by importing the
E42 protocol verbatim (kaggle/e42_calib.py), reads each held-out set once,
writes runs/e51/e51_summary.json and prints the gate table.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
import e42_calib as K  # noqa: E402  (protocol: temp_probs, fit_T, pick_tau, risk_at, ece15)

SETS = ["cal_temperature", "cal_policy", "test_nonfr", "test_fr", "testC",
        "fresh_ft", "cord", "gui", "task"]
HELD_OUT = ["test_nonfr", "test_fr", "testC", "fresh_ft", "cord", "gui", "task"]


def load_pair(name: str) -> list[dict]:
    a = [json.loads(l) for l in (ROOT / f"runs/e48/percase_s42__{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    b = [json.loads(l) for l in (ROOT / f"runs/e48/percase_s43__{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(a) == len(b) and [r["id"] for r in a] == [r["id"] for r in b], name
    rows = []
    for ra, rb in zip(a, b):
        assert ra["gold"] == rb["gold"], ra["id"]
        p = [(x + y) / 2 for x, y in zip(ra["probs"], rb["probs"])]
        rows.append({"id": ra["id"], "family": ra.get("family"), "gold": ra["gold"], "probs": p,
                     "pred": max(range(len(p)), key=lambda i: p[i])})
    return rows


def geomean_pair(name: str) -> list[dict]:  # descriptive-only ablation
    import math
    a = [json.loads(l) for l in (ROOT / f"runs/e48/percase_s42__{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    b = [json.loads(l) for l in (ROOT / f"runs/e48/percase_s43__{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for ra, rb in zip(a, b):
        logs = [(math.log(max(x, 1e-12)) + math.log(max(y, 1e-12))) / 2
                for x, y in zip(ra["probs"], rb["probs"])]
        m = max(logs)
        ex = [math.exp(x - m) for x in logs]
        s = sum(ex)
        p = [e / s for e in ex]
        rows.append({"id": ra["id"], "family": ra.get("family"), "gold": ra["gold"],
                     "probs": p, "pred": max(range(len(p)), key=lambda i: p[i])})
    return rows


def acc(rows: list[dict], t: float) -> float:
    return sum(1 for c in rows if max(range(len(c["probs"])), key=lambda i: K.temp_probs(c["probs"], t)[i]) == c["gold"]) / len(rows)


def run_arm(loader, label: str) -> dict:
    data = {n: loader(n) for n in SETS}
    T = K.fit_T(data["cal_temperature"])
    tau = K.pick_tau(data["cal_policy"], T)
    out = {"label": label, "T": T, "tau": tau["threshold"], "tau_met_target": tau["met_target"],
           "sets": {}}
    for n in HELD_OUT:
        r = K.risk_at(data[n], tau["threshold"], T)
        out["sets"][n] = {"n": r["n"], "full_acc": round(acc(data[n], T), 4),
                          "coverage": r["coverage"], "risk": r["risk"], "cp95": r["cp95"],
                          "meets_budget": r["meets_budget"]}
    out["ece15_cal_temperature"] = K.ece15(data["cal_temperature"], T)
    return out


def main() -> None:
    res = {"primary": run_arm(load_pair, "prob_avg"),
           "ablation_logit_avg_descriptive_only": run_arm(geomean_pair, "logit_avg"),
           "single_seed_reference": {}}
    # single-seed full accuracy on the same files (descriptive; no refit)
    for seed in (42, 43):
        res["single_seed_reference"][f"s{seed}"] = {}
        for n in HELD_OUT:
            rows = [json.loads(l) for l in (ROOT / f"runs/e48/percase_s{seed}__{n}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            res["single_seed_reference"][f"s{seed}"][n] = {
                "full_acc": round(sum(1 for c in rows if c["pred"] == c["gold"]) / len(rows), 4)}

    p = res["primary"]
    g1 = p["sets"]["test_nonfr"]["full_acc"] >= 0.9488
    g2 = p["sets"]["test_nonfr"]["meets_budget"] and p["sets"]["test_nonfr"]["coverage"] >= 0.90
    s43 = res["single_seed_reference"]["s43"]
    g3 = all(p["sets"][n]["full_acc"] >= s43[n]["full_acc"] - 0.02 for n in HELD_OUT)
    res["gates"] = {"G1_floor_0.9488": bool(g1), "G2_gate_holds": bool(g2),
                    "G3_no_regression_vs_s43_2pp": bool(g3)}

    out = ROOT / "runs/e51"
    out.mkdir(exist_ok=True)
    (out / "e51_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    print(f"ens  T={p['T']}  tau={p['tau']} (met_target={p['tau_met_target']})")
    print(f"{'set':12} {'full_acc':>8} {'cov':>7} {'risk':>6} {'cp95u':>6}   s43_acc")
    for n in HELD_OUT:
        s = p["sets"][n]
        print(f"{n:12} {s['full_acc']:>8} {s['coverage']:>7} {s['risk']:>6} "
              f"{s['cp95'][1]:>6}   {s43[n]['full_acc']}")
    print("GATES:", res["gates"])


if __name__ == "__main__":
    main()
