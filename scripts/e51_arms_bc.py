"""E51 Arms B and C analysis: screens-domain ensemble gate + soup smoke.

    .venv/Scripts/python.exe -I scripts/e51_arms_bc.py

Reads the regenerated gui_cal per-case (runs/e51/percase_e47s4X__gui_cal.jsonl),
the soup smoke per-case (runs/e51/percase_soup__smoke.jsonl), and the E48 per-case
files. Fits T_scr/tau_scr on the ensemble gui_cal by the row split field with
E43's extended grid; G4 evaluates once on the E48 held-out gui. Writes
runs/e51/e51_arms_bc.json.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
import e42_calib as K  # noqa: E402

GRID_E43 = [round(0.30 + 0.05 * i, 2) for i in range(14)] + [0.96, 0.97, 0.98, 0.99, 0.995, 0.999]


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in (ROOT / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def ens_pair(pa: list[dict], pb: list[dict]) -> list[dict]:
    assert [r["id"] for r in pa] == [r["id"] for r in pb]
    out = []
    for ra, rb in zip(pa, pb):
        assert ra["gold"] == rb["gold"], ra["id"]
        p = [(x + y) / 2 for x, y in zip(ra["probs"], rb["probs"])]
        out.append({"id": ra["id"], "gold": ra["gold"], "probs": p,
                    "pred": max(range(len(p)), key=lambda i: p[i])})
    return out


def pick_tau_ext(cases: list[dict], t: float) -> dict:
    for tau in GRID_E43:
        r = K.risk_at(cases, tau, t)
        if r["cp95"] is not None and r["cp95"][1] <= 0.05:
            return {"threshold": tau, "cal_risk": r["risk"], "cal_cp95": r["cp95"], "feasible": True}
    return {"threshold": None, "feasible": False}


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den


def main() -> None:
    res = {}
    # ---- Arm B ----
    split_of = {r["id"]: r["split"] for r in load("data/e43/gui_cal/rows.jsonl")}
    g42 = load("runs/e51/percase_e47s42__gui_cal.jsonl")
    g43 = load("runs/e51/percase_e47s43__gui_cal.jsonl")
    ens = ens_pair(g42, g43)
    cal_t = [c for c in ens if split_of[c["id"]] == "cal_temperature"]
    cal_p = [c for c in ens if split_of[c["id"]] == "cal_policy"]
    T = K.fit_T(cal_t)
    tau = pick_tau_ext(cal_p, T)
    res["arm_b_fit"] = {"T_scr": T, "n_cal_t": len(cal_t), "n_cal_p": len(cal_p), **tau}
    if tau["feasible"]:
        gui_ens = ens_pair(load("runs/e48/percase_s42__gui.jsonl"), load("runs/e48/percase_s43__gui.jsonl"))
        r = K.risk_at(gui_ens, tau["threshold"], T)
        res["arm_b_heldout"] = r
        res["arm_b_G4"] = bool(r["cp95"] and r["cp95"][1] <= 0.05 and r["coverage"] >= 0.30)
        curve = [{"tau": g, **K.risk_at(gui_ens, g, T)} for g in GRID_E43]
        res["arm_b_curve"] = curve
        # descriptive: single seeds at the same operating point
        for seed in (42, 43):
            rows = [dict(c, pred=c["pred"]) for c in load(f"runs/e51/percase_e47s{seed}__gui_cal.jsonl")]
            res[f"arm_b_s{seed}_cal"] = K.risk_at(rows, tau["threshold"], T)

    # ---- Arm C smoke ----
    smoke = load("runs/e51/percase_soup__smoke.jsonl")
    e48t = {c["id"]: c for c in load("runs/e48/percase_s42__cal_temperature.jsonl")}
    e48u = {c["id"]: c for c in load("runs/e48/percase_s43__cal_temperature.jsonl")}
    xs, ys, agree, bad = [], [], 0, 0
    for c in smoke:
        a, b = e48t[c["id"]], e48u[c["id"]]
        pen = [(x + y) / 2 for x, y in zip(a["probs"], b["probs"])]
        xs.append(max(pen))
        ys.append(max(c["probs"]))
        agree += int(max(range(len(pen)), key=lambda i: pen[i]) == max(range(len(c["probs"])), key=lambda i: c["probs"][i]))
        bad += int(any(p != p or abs(p) == float("inf") for p in c["probs"]))  # NaN/inf
    sp = spearman(xs, ys)
    res["arm_c_smoke"] = {"n": len(smoke), "spearman": round(sp, 4),
                          "argmax_agreement": round(agree / len(smoke), 4), "nan_rows": bad,
                          "pass": bool(sp >= 0.9 and agree / len(smoke) >= 0.95 and bad == 0)}

    out = ROOT / "runs/e51"
    out.mkdir(exist_ok=True)
    (out / "e51_arms_bc.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "arm_b_curve"}, indent=1))


if __name__ == "__main__":
    main()
