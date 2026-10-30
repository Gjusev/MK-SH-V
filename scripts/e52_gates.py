"""E52 gate computation (criteria docs/E52_CRITERIA_2026-10-29.md).

    .venv/Scripts/python.exe -I scripts/e52_gates.py

Reads the e52 per-case files (kernel evals + local evals) and the E47/E48
references; prints the frozen-gate table and writes runs/e52/e52_gates.json.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs"
BASELINE = {42: 0.3433, 43: 0.3312}


def load(p: str) -> list[dict]:
    rows = [json.loads(l) for l in (R / p).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for r in rows:  # kernel schema -> e43_infer schema
        if "pred" not in r:
            r = {**r, "pred": r["pred_idx"], "gold": r["gold_idx"]}
        out.append(r)
    return out


def acc(rows):
    return round(sum(1 for r in rows if r["pred"] == r["gold"]) / len(rows), 4)


def split_real(rows):
    gui = [r for r in rows if not r["id"].startswith("e38c-cord")]
    cord = [r for r in rows if r["id"].startswith("e38c-cord")]
    return gui, cord


def main() -> None:
    out = {"gates": {}, "seeds": {}}
    frows = {r["id"]: r for r in (json.loads(l) for l in
             (ROOT / "data/e52/receipts/fresh_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    for s in (42, 43):
        d = {}
        d["test_nonfr"] = acc(load(f"e52/runs/a1v_e52_s{s}/eval_test_nonfr.jsonl"))
        gui, cord = split_real(load(f"e52/runs/a1v_e52_s{s}/eval_real.jsonl"))
        d["gui"], d["cord"] = acc(gui), acc(cord)
        d["fresh"] = acc(load(f"e52/percase_e52s{s}__fresh.jsonl"))
        d["task"] = acc(load(f"e52/percase_e52s{s}__task.jsonl"))
        d["testC"] = acc(load(f"e52/percase_e52s{s}__testC.jsonl"))
        d["fresh_ft"] = acc(load(f"e52/percase_e52s{s}__fresh_ft.jsonl"))
        m3 = load(f"e52/m3_e52s{s}.jsonl")
        d["m3_own"] = acc([r for r in m3 if r.get("family") == "task_done"])
        d["m3_swap"] = acc([r for r in m3 if r.get("family") == "task_done_swap"])

        ref = {"test_nonfr": acc(load(f"e47/runs/a1v_e47_s{s}/eval_test_nonfr.jsonl"))}
        g47, c47 = split_real(load(f"e47/runs/a1v_e47_s{s}/eval_real.jsonl"))
        ref["gui"], ref["cord"] = acc(g47), acc(c47)
        ref["task"] = acc(load(f"e47/runs/a1v_e47_s{s}/eval_task.jsonl"))
        ref["testC"] = acc(load(f"e47/runs/a1v_e47_s{s}/eval_testC.jsonl"))
        ref["fresh_ft"] = acc(load(f"e48/percase_s{s}__fresh_ft.jsonl"))
        m3r = load(f"e47/m3_s{s}.jsonl")
        ref["m3_own"] = acc([r for r in m3r if r.get("family") == "task_done"])
        ref["m3_swap"] = acc([r for r in m3r if r.get("family") == "task_done_swap"])

        deltas = {k: round((d[k] - ref[k]) * 100, 2) for k in ref}
        by_fmt = {}
        for r in load(f"e52/percase_e52s{s}__fresh.jsonl"):
            f = frows[r["id"]]["fmt"]
            by_fmt.setdefault(f, [0, 0])
            by_fmt[f][0] += int(r["pred"] == r["gold"])
            by_fmt[f][1] += 1
        out["seeds"][f"s{s}"] = {"e52": d, "e47_ref": ref, "delta_pp": deltas,
                                 "fresh_by_fmt": {k: [v[0], v[1], round(v[0] / v[1], 3)] for k, v in by_fmt.items()}}

    g1 = all(out["seeds"][f"s{s}"]["e52"]["fresh"] - BASELINE[s] >= 0.04 for s in (42, 43))
    ok2 = True
    for s in (42, 43):
        e = out["seeds"][f"s{s}"]
        ok2 &= e["delta_pp"]["test_nonfr"] >= -2.0 and e["delta_pp"]["gui"] >= -3.0
        ok2 &= e["delta_pp"]["task"] >= -3.0 and e["delta_pp"]["testC"] >= -3.0
        ok2 &= e["delta_pp"]["fresh_ft"] >= -2.0 and e["delta_pp"]["cord"] >= -3.0
    g3 = all(out["seeds"][f"s{s}"]["delta_pp"]["m3_swap"] >= -3.0
             and out["seeds"][f"s{s}"]["delta_pp"]["m3_own"] >= -3.0 for s in (42, 43))
    out["gates"] = {"G1_delta_plus4pp": bool(g1), "G2_no_regression": bool(ok2), "G3_goal_guard": bool(g3)}
    (R / "e52/e52_gates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["gates"]))
    for s in (42, 43):
        e = out["seeds"][f"s{s}"]
        print(f"-- s{s} fresh {e['e52']['fresh']} (baseline {BASELINE[s]}, "
              f"delta {round((e['e52']['fresh'] - BASELINE[s]) * 100, 1)}pp) | by_fmt {e['fresh_by_fmt']}")
        print("   deltas_pp:", e["delta_pp"])


if __name__ == "__main__":
    main()
