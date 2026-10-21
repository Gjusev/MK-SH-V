"""Post-hoc E41 gates (frozen criteria docs/E41_CRITERIA_2026-10-22.md).

    python scripts/report_e41.py runs/e41 runs/e39-train-v7 runs/e39r2

Paired Δ(E41 − E39 A1-V) by episode with the frozen `paired_delta_ci`
(1,000 group resamples, seed 42) against the official v7 per-case files;
jev (E39-r2) and OCR→MK-SH-1 (v7) are reported references, no gate.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mksh_v.train_compare import ece_15, paired_delta_ci  # noqa: E402

SEEDS = (42, 43)
TEXT_BARS = {"testC": 0.959, "task": 0.780}
G2_FLOOR = -0.02


def load(path: Path) -> dict[str, dict]:
    return {(r := json.loads(l))["id"]: r for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def delta(new: dict[str, dict], ref: dict[str, dict], ids: list[str]) -> dict:
    return paired_delta_ci([new[i]["correct"] for i in ids], [int(ref[i]["correct"]) for i in ids],
                           [new[i]["group_id"] for i in ids], n_boot=1000, seed=42)


def acc(rows: dict[str, dict], ids) -> float:
    ids = list(ids)
    return sum(rows[i]["correct"] for i in ids) / len(ids)


def main() -> int:
    e41, v7, r2 = map(Path, sys.argv[1:4])
    ref = {n: load(v7 / f"eval_{n}__a1v.jsonl") for n in ("test_nonfr", "real")}
    ocr_real, jev_real = load(v7 / "eval_real__ocr.jsonl"), load(r2 / "jev_real.jsonl")
    rep: dict = {"criteria": "docs/E41_CRITERIA_2026-10-22.md", "seeds": {}}
    for seed in SEEDS:
        d = e41 / "runs" / f"a1v_real_s{seed}"
        if not (d / "eval_real.jsonl").exists():
            rep["seeds"][seed] = {"status": "missing"}
            continue
        ev = {n: load(d / f"eval_{n}.jsonl") for n in ("real", "test_nonfr", "test_fr", "testC", "task")}
        real_ids, syn_ids = sorted(ev["real"]), sorted(ev["test_nonfr"])
        assert set(real_ids) == set(ref["real"]) and set(syn_ids) == set(ref["test_nonfr"])
        g1, g2 = delta(ev["real"], ref["real"], real_ids), delta(ev["test_nonfr"], ref["test_nonfr"], syn_ids)
        text = {k: acc(ev[k], ev[k]) for k in TEXT_BARS}
        by = defaultdict(list)
        for i in real_ids:
            src = "cord" if i.startswith("e38c-cord") else "gui"
            by[src].append(i)
            by[f"{src}-{i.rsplit('-', 1)[-1]}"].append(i)
        ans = list(ev["real"].values())
        rep["seeds"][seed] = {
            "summary": json.loads((d / "summary.json").read_text(encoding="utf-8")),
            "real_acc": acc(ev["real"], real_ids), "test_nonfr_acc": acc(ev["test_nonfr"], syn_ids),
            "test_fr_acc": acc(ev["test_fr"], ev["test_fr"]), "text": text,
            "G1_real_delta_vs_e39": g1, "G1": g1["ci95"][0] > 0,
            "G2_synthetic_delta_vs_e39": g2, "G2": g2["ci95"][0] >= G2_FLOOR,
            "G3": all(text[k] >= b for k, b in TEXT_BARS.items()),
            "real_by_slice": {k: {"n": len(v), "e41": round(acc(ev["real"], v), 4),
                                  "e39": round(acc(ref["real"], v), 4),
                                  "delta_vs_e39": delta(ev["real"], ref["real"], v)}
                              for k, v in sorted(by.items())},
            "ref_delta_vs_jev": delta(ev["real"], jev_real, real_ids),
            "ref_delta_vs_ocr": delta(ev["real"], ocr_real, real_ids),
            "real_ece15": ece_15([max(r["probabilities"]) for r in ans], [r["correct"] for r in ans]),
        }
    done = [v for v in rep["seeds"].values() if "G1" in v]
    gates = {g: (len(done) == len(SEEDS) and all(v[g] for v in done)) for g in ("G1", "G2", "G3")}
    rep["gates_all_seeds"] = gates
    rep["verdict"] = ("GO-E41" if all(gates.values()) else
                      "G1 fail: real gap not closed by this real data" if not gates["G1"] else
                      "G1 pass with declared cost (G2/G3 fail)")
    (e41 / "e41_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    for seed, v in rep["seeds"].items():
        if "G1" in v:
            print(seed, {k: v[k] for k in ("real_acc", "test_nonfr_acc", "text", "G1", "G2", "G3")},
                  "G1", [round(x, 4) for x in v["G1_real_delta_vs_e39"]["ci95"]],
                  "vs jev", [round(x, 4) for x in v["ref_delta_vs_jev"]["ci95"]])
    print(gates, rep["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
