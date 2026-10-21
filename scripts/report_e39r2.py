"""Post-hoc E39-r2 report (frozen criteria docs/E39R2_CRITERIA_2026-10-22.md).

Joins the comparators' per-case rows with the OFFICIAL E39 v7 A1-V per-case
files and applies the frozen functions (paired_delta_ci, accuracy_ci_by_group,
ece_15; 1,000 group resamples, seed 42). Nothing here is fitted.

    python scripts/report_e39r2.py runs/e39r2 runs/e39-train-v7
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mksh_v.train_compare import accuracy_ci_by_group, ece_15, paired_delta_ci  # noqa: E402


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def acc_ci(rows: list[dict]) -> dict:
    return accuracy_ci_by_group([r["correct"] for r in rows], [r["group_id"] for r in rows])


def delta(a1v: dict[str, dict], comp: list[dict]) -> dict:
    """Paired Δ(A1-V − comparator) by episode, over the comparator's rows."""
    return paired_delta_ci([int(a1v[r["id"]]["correct"]) for r in comp],
                           [r["correct"] for r in comp],
                           [r["group_id"] for r in comp], n_boot=1000, seed=42)


def per_family(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["family"]].append(r)
    return dict(sorted(out.items()))


def main() -> int:
    r2, v7 = Path(sys.argv[1]), Path(sys.argv[2])
    a1v = {name: {r["id"]: r for r in load(v7 / f"eval_{name}__a1v.jsonl")}
           for name in ("test_nonfr", "test_fr", "real")}
    rep: dict = {"criteria": "docs/E39R2_CRITERIA_2026-10-22.md",
                 "summary": json.loads((r2 / "e39r2_summary.json").read_text(encoding="utf-8"))}

    smol = {}
    for name, key in (("test_nonfr", "smol_test_nonfr"), ("test_fr", "smol_test_fr"),
                      ("real", "smol_real")):
        f = r2 / f"{key}.jsonl"
        if not f.exists():
            continue
        rows = load(f)
        for r in rows:  # the v7 A1-V file supplies family/group for real rows
            r.setdefault("group_id", a1v[name][r["id"]].get("group_id", r["id"]))
        smol[name] = {"acc_ci": acc_ci(rows), "delta_a1v_minus_smol": delta(a1v[name], rows)}
        if name == "test_nonfr":
            smol[name]["per_family"] = {
                f: {"acc": acc_ci(rs), "delta_a1v_minus_smol": delta(a1v[name], rs)}
                for f, rs in per_family(rows).items()}
    if smol:
        d = smol["test_nonfr"]["delta_a1v_minus_smol"]
        smol["claim_rule"] = ("SmolVLM2 clearly better: declared, no 'best small visual "
                              "model' claim" if d["ci95"][1] < 0 else "no SmolVLM2 claim triggered")
    rep["smol"] = smol

    jev = {}
    for name, key in (("sample", "jev_sample"), ("real", "jev_real")):
        f = r2 / f"{key}.jsonl"
        if not f.exists():
            continue
        rows = load(f)
        ref = a1v["test_nonfr" if name == "sample" else "real"]
        ans = [r for r in rows if r["answered"]]
        jev[name] = {"n": len(rows), "answered": len(ans), "acc_ci": acc_ci(rows),
                     "ece15_answered": ece_15([max(r["probabilities"]) for r in ans],
                                              [r["correct"] for r in ans]),
                     "delta_a1v_minus_jev": delta(ref, rows),
                     "errors": sorted({r["error"][:80] for r in rows if not r["answered"]})[:5]}
        if name == "sample":
            jev[name]["per_family"] = {f: acc_ci(rs) for f, rs in per_family(rows).items()}
    jev["framing"] = ("zero-shot: an existing ~15x larger model without adaptation; "
                      "reference, not an equal-footing comparison")
    rep["jev"] = jev

    out = r2 / "e39r2_report.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("smol", "jev")}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
