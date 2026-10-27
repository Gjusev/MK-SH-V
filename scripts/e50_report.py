"""E50 report (criteria: docs/E50_CRITERIA_2026-10-27.md). Measurement only.

    python scripts/e50_report.py    # -> runs/e50/e50_report.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e44_probe_report import load, p_true  # noqa: E402

SEEDS = (42, 43)
CKPT = {  # pinned in the criteria, sha256-verified by e43_infer before inference
    "e47": {42: ("runs/e47/runs/a1v_e47_s42/best.pt",
                 "8b166fb995069d0f80f95aec9fd759ef47a72fc0cde1338ca453b8adb6d19452"),
            43: ("runs/e47/runs/a1v_e47_s43/best.pt",
                 "afcbeaa45a4c033b5e7d3f033b85fdda2642c0fef78397fcecc5bef53cd83c29")},
    "e49": {42: ("runs/e49/runs/a1v_e49_s42/best.pt",
                 "0f6aa8603847fc765983967422b91b8d249fc2323911744936b8e8ae3ee7d621"),
            43: ("runs/e49/runs/a1v_e49_s43/best.pt",
                 "4aa15062d33bf069ba64354115a50c5279b4a2a51b07e400a697add3e2b1ef4d")}}
OUTCOMES = {
    "1_and_2": "The E38c drop is not about goals on unseen tasks. E51 keeps the E49 recipe and looks for the GUI regression elsewhere.",
    "1_only": "The goal gain transfers but costs own-goal recall on unseen tasks. E51 targets recall on unseen tasks (selection on a disjoint test-pool dev set, more own-goal positives), gated on a fresh test-pool set.",
    "transfer_fails": "The E49 gain does not transfer to unseen task types. E51 must change the goal data itself (task diversity), not the selection.",
}


def measure(cases: list[dict], meta: dict[str, dict]) -> dict:
    """M3T per-checkpoint: swap/own accuracy, both-right episodes, swap acc without
    ' any ' goals (as e44_report.goal_sets), mean p(true) on own and swap rows."""
    ok: dict[str, dict[str, bool]] = defaultdict(dict)
    swap_specific: list[bool] = []
    pt = {"own": [], "swap": []}
    for c in cases:
        r = meta[c["id"]]
        correct = c["pred"] == c["gold"]
        ok[r["group_id"]][r["m3"]] = correct
        pt[r["m3"]].append(p_true(c, r))
        if r["m3"] == "swap":
            goal = r.get("goal") or r["state"]
            if " any " not in f" {goal.lower()} ":
                swap_specific.append(correct)
    assert all(set(v) == {"own", "swap"} for v in ok.values())
    n = len(ok)
    return {"episodes": n,
            "swap_acc": round(sum(v["swap"] for v in ok.values()) / n, 4),
            "swap_n": sum(v["swap"] for v in ok.values()),
            "own_acc": round(sum(v["own"] for v in ok.values()) / n, 4),
            "own_n": sum(v["own"] for v in ok.values()),
            "both_right": sum(v["own"] and v["swap"] for v in ok.values()),
            "swap_without_any": {"acc": round(sum(swap_specific) / len(swap_specific), 4),
                                 "n": len(swap_specific)},
            "mean_p_true": {k: round(sum(v) / len(v), 4) for k, v in pt.items()}}


def readings(runs: dict[str, dict[int, dict]]) -> dict:
    """The two criteria readings on exact counts (no rounding decides a gate)."""
    transfer = all(m["swap_n"] * 5 >= m["episodes"] * 4 for m in runs["e49"].values())  # >= 0.80
    own_cost = all(runs["e49"][s]["own_n"] * 10000
                   >= runs["e47"][s]["own_n"] * 10000 - 300 * runs["e47"][s]["episodes"]  # >= own - 3 pp
                   for s in runs["e47"] if s in runs["e49"])
    outcome = ("transfer_fails" if not transfer else "1_only" if not own_cost else "1_and_2")
    return {"transfer_e49_swap_ge_080_both_seeds": transfer,
            "own_cost_e49_ge_e47_minus_3pp_both_seeds": own_cost,
            "outcome": outcome, "outcome_text": OUTCOMES[outcome]}


def main() -> int:
    meta = {r["id"]: r for r in load(ROOT / "data/e50/m3t/rows.jsonl")}
    runs = {run: {s: measure(load(ROOT / f"runs/e50/m3t_{run}_s{s}.jsonl"), meta) for s in SEEDS}
            for run in CKPT}
    out = {"criteria": "docs/E50_CRITERIA_2026-10-27.md",
           "checkpoints": {run: {str(s): sha for s, (_, sha) in CKPT[run].items()} for run in CKPT},
           "runs": {run: {str(s): m for s, m in per.items()} for run, per in runs.items()},
           "readings": readings(runs)}
    (ROOT / "runs/e50").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs/e50/e50_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
