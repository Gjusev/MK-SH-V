"""Write release/inference-policy.json from the E42 summary (deterministic).

    python scripts/make_policy.py runs/e42-calib/e42_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RESPONSE_SEMANTICS = {
    "decide": "softmax(logits / T) over the valid options; argmax is the choice",
    "abstain": "max calibrated probability < tau -> choice null, reason_code below_validated_threshold",
    "reason_codes": {"below_validated_threshold":
                     "max calibrated probability below tau; the 5 % selective-risk budget "
                     "was validated on the held-out synthetic test (E42), both seeds"},
}


def main() -> int:
    summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    seed = sys.argv[2] if len(sys.argv) > 2 else "42"
    v = summary["seeds"][seed]
    policy = {
        "model": "MK-SH-V (A1-V)",
        "checkpoint": f"a1v_real_s{seed}",
        "checkpoint_sha256": v["sha256"],
        "experiment": "E42 (docs/E42_CRITERIA_2026-10-22.md)",
        "temperature": {
            "value": v["T"],
            "fitted_on": "cal_temperature (3,150 synthetic calibration cases), NLL grid",
            "crosscheck_mokka": v["T_mokka_crosscheck"],
        },
        "abstention": {
            "threshold": v["tau"]["threshold"],
            "rule": "smallest grid value whose cal_policy CP95 UPPER bound of selective risk <= 0.05",
            "risk_budget": 0.05,
            "fitted_on": "cal_policy (3,150), disjoint from every reported set",
            "held_out_validation": {
                "set": "synthetic test non-FR (4,200)",
                "coverage": v["tables"]["test_nonfr"]["overall"]["coverage"],
                "selective_risk": v["tables"]["test_nonfr"]["overall"]["risk"],
                "cp95": v["tables"]["test_nonfr"]["overall"]["cp95"],
            },
        },
        "tau_grid": summary["tau_grid"],
        "response_semantics": RESPONSE_SEMANTICS,
        "bounds": "guarantee protocol- and distribution-specific (synthetic families); "
                  "real-image abstention not validated (see model card)",
    }
    out = ROOT / "release/inference-policy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(out, json.dumps({k: policy[k] for k in ("checkpoint", "temperature", "abstention")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
