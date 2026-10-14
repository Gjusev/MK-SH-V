"""E38c probe r2: GUI-Odyssey negative-class construction (criteria:
docs/E38C_PROBE_CRITERIA_2026-10-14-r2.md).

    python scripts/probe_e38c_r2.py    # -> data/e38c-probe/gui_odyssey/r2_*, runs/e38c-probe/probe_r2.json
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_e38c import fetch_rows, sha256_file  # noqa: E402

REPO = "OpenGVLab/GUI-Odyssey"
DEST = ROOT / "data/e38c-probe/gui_odyssey"
N_SAMPLE = 100
N_INSPECT = 20
REPO_API = "https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey/resolve/main/"


def parse_steps(row: dict) -> list:
    for parse in (json.loads, ast.literal_eval):
        try:
            steps = parse(row.get("steps") or "[]")
            if isinstance(steps, list):
                return steps
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return []

def fetch_shot(path: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        resp = requests.get(REPO_API + path, timeout=120)
    except requests.RequestException:
        return False
    if resp.status_code != 200:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return True


def main() -> int:
    from huggingface_hub import HfApi

    RUNS = ROOT / "runs/e38c-probe"
    RUNS.mkdir(parents=True, exist_ok=True)
    print("fetching rows...", flush=True)
    rows = fetch_rows(REPO, "default", "all")
    complete = []
    for r in rows:
        steps = parse_steps(r["row"])
        if steps and str(steps[-1].get("action") or "").upper() == "COMPLETE":
            complete.append((r, steps))
    keyed = sorted(complete, key=lambda t: hashlib.sha256(
        json.dumps(t[0], sort_keys=False).encode("utf-8")).hexdigest())
    sample = keyed[:N_SAMPLE]
    print(f"{len(rows)} rows, {len(complete)} COMPLETE episodes, sampling {len(sample)}", flush=True)

    files = set()
    for f in HfApi().list_repo_tree(REPO, repo_type="dataset", recursive=True):
        files.add(f.path)
    shots = DEST / "r2_shots"
    out_rows, clean_ids = [], []
    for r, steps in sample:
        row = r["row"]
        eid = str(row.get("episode_id"))
        n = len(steps)
        rec = {"episode_id": eid, "row_idx": r["row_idx"], "n_steps": n}
        if n < 2:
            rec.update(clean=False, reason="degenerate_single_step")
            out_rows.append(rec)
            continue
        k = int(hashlib.sha256(eid.encode()).hexdigest()[:15], 16) % (n // 2)
        neg_ref, fin_ref = str(steps[k].get("screenshot") or ""), str(steps[-1].get("screenshot") or "")
        neg_paths = [p for p in files if p.endswith(neg_ref)]
        fin_paths = [p for p in files if p.endswith(fin_ref)]
        if not neg_paths or not fin_paths:
            rec.update(clean=False, reason="screenshot_missing_in_repo", k=k, neg_ref=neg_ref, fin_ref=fin_ref)
            out_rows.append(rec)
            continue
        if neg_paths[0] == fin_paths[0]:
            rec.update(clean=False, reason="negative_equals_final", k=k)
            out_rows.append(rec)
            continue
        neg_ok = fetch_shot(neg_paths[0], shots / f"{eid}_neg.png")
        fin_ok = fetch_shot(fin_paths[0], shots / f"{eid}_fin.png")
        if not (neg_ok and fin_ok):
            rec.update(clean=False, reason="screenshot_download_failed", k=k)
            out_rows.append(rec)
            continue
        rec.update(clean=True, k=k, label_pair={"negative": "false", "positive": "true"},
                   group_id=eid, neg_path=neg_paths[0], fin_path=fin_paths[0],
                   neg_local=f"r2_shots/{eid}_neg.png", fin_local=f"r2_shots/{eid}_fin.png")
        out_rows.append(rec)
        clean_ids.append(eid)
    clean_n = len(clean_ids)
    inspect_ids = sorted(random.Random(7).sample(clean_ids, min(N_INSPECT, len(clean_ids))))
    result = {"criteria": "docs/E38C_PROBE_CRITERIA_2026-10-14-r2.md", "rows_fetched": len(rows),
              "complete_episodes": len(complete), "sampled": len(sample), "clean": clean_n,
              "clean_share": round(clean_n / max(len(sample), 1), 4),
              "passed": clean_n / max(len(sample), 1) >= 0.70,
              "reasons": _reasons(out_rows), "inspection_ids": inspect_ids,
              "operator_verdicts": "pending"}
    (RUNS / "probe_r2.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (DEST / "r2_rows.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows),
                                        encoding="utf-8")
    stored = list(shots.glob("*.png"))
    (DEST / "r2_manifest.json").write_text(json.dumps(
        {"files_sha256": {p.name: sha256_file(p) for p in sorted(stored)}}, indent=2), encoding="utf-8")
    print(f"clean {clean_n}/{len(sample)} = {clean_n / max(len(sample), 1):.0%} -> "
          f"{'PASS' if clean_n / max(len(sample), 1) >= 0.70 else 'FAIL'}; "
          f"inspection pairs: {inspect_ids[:5]}...", flush=True)
    return 0


def _reasons(rows: list[dict]) -> dict:
    from collections import Counter
    return dict(Counter(r["reason"] for r in rows if not r.get("clean")))


if __name__ == "__main__":
    raise SystemExit(main())


