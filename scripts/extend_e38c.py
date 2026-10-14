"""E38c pool extension (criteria r2: docs/E38C_CRITERIA_2026-10-14-r2.md).

Extends the GUI-Odyssey pool beyond the first sha256-ordered 500 COMPLETE test
episodes so the kept-pairs target can be approached after positive filtering.

    python scripts/extend_e38c.py fin            # -> extend_pool.jsonl + _fin.png downloads
    python scripts/extend_e38c.py negs <keep.json>  # -> _neg.png for kept episodes only

Selection rule identical to gen_e38c.gen_gui (sha256 over row-minus-steps,
sort_keys=True), evaluated on the FULL COMPLETE test universe; episodes whose
_fin.png already exists are kept in the pool (review needed) but not re-
downloaded.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DATA = ROOT / "data/e38c/gui_odyssey"
GUI_REPO = "OpenGVLab/GUI-Odyssey"
GUI_RESOLVE = f"https://huggingface.co/datasets/{GUI_REPO}/resolve/main/"
START = 500  # episodes already sampled by gen_e38c


def parse_steps(value):
    for parse in (json.loads, ast.literal_eval):
        try:
            steps = parse(value or "[]")
            if isinstance(steps, list):
                return steps
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return []


def fetch(url: str, dest: Path, tries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for _ in range(tries):
        try:
            resp = requests.get(url, timeout=120)
        except requests.RequestException:
            time.sleep(2)
            continue
        if resp.status_code == 200 and resp.content:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            return True
        time.sleep(2)
    return False


def fin_pass() -> int:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    test_ids = {x.removesuffix(".json") for x in
                json.loads((DATA / "task_split.json").read_text(encoding="utf-8"))["test"]}
    print(f"streaming rows for COMPLETE test universe (test ids: {len(test_ids)})...", flush=True)
    universe = []
    for r in load_dataset(GUI_REPO, split="all", streaming=True):
        row = r["row"] if "row" in r else r
        eid = str(row.get("episode_id"))
        if eid not in test_ids:
            continue
        steps = parse_steps(row.get("steps"))
        if not steps or str(steps[-1].get("action") or "").upper() != "COMPLETE":
            continue
        universe.append({"row": {k: v for k, v in row.items() if k != "steps"},
                         "steps": steps, "eid": eid})
    keyed = sorted(universe, key=lambda x: hashlib.sha256(
        json.dumps(x["row"], sort_keys=True).encode("utf-8")).hexdigest())
    have = {p.name.split("_")[0] for p in (DATA / "shots").glob("*_fin.png")}
    overlap = sum(1 for x in keyed[:START] if x["eid"] in have)
    print(f"COMPLETE universe: {len(keyed)}; first-{START} overlap with downloaded: {overlap}/{START}",
          flush=True)
    print("listing repo tree...", flush=True)
    files = set(f.path for f in HfApi().list_repo_tree(GUI_REPO, repo_type="dataset", recursive=True))
    pool, dl, fail = [], 0, 0
    for x in keyed[START:]:
        row, steps, eid = x["row"], x["steps"], x["eid"]
        n = len(steps)
        rec = {"episode_id": eid, "goal": row.get("task", ""), "n_steps": n}
        if n < 2:
            rec.update(clean=False, reason="degenerate_single_step")
            pool.append(rec)
            continue
        k = int(hashlib.sha256(eid.encode()).hexdigest()[:15], 16) % (n // 2)
        neg_ref = str(steps[k].get("screenshot") or "")
        fin_ref = str(steps[-1].get("screenshot") or "")
        neg_paths = [p for p in files if p.endswith(neg_ref)]
        fin_paths = [p for p in files if p.endswith(fin_ref)]
        if not neg_paths or not fin_paths or neg_paths[0] == fin_paths[0]:
            rec.update(clean=False, reason="screenshot_ref_unresolvable", k=k)
            pool.append(rec)
            continue
        fin_local = f"shots/{eid}_fin.png"
        if fetch(GUI_RESOLVE + fin_paths[0], DATA / fin_local):
            dl += 1
        else:
            fail += 1
            rec.update(clean=False, reason="fin_download_failed", k=k)
            pool.append(rec)
            continue
        rec.update(clean=True, k=k, neg_ref=neg_ref, fin_ref=fin_ref,
                   neg_local=f"shots/{eid}_neg.png", fin_local=fin_local, steps=steps)
        pool.append(rec)
        if len(pool) % 50 == 0:
            print(f"  {len(pool)} pool entries, {dl} downloaded, {fail} failed", flush=True)
    clean_n = sum(1 for r in pool if r.get("clean"))
    (DATA / "extend_pool.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in pool), encoding="utf-8")
    print(f"pool entries: {len(pool)} clean={clean_n}; fin downloaded={dl} failed={fail}", flush=True)
    return 0


def negs_pass(keep_path: str) -> int:
    keep = set(json.loads(Path(keep_path).read_text(encoding="utf-8")))
    print("listing repo tree...", flush=True)
    from huggingface_hub import HfApi
    files = set(f.path for f in HfApi().list_repo_tree(GUI_REPO, repo_type="dataset", recursive=True))
    pool = [json.loads(l) for l in (DATA / "extend_pool.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    dl = fail = 0
    targets = [r for r in pool if r.get("clean") and r["episode_id"] in keep
               and not (DATA / f"shots/{r['episode_id']}_neg.png").exists()]
    for i, rec in enumerate(targets):
        cands = [p for p in files if p.endswith(rec["neg_ref"])]
        if cands and fetch(GUI_RESOLVE + cands[0], DATA / f"shots/{rec['episode_id']}_neg.png"):
            dl += 1
        else:
            fail += 1
            print(f"  NEG FAIL {rec['episode_id']}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(targets)} negs, {dl} ok, {fail} failed", flush=True)
    print(f"neg downloads: {dl} ok, {fail} failed, {len(targets)} targeted", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "negs":
        raise SystemExit(negs_pass(sys.argv[2]))
    raise SystemExit(fin_pass())
