"""E50 data (criteria: docs/E50_CRITERIA_2026-10-27.md).

    python scripts/gen_e50.py    # -> data/e50/m3t/{rows.jsonl, shots/}, data/e50/manifest.json

GUI-Odyssey task_split TEST COMPLETE episodes (E41 candidate procedure, full
test pool scanned), excluding every E38c episode; disjointness from E41, E43
and E49 episodes guard-asserted. Final screenshot only, first 400 fetchable
in sorted order form M3T: own goal (gold true) + E43-rule next-different-goal
swap (gold false), EN. Evaluation only, never trained on, read once per
checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_e38c import GUI_REPO, GUI_RESOLVE, SCR_QUESTION, SCR_STATE, _td_options, parse_steps, sha256_file  # noqa: E402
from gen_e41 import _get, case_row  # noqa: E402

M3T = ROOT / "data/e50/m3t"
N_EPISODES = 400


def gid(path: Path) -> set[str]:
    return {json.loads(l)["group_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def fetch_episodes() -> tuple[list[dict], dict]:
    from datasets import load_dataset
    from huggingface_hub import HfApi, get_token

    split = json.loads((ROOT / "data/e38c/gui_odyssey/task_split.json").read_text(encoding="utf-8"))
    test_ids = {x.removesuffix(".json") for x in split["test"]}
    e38c = gid(ROOT / "data/e38c/gui_odyssey/rows.jsonl")
    e41 = gid(ROOT / "data/e41/gui_odyssey/rows.jsonl")
    e43 = gid(ROOT / "data/e43/gui_cal/rows.jsonl")
    e49m = json.loads((ROOT / "data/e49/manifest.json").read_text(encoding="utf-8"))
    e49 = set(e49m["m3f_episodes"]) | set(e49m["train_episodes"])
    assert len(e38c) == 335 and len(e41) == 1200 and len(e43) == 400 and len(e49) == 2400
    assert not (e41 | e43 | e49) & test_ids, "guard: E41/E43/E49 episodes cannot be in the test split"

    cands, seen = [], set()
    for r in load_dataset(GUI_REPO, split="all", streaming=True):  # full test pool, no scan cap
        row = r["row"] if "row" in r else r
        eid = str(row.get("episode_id"))
        if eid not in test_ids or eid in seen:
            continue
        seen.add(eid)
        steps = parse_steps(row.get("steps"))
        if len(steps) >= 2 and str(steps[-1].get("action") or "").upper() == "COMPLETE":
            cands.append({"row": row, "steps": steps})
    cands.sort(key=lambda x: hashlib.sha256(json.dumps(
        {k: v for k, v in x["row"].items() if k != "steps"}, sort_keys=True).encode()).hexdigest())
    fresh = [c for c in cands if str(c["row"].get("episode_id")) not in e38c]
    print(f"candidates {len(cands)}, fresh after E38c exclusion {len(fresh)}", flush=True)

    by_name: dict[str, str] = {}
    for f in HfApi().list_repo_tree(GUI_REPO, repo_type="dataset", recursive=True):
        by_name.setdefault(f.path.rsplit("/", 1)[-1], f.path)
    token = get_token()

    def fetch(x):
        row, steps = x["row"], x["steps"]
        eid = str(row.get("episode_id"))
        remote = by_name.get(str(steps[-1].get("screenshot") or ""))
        goal = str(row.get("task") or "").strip()
        if not remote or not goal:
            return None
        p = M3T / "shots" / f"{eid}_fin{Path(remote).suffix}"
        if not p.exists():
            blob = _get(GUI_RESOLVE + remote, token)
            if blob is None:
                return None
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(blob)
        return {"eid": eid, "goal": goal, "file": p}

    got = []
    with ThreadPoolExecutor(8) as pool:
        for start in range(0, len(fresh), 200):
            got += [g for g in pool.map(fetch, fresh[start:start + 200]) if g]  # map keeps sorted order
            print(f"  {len(got)} episodes ready ({min(start + 200, len(fresh))}/{len(fresh)} tried)", flush=True)
            if len(got) >= N_EPISODES:
                break
    assert len(got) >= N_EPISODES, f"only {len(got)} fetchable test episodes (criteria wants {N_EPISODES})"
    got = got[:N_EPISODES]
    ids = {g["eid"] for g in got}
    assert len(ids) == N_EPISODES and not ids & (e38c | e41 | e43 | e49) and ids <= test_ids
    stats = {"candidates": len(cands), "fresh": len(fresh), "episodes_used": N_EPISODES,
             "revision": HfApi().dataset_info(GUI_REPO).sha,
             "exclusions_asserted": {"e38c": len(e38c), "e41": len(e41), "e43": len(e43), "e49": len(e49)}}
    return got, stats


def build_m3t(eps: list[dict]) -> list[dict]:
    rows = []
    for i, g in enumerate(eps):
        img = f"shots/{g['file'].name}"
        for d in range(1, len(eps)):  # the E43 swap rule: next different goal in set order
            other = eps[(i + d) % len(eps)]["goal"]
            if other.lower() != g["goal"].lower():
                break
        for kind, goal, gold in (("own", g["goal"], "true"), ("swap", other, "false")):
            rows.append(case_row(
                id=f"e50-m3t-{g['eid']}-{kind}-en", group_id=g["eid"], source="e50-m3t", language="en",
                domain="screens", state=SCR_STATE["task_done"]["en"].format(goal=goal),
                question=SCR_QUESTION["task_done"]["en"], options=_td_options("en"), target_option=gold,
                label_origin="e50-goal-swap", split="test",
                extra={"set": "m3t", "m3": kind, "family": "task_done" if kind == "own" else "task_done_swap",
                       "image": img, "own_goal": g["goal"], "goal": goal}))
    return rows


def main() -> int:
    got, stats = fetch_episodes()
    rows = build_m3t(got)

    # checks of the criteria
    assert len(rows) == 2 * N_EPISODES and all(r["own_goal"].lower() != r["goal"].lower() for r in rows if r["m3"] == "swap")
    assert {r["m3"] for r in rows} == {"own", "swap"} and {r["language"] for r in rows} == {"en"}
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)
    for r in rows:
        assert (M3T / r["image"]).is_file()

    M3T.mkdir(parents=True, exist_ok=True)
    (M3T / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    manifest = {
        "criteria": "docs/E50_CRITERIA_2026-10-27.md", "source": {"repo": GUI_REPO, **stats},
        "license": "CC BY 4.0 (GUI-Odyssey - Lu et al. 2024, OpenGVLab), evaluation only",
        "n_rows": len(rows), "rows_sha256": sha256_file(M3T / "rows.jsonl"),
        "m3t_episodes": [g["eid"] for g in got],
        "files_sha256": {p.relative_to(M3T).as_posix(): sha256_file(p)
                         for p in sorted((M3T / "shots").glob("*"))},
    }
    (ROOT / "data/e50/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_rows", "rows_sha256", "source")}, indent=1))
    for r in rows[:2]:
        print(" ", r["id"], r["target_option"], "|", r["state"][:90])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
