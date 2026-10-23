"""E43 real GUI CALIBRATION data (criteria: docs/E43_CRITERIA_2026-10-23.md).

    python scripts/gen_e43.py    # -> data/e43/gui_cal/{rows.jsonl, swap.jsonl, shots/}, manifest

GUI-Odyssey task_split TRAIN COMPLETE episodes, candidates chosen exactly as
E41 (scripts/gen_e41.py), minus every E41 episode (train and dev), every E38c
episode and every task_split test episode (asserted). The next 400 fetchable
episodes in E41's sorted order; rows built as E41 (neg/fin x EN/ES/DE).
Episodes split cal_temperature / cal_policy by sha256("e43|" + eid) % 2.
`swap`: each episode's fin screenshot with the next different goal, EN, gold false.
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

from gen_e38c import GUI_REPO, GUI_RESOLVE, LANGS, SCR_QUESTION, SCR_STATE, _td_options, parse_steps, sha256_file  # noqa: E402
from gen_e41 import N_GUI, _get, case_row  # noqa: E402

DATA = ROOT / "data/e43/gui_cal"
E38C = ROOT / "data/e38c"
E41 = ROOT / "data/e41"
N_E43 = 400


def cal_split(eid: str) -> str:
    return "cal_policy" if int(hashlib.sha256(f"e43|{eid}".encode()).hexdigest()[:8], 16) % 2 else "cal_temperature"


def main() -> int:
    from datasets import load_dataset
    from huggingface_hub import HfApi, get_token

    DATA.mkdir(parents=True, exist_ok=True)
    split = json.loads((E38C / "gui_odyssey/task_split.json").read_text(encoding="utf-8"))
    train_ids = {x.removesuffix(".json") for x in split["train"]}
    test_ids = {x.removesuffix(".json") for x in split["test"]}
    e38c_ids = {json.loads(l)["group_id"] for l in
                (E38C / "gui_odyssey/rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    e41_ids = {json.loads(l)["group_id"] for l in
               (E41 / "gui_odyssey/rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    assert len(e41_ids) == 1200, len(e41_ids)
    excluded = e41_ids | e38c_ids | test_ids

    # E41 candidate procedure, verbatim
    cands, seen = [], set()
    for r in load_dataset(GUI_REPO, split="all", streaming=True):
        row = r["row"] if "row" in r else r
        eid = str(row.get("episode_id"))
        if eid not in train_ids or eid in seen:
            continue
        seen.add(eid)
        steps = parse_steps(row.get("steps"))
        if len(steps) >= 2 and str(steps[-1].get("action") or "").upper() == "COMPLETE":
            cands.append({"row": row, "steps": steps})
        if len(cands) >= N_GUI * 3:
            break
    cands.sort(key=lambda x: hashlib.sha256(json.dumps(
        {k: v for k, v in x["row"].items() if k != "steps"}, sort_keys=True).encode()).hexdigest())
    fresh = [c for c in cands if str(c["row"].get("episode_id")) not in excluded]
    print(f"candidates {len(cands)}, fresh after exclusions {len(fresh)}", flush=True)

    by_name: dict[str, str] = {}
    for f in HfApi().list_repo_tree(GUI_REPO, repo_type="dataset", recursive=True):
        by_name.setdefault(f.path.rsplit("/", 1)[-1], f.path)
    token = get_token()

    def fetch(x):
        row, steps = x["row"], x["steps"]
        eid, n = str(row.get("episode_id")), len(steps)
        k = int(hashlib.sha256(eid.encode()).hexdigest()[:15], 16) % (n // 2)
        neg_ref, fin_ref = str(steps[k].get("screenshot") or ""), str(steps[-1].get("screenshot") or "")
        neg_remote, fin_remote = by_name.get(neg_ref), by_name.get(fin_ref)
        if not neg_remote or not fin_remote or neg_remote == fin_remote:
            return None
        out = {}
        for pol, remote in (("neg", neg_remote), ("fin", fin_remote)):
            rel = f"shots/{eid}_{pol}{Path(remote).suffix}"
            p = DATA / rel
            if not p.exists():
                blob = _get(GUI_RESOLVE + remote, token)
                if blob is None:
                    return None
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(blob)
            out[pol] = rel
        return {"eid": eid, "row": row, "k": k, "n": n, "neg_ref": neg_ref, "fin_ref": fin_ref, **out}

    got = []
    with ThreadPoolExecutor(8) as pool:
        for start in range(0, len(fresh), 100):
            got += [g for g in pool.map(fetch, fresh[start:start + 100]) if g]
            print(f"  {len(got)} episodes ready ({min(start + 100, len(fresh))}/{len(fresh)} tried)", flush=True)
            if len(got) >= N_E43:
                break
    got = got[:N_E43]
    assert len(got) == N_E43, len(got)

    rows, swap = [], []
    for g in got:
        goal = g["row"].get("task", "")
        for lang in LANGS:
            for pol, label in (("neg", "false"), ("fin", "true")):
                rows.append(case_row(
                    id=f"e43-gui-{g['eid']}-{pol}-{lang}", group_id=g["eid"], source="e43-gui",
                    language=lang, domain="screens",
                    state=SCR_STATE["task_done"][lang].format(goal=goal),
                    question=SCR_QUESTION["task_done"][lang], options=_td_options(lang),
                    target_option=label, label_origin="e43-gui-steps-unverified",
                    split=cal_split(g["eid"]),
                    extra={"family": "task_done", "image": g[pol], "polarity": pol, "k": g["k"],
                           "n_steps": g["n"], "neg_ref": g["neg_ref"], "fin_ref": g["fin_ref"],
                           "goal": goal}))
    for i, g in enumerate(got):
        goal = g["row"].get("task", "")
        for d in range(1, len(got)):
            other = got[(i + d) % len(got)]["row"].get("task", "")
            if other.strip().lower() != goal.strip().lower():
                break
        swap.append(case_row(
            id=f"e43-swap-{g['eid']}-fin-en", group_id=g["eid"], source="e43-swap",
            language="en", domain="screens",
            state=SCR_STATE["task_done"]["en"].format(goal=other),
            question=SCR_QUESTION["task_done"]["en"], options=_td_options("en"),
            target_option="false", label_origin="e43-goal-swap",
            split="test",  # contract split names are fixed; the set is marked below, never used to fit
            extra={"set": "swap", "family": "task_done_swap", "image": g["fin"], "polarity": "fin_swapped",
                   "own_goal": goal, "swapped_goal": other}))

    # asserts of the criteria
    ids = {r["group_id"] for r in rows}
    assert len(ids) == N_E43 and not ids & excluded and ids <= train_ids
    for r in rows:
        assert r["target_option"] == ("true" if r["polarity"] == "fin" else "false")
        assert r["k"] < r["n_steps"] // 2 and (DATA / r["image"]).is_file()
    for s in swap:
        assert s["own_goal"].strip().lower() != s["swapped_goal"].strip().lower()

    (DATA / "rows.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    (DATA / "swap.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in swap), encoding="utf-8")
    manifest = {
        "criteria": "docs/E43_CRITERIA_2026-10-23.md",
        "source": {"repo": GUI_REPO, "split": "task_split train", "revision": HfApi().dataset_info(GUI_REPO).sha},
        "license": "CC BY 4.0 (GUI-Odyssey - Lu et al. 2024, OpenGVLab)",
        "stats": {"candidates": len(cands), "fresh": len(fresh), "episodes_used": len(got)},
        "n_rows": {sp: sum(r["split"] == sp for r in rows) for sp in ("cal_temperature", "cal_policy")} | {"swap": len(swap)},
        "n_episodes": {sp: len({r["group_id"] for r in rows if r["split"] == sp}) for sp in ("cal_temperature", "cal_policy")},
        "exclusions_asserted": {"e41_episodes": len(e41_ids), "e38c_episodes": len(e38c_ids), "task_split_test": len(test_ids)},
        "rows_sha256": {"rows": sha256_file(DATA / "rows.jsonl"), "swap": sha256_file(DATA / "swap.jsonl")},
        "files_sha256": {p.relative_to(DATA).as_posix(): sha256_file(p) for p in sorted((DATA / "shots").glob("*"))},
    }
    (DATA.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("source", "stats", "n_rows", "n_episodes", "rows_sha256")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
