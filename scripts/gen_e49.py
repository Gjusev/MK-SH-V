"""E49 data (criteria: docs/E49_CRITERIA_2026-10-26.md).

    python scripts/gen_e49.py   # -> data/e49/bind/{rows.jsonl, shots/, manifest.json}
                                #    data/e49/m3f/{rows.jsonl, shots/}

New GUI-Odyssey task_split TRAIN COMPLETE episodes (E41 candidate procedure,
scan widened to 9,000 candidates), excluding E41, E43, E38c and the test
split. Final screenshots only, in sorted order: the first 400 fetchable form
M3F (own + E43-rule swap, EN), the next 2,000 add goal / goal_match rows to
the E47 bind pool, which is otherwise kept byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_e38c import GUI_REPO, GUI_RESOLVE, LANGS, SCR_QUESTION, SCR_STATE, _td_options, parse_steps, sha256_file  # noqa: E402
from gen_e41 import _get, case_row  # noqa: E402
from gen_e45 import MATCH_Q, MATCH_STATE  # noqa: E402
from mksh_v.gen_e38a import stable_seed  # noqa: E402

E47_ROWS = ROOT / "data/e47/bind/rows.jsonl"
E47_SHA = "d3236aed192a4990b208729e95f49c4c4de0615b95c86e61b7f914dae09deac6"
OUT, M3F = ROOT / "data/e49/bind", ROOT / "data/e49/m3f"
CACHE = ROOT / "data/e49/fetch"  # downloaded final screenshots, one per episode
SCAN, N_M3F, N_TRAIN = 9000, 400, 2000


def gid(path: Path) -> set[str]:
    return {json.loads(l)["group_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def fetch_episodes() -> tuple[list[dict], dict]:
    from datasets import load_dataset
    from huggingface_hub import HfApi, get_token

    split = json.loads((ROOT / "data/e38c/gui_odyssey/task_split.json").read_text(encoding="utf-8"))
    train_ids = {x.removesuffix(".json") for x in split["train"]}
    test_ids = {x.removesuffix(".json") for x in split["test"]}
    e41, e43 = gid(ROOT / "data/e41/gui_odyssey/rows.jsonl"), gid(ROOT / "data/e43/gui_cal/rows.jsonl")
    e38c = gid(ROOT / "data/e38c/gui_odyssey/rows.jsonl")
    assert len(e41) == 1200 and len(e43) == 400, (len(e41), len(e43))
    excluded = e41 | e43 | e38c | test_ids

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
        if len(cands) >= SCAN:
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
        eid = str(row.get("episode_id"))
        fin_ref = str(steps[-1].get("screenshot") or "")
        remote = by_name.get(fin_ref)
        goal = str(row.get("task") or "").strip()
        if not remote or not goal:
            return None
        p = CACHE / f"{eid}_fin{Path(remote).suffix}"
        if not p.exists():
            blob = _get(GUI_RESOLVE + remote, token)
            if blob is None:
                return None
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(blob)
        return {"eid": eid, "goal": goal, "file": p, "fin_ref": fin_ref, "n_steps": len(steps)}

    need, got = N_M3F + N_TRAIN, []
    with ThreadPoolExecutor(8) as pool:
        for start in range(0, len(fresh), 200):
            got += [g for g in pool.map(fetch, fresh[start:start + 200]) if g]  # map keeps sorted order
            print(f"  {len(got)} episodes ready ({min(start + 200, len(fresh))}/{len(fresh)} tried)", flush=True)
            if len(got) >= need:
                break
    assert len(got) >= need, f"only {len(got)} fetchable episodes"
    got = got[:need]
    ids = {g["eid"] for g in got}
    assert len(ids) == need and not ids & excluded and ids <= train_ids
    stats = {"candidates": len(cands), "fresh": len(fresh), "episodes_used": need,
             "revision": HfApi().dataset_info(GUI_REPO).sha,
             "exclusions_asserted": {"e41": len(e41), "e43": len(e43), "e38c": len(e38c), "test": len(test_ids)}}
    return got, stats


def place(g: dict, root: Path) -> str:
    rel = f"shots/{g['eid']}_fin{g['file'].suffix}"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    if not (root / rel).exists():
        shutil.copyfile(g["file"], root / rel)
    return rel


def build_m3f(eps: list[dict]) -> list[dict]:
    rows = []
    for i, g in enumerate(eps):
        img = place(g, M3F)
        for d in range(1, len(eps)):  # the E43 swap rule: next different goal in set order
            other = eps[(i + d) % len(eps)]["goal"]
            if other.lower() != g["goal"].lower():
                break
        for kind, goal, gold in (("own", g["goal"], "true"), ("swap", other, "false")):
            rows.append(case_row(
                id=f"e49-m3f-{g['eid']}-{kind}-en", group_id=g["eid"], source="e49-m3f", language="en",
                domain="screens", state=SCR_STATE["task_done"]["en"].format(goal=goal),
                question=SCR_QUESTION["task_done"]["en"], options=_td_options("en"), target_option=gold,
                label_origin="e49-goal-swap", split="test",
                extra={"set": "m3f", "m3": kind, "family": "task_done" if kind == "own" else "task_done_swap",
                       "image": img, "own_goal": g["goal"], "goal": goal}))
    return rows


def build_bind(eps: list[dict], e41_goals: list[str]) -> list[dict]:
    from mokka_decisions.contracts import Option

    rows: list[dict] = []
    pool_all = sorted({*e41_goals, *(g["goal"] for g in eps)}, key=str.lower)

    def emit(pair, idx, kind, lang, image, state, question, options, gold, **extra):
        rows.append(case_row(
            id=f"e49-{pair}-{idx}", group_id=pair, source="e49-bind", language=lang, domain="screens",
            state=state, question=question, options=tuple(Option(id=i, description=d) for i, d in options),
            target_option=gold, label_origin="e49-bind", split="train",
            extra={"family": "task_done" if kind == "goal" else "goal_match", "bind": kind,
                   "pair_id": pair, "img_src": "bind", "image": image, **extra}))

    matches = []
    for i, g in enumerate(eps):
        img, own = place(g, OUT), g["goal"]
        rng = random.Random(stable_seed("e49-goal", g["eid"]))
        pool = [x for x in pool_all if x.lower() != own.lower()]
        lang = LANGS[i % len(LANGS)]
        other = rng.choice(pool)
        for k, (goal, gold) in enumerate(((own, "true"), (other, "false"))):
            emit(f"goal-e49-{g['eid']}", k, "goal", lang, img, SCR_STATE["task_done"][lang].format(goal=goal),
                 SCR_QUESTION["task_done"][lang], [(o.id, o.description) for o in _td_options(lang)], gold,
                 own_goal=own, goal=goal)
        others, pos = rng.sample(pool, 3), rng.randrange(4)
        matches.append((img, own, others[:pos] + [own] + others[pos:], pos, lang))
    for j in range(0, len(matches) - 1, 2):  # singletons two per slot, as E45
        pair = f"goal_match-train-e49-{j // 2}"
        for k, (img, own, goals, pos, lang) in enumerate(matches[j:j + 2]):
            emit(pair, k, "goal_match", lang, img, MATCH_STATE[lang], MATCH_Q[lang],
                 [(f"g{n}", x) for n, x in enumerate(goals)], f"g{pos}", own_goal=own)
    return rows


def main() -> int:
    assert sha256_file(E47_ROWS) == E47_SHA
    e47 = [json.loads(l) for l in E47_ROWS.read_text(encoding="utf-8").splitlines() if l.strip()]
    e41_goals = [r["own_goal"].strip() for r in e47 if r["bind"] == "goal" and r["split"] == "train"]
    got, stats = fetch_episodes()
    m3f_eps, train_eps = got[:N_M3F], got[N_M3F:]

    m3f = build_m3f(m3f_eps)
    new = build_bind(train_eps, e41_goals)
    rows = e47 + new

    # checks of the criteria
    assert len(m3f) == 2 * N_M3F and all(r["own_goal"].lower() != r["goal"].lower() for r in m3f if r["m3"] == "swap")
    assert {r["group_id"] for r in m3f}.isdisjoint({g["eid"] for g in train_eps})
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)
    by_pair: dict[str, list[dict]] = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    assert all(len(v) == 2 for v in by_pair.values())
    for r in new:
        assert (OUT / r["image"]).is_file() and r["target_option"] in {o["id"] for o in r["options"]}
    for r in m3f:
        assert (M3F / r["image"]).is_file()

    OUT.mkdir(parents=True, exist_ok=True)
    M3F.mkdir(parents=True, exist_ok=True)
    (OUT / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (M3F / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in m3f), encoding="utf-8")
    kinds = sorted({r["bind"] for r in rows})
    manifest = {
        "criteria": "docs/E49_CRITERIA_2026-10-26.md", "source": {"repo": GUI_REPO, **stats},
        "license": "CC BY 4.0 (GUI-Odyssey - Lu et al. 2024, OpenGVLab); E47 rows as their manifest",
        "e47_rows_sha256": E47_SHA, "n_e47_rows": len(e47), "n_new_rows": len(new),
        "n_rows": {sp: {k: sum(r["split"] == sp and r["bind"] == k for r in rows) for k in kinds} for sp in ("train", "dev")},
        "rows_sha256": sha256_file(OUT / "rows.jsonl"), "m3f_rows_sha256": sha256_file(M3F / "rows.jsonl"),
        "m3f_episodes": [g["eid"] for g in m3f_eps], "train_episodes": [g["eid"] for g in train_eps],
        "files_sha256": {p.relative_to(OUT.parent).as_posix(): sha256_file(p)
                         for root in (OUT, M3F) for p in sorted((root / "shots").glob("*"))},
    }
    (OUT.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_e47_rows", "n_new_rows", "n_rows", "rows_sha256", "m3f_rows_sha256")}
                     | {"source": stats}, indent=1))
    for r in (new[0], new[1], next(x for x in new if x["bind"] == "goal_match"), m3f[0], m3f[1]):
        print(" ", r["id"], r["target_option"], "|", r["state"][:90], "|", [o["description"][:25] for o in r["options"]][:4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
