"""E41 real TRAINING data (criteria: docs/E41_CRITERIA_2026-10-22.md).

    python scripts/gen_e41.py    # -> data/e41/{cord,gui_odyssey}, manifest with SHA-256

CORD v2 train -> field_threshold (two thresholds per receipt, one per side);
GUI-Odyssey task_split TRAIN COMPLETE episodes -> task_done pairs. Same
templates, formats and pairing rules as E38c (its helpers are reused), but
sources disjoint from the E38c evaluation (asserted). 10 % of groups -> dev.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_e38c import (DOC_QUESTION, DOC_STATE, GUI_REPO, GUI_RESOLVE, LANGS, SCR_QUESTION,  # noqa: E402
                      SCR_STATE, _ft_options, _td_options, fmt_idr, parse_steps, sha256_file,
                      stable_seed)
from mokka_decisions.contracts import DecisionCase  # noqa: E402
from probe_e38c import cord_total, iter_strings, parse_id_price  # noqa: E402

DATA = ROOT / "data/e41"
E38C = ROOT / "data/e38c"
N_GUI = 1200


def split_of(group_id: str) -> str:
    return "dev" if int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"


def case_row(**kw) -> dict:
    extra = kw.pop("extra")
    return DecisionCase(target_kind="single", **kw).to_dict() | extra


def gen_cord(dest: Path) -> tuple[list[dict], dict]:
    from datasets import Image as HFImage, load_dataset
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("naver-clova-ix/cord-v2", split="train", streaming=True)
    ds = ds.cast_column("image", HFImage(decode=False))  # original bytes, no re-encode
    rows, seen, clean = [], 0, 0
    for i, row in enumerate(ds):
        seen += 1
        gt_txt = row.get("ground_truth") or ""
        try:
            gt = json.loads(gt_txt) if isinstance(gt_txt, str) else gt_txt
        except json.JSONDecodeError:
            continue
        tot = cord_total(gt)
        if tot is None:
            continue
        k, txt = tot
        total = parse_id_price(txt)
        if not total or total <= 0 or txt.strip() not in set(iter_strings(gt)):
            continue
        clean += 1
        raw = row["image"]["bytes"]
        ext = Image.open(io.BytesIO(raw)).format.lower().replace("jpeg", "jpg")
        name = f"train-{i}.{ext}"
        if not (dest / name).exists():
            (dest / name).write_bytes(raw)
        gid = f"e41-cord-train-{i}"
        rng = random.Random(stable_seed("e41-cord", gid))
        for side in ("below", "above"):
            delta = math.exp(rng.uniform(math.log(0.001), math.log(0.5)))
            thr = round(total * (1 - delta) if side == "below" else total * (1 + delta))
            label = "true" if total >= thr else "false"
            for lang in LANGS:
                q = DOC_QUESTION["field_threshold"][lang].replace("EUR", "IDR").format(thr=fmt_idr(thr))
                rows.append(case_row(
                    id=f"{gid}-{side}-{lang}", group_id=gid, source="e41-cord", language=lang,
                    domain="documents", state=DOC_STATE[lang], question=q,
                    options=_ft_options(lang), target_option=label,
                    label_origin="e41-cord-gt", split=split_of(gid),
                    extra={"family": "field_threshold", "image": f"cord/{name}", "gt_total_key": k,
                           "gt_total_text": txt, "gt_total_value": total, "threshold": thr,
                           "side": side, "cord_split": "train", "row_in_split": i}))
    return rows, {"seen": seen, "clean": clean}


def _get(url: str, token: str | None, tries: int = 5) -> bytes | None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for t in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=120)
            if r.status_code == 200:
                return r.content
            if r.status_code in (429, 500, 502, 503):
                time.sleep(5 * (t + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(5 * (t + 1))
    return None


def gen_gui(dest: Path) -> tuple[list[dict], dict]:
    from datasets import load_dataset
    from huggingface_hub import HfApi, get_token

    dest.mkdir(parents=True, exist_ok=True)
    split = json.loads((E38C / "gui_odyssey/task_split.json").read_text(encoding="utf-8"))
    train_ids = {x.removesuffix(".json") for x in split["train"]}
    test_ids = {x.removesuffix(".json") for x in split["test"]}
    e38c_ids = {json.loads(l)["group_id"] for l in
                (E38C / "gui_odyssey/rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    assert not train_ids & test_ids and not train_ids & e38c_ids
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
            rel = f"gui_odyssey/shots/{eid}_{pol}{Path(remote).suffix}"
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
        for start in range(0, len(cands), 200):
            got += [g for g in pool.map(fetch, cands[start:start + 200]) if g]
            print(f"  gui: {len(got)} episodes ready ({start + 200}/{len(cands)} tried)", flush=True)
            if len(got) >= N_GUI:
                break
    got = got[:N_GUI]
    rows = []
    for g in got:
        goal = g["row"].get("task", "")
        for lang in LANGS:
            for pol, label in (("neg", "false"), ("fin", "true")):
                rows.append(case_row(
                    id=f"e41-gui-{g['eid']}-{pol}-{lang}", group_id=g["eid"], source="e41-gui",
                    language=lang, domain="screens",
                    state=SCR_STATE["task_done"][lang].format(goal=goal),
                    question=SCR_QUESTION["task_done"][lang], options=_td_options(lang),
                    target_option=label, label_origin="e41-gui-steps-unverified",
                    split=split_of(g["eid"]),
                    extra={"family": "task_done", "image": g[pol], "polarity": pol, "k": g["k"],
                           "n_steps": g["n"], "neg_ref": g["neg_ref"], "fin_ref": g["fin_ref"],
                           "goal": goal}))
    return rows, {"train_ids": len(train_ids), "complete_candidates": len(cands), "episodes_used": len(got)}


def check(rows: list[dict]) -> None:
    """Generator asserts of the criteria: labels from ground truth, pure groups."""
    split_by_group: dict[str, set] = {}
    for r in rows:
        split_by_group.setdefault(r["group_id"], set()).add(r["split"])
        if r["family"] == "field_threshold":
            assert r["target_option"] == ("true" if r["gt_total_value"] >= r["threshold"] else "false")
            assert r["cord_split"] == "train"
        else:
            assert r["target_option"] == ("true" if r["polarity"] == "fin" else "false")
            assert r["k"] < r["n_steps"] // 2
        assert (DATA / r["image"]).is_file()
    assert all(len(s) == 1 for s in split_by_group.values())


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    print("[cord train]", flush=True)
    cord_rows, cord_stats = gen_cord(DATA / "cord")
    print(f"  receipts clean {cord_stats['clean']}/{cord_stats['seen']}, rows {len(cord_rows)}", flush=True)
    print("[gui_odyssey task_split train]", flush=True)
    gui_rows, gui_stats = gen_gui(DATA / "gui_odyssey")
    print(f"  {gui_stats}, rows {len(gui_rows)}", flush=True)
    check(cord_rows + gui_rows)
    for name, rows in (("cord", cord_rows), ("gui_odyssey", gui_rows)):
        (DATA / name / "rows.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    write_manifest(cord_rows, gui_rows, cord_stats, gui_stats)
    return 0


def write_manifest(cord_rows, gui_rows, cord_stats, gui_stats) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    manifest = {
        "criteria": "docs/E41_CRITERIA_2026-10-22.md",
        "sources": {"cord": {"repo": "naver-clova-ix/cord-v2", "split": "train",
                             "revision": api.dataset_info("naver-clova-ix/cord-v2").sha},
                    "gui_odyssey": {"repo": GUI_REPO, "split": "task_split train",
                                    "revision": api.dataset_info(GUI_REPO).sha}},
        "license": "CC BY 4.0 (both sources; attribution: CORD - Park et al. 2019, NAVER Clova; "
                   "GUI-Odyssey - Lu et al. 2024, OpenGVLab)",
        "stats": {"cord": cord_stats, "gui_odyssey": gui_stats},
        "n_rows": {name: {sp: sum(r["split"] == sp for r in rows) for sp in ("train", "dev")}
                   for name, rows in (("cord", cord_rows), ("gui_odyssey", gui_rows))},
        "rows_sha256": {name: sha256_file(DATA / name / "rows.jsonl") for name in ("cord", "gui_odyssey")},
        "files_sha256": {p.relative_to(DATA).as_posix(): sha256_file(p)
                         for p in sorted(DATA.rglob("*")) if p.is_file() and p.name != "manifest.json"
                         and p.name != "dataset-metadata.json"},
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_rows", "rows_sha256", "sources")}, indent=1), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
