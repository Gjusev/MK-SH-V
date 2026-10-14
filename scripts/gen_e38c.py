"""E38c generator: real-pixel evaluation floor (criteria: docs/E38C_CRITERIA_2026-10-14.md).

    python scripts/gen_e38c.py    # -> data/e38c/{cord,gui_odyssey}, manifest with SHA-256

CORD validation+test -> field_threshold (threshold drawn around the real total,
50/50 by construction). GUI-Odyssey task_split/test COMPLETE episodes ->
task_done pairs (final=true, sha256 first-half step=false), group_id=episode.
Evaluation-only: every contract row has split="test".
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mokka_decisions.contracts import DecisionCase, Option  # noqa: E402
from mksh_v.gen_e38a import QUESTION as DOC_QUESTION, STATE as DOC_STATE  # noqa: E402
from mksh_v.gen_e38b import QUESTION as SCR_QUESTION, STATE as SCR_STATE  # noqa: E402
from probe_e38c import cord_total, iter_strings, parse_id_price  # noqa: E402

DATA = ROOT / "data/e38c"
GUI_REPO = "OpenGVLab/GUI-Odyssey"
GUI_RESOLVE = f"https://huggingface.co/datasets/{GUI_REPO}/resolve/main/"
N_EPISODES = 500
LANGS = ("en", "es", "de")


def parse_steps(value):
    import ast

    for parse in (json.loads, ast.literal_eval):
        try:
            steps = parse(value or "[]")
            if isinstance(steps, list):
                return steps
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return []


def fmt_idr(n: float) -> str:
    """Dot-thousands, no decimals (the receipts' printed convention)."""
    return f"{round(n):,.0f}".replace(",", ".")


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)

# ------------------------------------------------------------- CORD side ----
def gen_cord(dest: Path) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    dest.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n_seen = n_clean = 0
    for split in ("validation", "test"):
        ds = load_dataset("naver-clova-ix/cord-v2", split=split, streaming=True)
        for i, row in enumerate(ds):
            n_seen += 1
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
            if not total or total <= 0:
                continue
            if txt.strip() not in set(iter_strings(gt)):
                continue
            n_clean += 1
            rid = row.get("image", {}).get("path") or f"{split}-{i}"
            rng = random.Random(stable_seed("e38c-cord", rid))
            delta = math.exp(rng.uniform(math.log(0.001), math.log(0.5)))
            thr = round(total * (1 - delta) if rng.random() < 0.5 else total * (1 + delta))
            label = "true" if total >= thr else "false"
            img_path = dest / f"{split}-{i}.png"
            if not img_path.exists():
                img = row["image"]
                img_path.write_bytes(img["bytes"] if isinstance(img, dict) and "bytes" in img
                                     else Path(img["path"]).read_bytes())
            for lang in LANGS:
                q = DOC_QUESTION["field_threshold"][lang].replace("EUR", "IDR").format(thr=fmt_idr(thr))
                rows.append(DecisionCase(
                    id=f"e38c-cord-{split}-{i}-{lang}", group_id=f"e38c-cord-{split}-{i}",
                    source="e38c-cord", language=lang, domain="documents",
                    state=DOC_STATE[lang], question=q,
                    options=_ft_options(lang), target_kind="single", target_option=label,
                    label_origin="e38c-cord-gt", split="test").to_dict()
                    | {"family": "field_threshold", "image": f"{img_path.name}",
                       "gt_total_key": k, "gt_total_text": txt, "gt_total_value": total,
                       "threshold": thr, "cord_split": split, "row_in_split": i})
    return rows, {"seen": n_seen, "clean": n_clean}

def _ft_options(lang: str) -> tuple[Option, ...]:
    from mksh_v.gen_e38b import OPTION_TEXT as SCR_OPTION
    from mksh_v.gen_e38a import OPTION_TEXT as DOC_OPTION

    return tuple(Option(id=i, description=d) for i, d in
                 zip(("false", "true"), DOC_OPTION["field_threshold"][lang]))


def _td_options(lang: str) -> tuple[Option, ...]:
    from mksh_v.gen_e38b import OPTION_TEXT

    return tuple(Option(id=i, description=d) for i, d in
                 zip(("false", "true"), OPTION_TEXT["task_done"][lang]))

# ------------------------------------------------------- GUI-Odyssey side ----
def gen_gui(dest: Path) -> tuple[list[dict], dict]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "task_split.json").write_bytes(requests.get(GUI_RESOLVE + "splits/task_split.json", timeout=60).content)
    test_ids = set(json.loads((dest / "task_split.json").read_text(encoding="utf-8"))["test"])
    print(f"task_split test episodes: {len(test_ids)}", flush=True)
    rows: list[dict] = []
    seen = set()
    for r in load_dataset(GUI_REPO, split="all", streaming=True):
        row = r["row"] if "row" in r else r
        eid = str(row.get("episode_id"))
        if eid not in test_ids or eid in seen:
            continue
        seen.add(eid)
        steps = parse_steps(row.get("steps"))
        if not steps or str(steps[-1].get("action") or "").upper() != "COMPLETE":
            continue
        seen.add("__ok__" + eid)
        rows.append({"row_idx": eid, "row": row, "steps": steps})
        if len([x for x in rows]) >= N_EPISODES * 3:
            break
    keyed = sorted(rows, key=lambda x: hashlib.sha256(
        json.dumps({k: v for k, v in x["row"].items() if k != "steps"}, sort_keys=True).encode("utf-8")).hexdigest())
    sample = keyed[:N_EPISODES]
    print(f"COMPLETE test episodes found: {len(rows)}; sampling {len(sample)}", flush=True)
    files = set(f.path for f in HfApi().list_repo_tree(GUI_REPO, repo_type="dataset", recursive=True))
    out, dl = [], 0
    for x in sample:
        row, steps, eid = x["row"], x["steps"], str(x["row"].get("episode_id"))
        n = len(steps)
        if n < 2:
            continue
        k = int(hashlib.sha256(eid.encode()).hexdigest()[:15], 16) % (n // 2)
        neg_ref, fin_ref = str(steps[k].get("screenshot") or ""), str(steps[-1].get("screenshot") or "")
        neg_paths = [p for p in files if p.endswith(neg_ref)]
        fin_paths = [p for p in files if p.endswith(fin_ref)]
        if not neg_paths or not fin_paths or neg_paths[0] == fin_paths[0]:
            continue
        neg_local, fin_local = f"shots/{eid}_neg.png", f"shots/{eid}_fin.png"
        for rel, remote in ((neg_local, neg_paths[0]), (fin_local, fin_paths[0])):
            p = dest / rel
            if not p.exists():
                resp = requests.get(GUI_RESOLVE + remote, timeout=120)
                if resp.status_code != 200:
                    break
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(resp.content)
                dl += 1
        if not (dest / neg_local).exists() or not (dest / fin_local).exists():
            continue
        for lang in LANGS:
            for polarity, rel, label in (("neg", neg_local, "false"), ("fin", fin_local, "true")):
                out.append(DecisionCase(
                    id=f"e38c-gui-{eid}-{polarity}-{lang}", group_id=eid,
                    source="e38c-gui", language=lang, domain="screens",
                    state=SCR_STATE["task_done"][lang].format(goal=row.get("task", "")),
                    question=SCR_QUESTION["task_done"][lang],
                    options=_td_options(lang), target_kind="single", target_option=label,
                    label_origin="e38c-gui-steps", split="test").to_dict()
                    | {"family": "task_done", "image": rel, "polarity": polarity,
                       "k": k, "n_steps": n, "steps": steps, "final_action": "COMPLETE",
                       "goal": row.get("task", "")})
    return out, {"test_ids": len(test_ids), "episodes_used": len(sample), "screenshots_downloaded": dl}

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    print("[cord]", flush=True)
    cord_rows, cord_stats = gen_cord(DATA / "cord")
    print(f"  receipts clean: {cord_stats['clean']}/{cord_stats['seen']}", flush=True)
    print("[gui_odyssey]", flush=True)
    gui_rows, gui_stats = gen_gui(DATA / "gui_odyssey")
    print(f"  episodes used: {gui_stats['episodes_used']}, screenshots: {gui_stats['screenshots_downloaded']}",
          flush=True)
    for name, rows in (("cord", cord_rows), ("gui_odyssey", gui_rows)):
        p = DATA / name / "rows.jsonl"
        p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    files = {}
    for sub in ("cord", "gui_odyssey"):
        for p in (DATA / sub).rglob("*"):
            if p.is_file() and p.name != "manifest.json":
                files[str(p.relative_to(DATA))] = sha256_file(p)
    manifest = {"criteria": "docs/E38C_CRITERIA_2026-10-14.md", "stats": {"cord": cord_stats, "gui_odyssey": gui_stats},
                "n_rows": {"cord": len(cord_rows), "gui_odyssey": len(gui_rows)},
                "files_sha256": files}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"rows: cord={len(cord_rows)} gui={len(gui_rows)}; files hashed: {len(files)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


