"""E38c mappability probe (criteria: docs/E38C_PROBE_CRITERIA_2026-10-14.md).

    python scripts/probe_e38c.py          # probe all three sources -> data/e38c-probe, runs/e38c-probe/probe.json

Fetches the first min(2000, split_size) rows per source via the datasets-server
/rows API, selects 100 by ascending sha256 of the server row JSON (frozen rule),
derives family labels by the frozen mapping table, and reports clean shares and
rejection reasons. No full downloads: at most 100 individual images per source.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/e38c-probe"
RUNS = ROOT / "runs/e38c-probe"
API = "https://datasets-server.huggingface.co"
SOURCES = {
    "cord": ("naver-clova-ix/cord-v2", "default", "test"),
    "doclaynet": ("pierreguillou/DocLayNet-small", "DocLayNet_2022.08_processed_on_2023.01", "test"),
    "gui_odyssey": ("OpenGVLab/GUI-Odyssey", "default", "all"),
}
N_SAMPLE = 100
N_FETCH_CAP = 2000
PAGE = 100
PASS_SHARE = 0.70

def http_json(url: str, params: dict | None = None, tries: int = 4) -> dict:
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503):
                time.sleep(2 ** i)
                continue
            raise RuntimeError(f"HTTP {r.status_code} for {url}: {r.text[:200]}")
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError(f"unreachable: {url}")


def fetch_rows(repo: str, config: str, split: str) -> list[dict]:
    """First min(N_FETCH_CAP, split_size) rows in server order via /rows."""
    out, offset = [], 0
    while offset < N_FETCH_CAP:
        page = http_json(f"{API}/rows", {"dataset": repo, "config": config, "split": split,
                                         "offset": offset, "length": PAGE})
        rows = page.get("rows", [])
        if not rows:
            break
        out.extend(rows)
        if len(rows) < PAGE:
            break
        offset += PAGE
        if offset % 500 == 0:
            print(f"  fetched {offset} rows", flush=True)
    return out


def select_sample(rows: list[dict]) -> list[dict]:
    """Frozen rule: first 100 rows by ascending sha256 of the server row JSON."""
    keyed = sorted(rows, key=lambda r: hashlib.sha256(json.dumps(r, sort_keys=False).encode("utf-8")).hexdigest())
    return keyed[:N_SAMPLE]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---- CORD -> field_threshold (+ fixed doc_type=receipt) ----------------------
def iter_strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from iter_strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from iter_strings(v)


_PRICE_RE = re.compile(r"[^\d.,]")


def parse_id_price(s: str):
    """ID/EN price string -> float, frozen rule: >=2 separators = thousands;
    one separator + exactly 3 digits = thousands (IDR has no cents);
    one separator + 1-2 digits = decimal. Ambiguous -> None."""
    s = _PRICE_RE.sub("", s.strip())
    if not s or not re.fullmatch(r"[\d.,]+", s):
        return None
    if "." in s and "," in s:
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", s):
            return float(s.replace(",", ""))
        if re.fullmatch(r"\d{1,3}(\.\d{3})+,\d+", s):
            return float(s.replace(".", "").replace(",", "."))
        return None
    n_sep = s.count(".") + s.count(",")
    if n_sep >= 2:
        return float(re.sub(r"[.,]", "", s))
    m = re.fullmatch(r"(\d+)([.,])(\d{3})", s)
    if m:
        return float(m.group(1) + m.group(3))  # thousands (ID)
    return float(s.replace(",", "."))  # decimal

def cord_total(ground_truths: dict):
    """Exactly one total price from gt_parse.total.*_price, else None."""
    total = (ground_truths.get("gt_parse") or {}).get("total") or {}
    prices = {k: v for k, v in total.items() if k.endswith("_price") and isinstance(v, str) and v.strip()}
    if len(prices) != 1:
        return None
    k, txt = next(iter(prices.items()))
    return k, txt

def probe_cord(rows: list[dict], dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    clean, reasons, out_rows = 0, Counter(), []
    for r in select_sample(rows):
        row = r["row"]
        gt_txt = row.get("ground_truths") or ""
        try:
            gt = json.loads(gt_txt) if isinstance(gt_txt, str) else gt_txt
        except json.JSONDecodeError:
            reasons["gt_not_json"] += 1
            out_rows.append({"row_idx": r["row_idx"], "clean": False, "reason": "gt_not_json"})
            continue
        tot = cord_total(gt)
        if tot is None:
            reasons["no_single_total"] += 1
            out_rows.append({"row_idx": r["row_idx"], "clean": False, "reason": "no_single_total"})
            continue
        k, txt = tot
        val = parse_id_price(txt)
        if val is None or val <= 0:
            reasons["price_not_parseable"] += 1
            out_rows.append({"row_idx": r["row_idx"], "clean": False, "reason": "price_not_parseable"})
            continue
        if txt.strip() not in set(iter_strings(gt)):
            reasons["total_not_visible"] += 1
            out_rows.append({"row_idx": r["row_idx"], "clean": False, "reason": "total_not_visible"})
            continue
        clean += 1
        out_rows.append({"row_idx": r["row_idx"], "clean": True, "family": "field_threshold",
                         "doc_type_fixed": "receipt", "total_key": k, "total_text": txt, "total_value": val})
    return {"clean": clean, "n": len(out_rows), "reasons": dict(reasons), "rows": out_rows}

# ---- DocLayNet -> doc_type ---------------------------------------------------
DOC_CLASSES = ("invoice", "receipt", "contract", "form", "letter", "other")


def probe_doclaynet(rows: list[dict], dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    clean, reasons, out_rows = 0, Counter(), []
    for r in select_sample(rows):
        row = r["row"]
        cat = row.get("doc_category")
        if cat is None:
            reasons["missing_doc_category"] += 1
            out_rows.append({"row_idx": r["row_idx"], "clean": False,
                             "reason": "missing_doc_category (mirror has layout labels only)"})
            continue
        if cat not in DOC_CLASSES:
            reasons["no_doc_type_match"] += 1
            out_rows.append({"row_idx": r["row_idx"], "clean": False,
                             "reason": f"no_doc_type_match: {cat}"})
            continue
        clean += 1
        out_rows.append({"row_idx": r["row_idx"], "clean": True, "family": "doc_type", "doc_category": cat})
    return {"clean": clean, "n": len(out_rows), "reasons": dict(reasons), "rows": out_rows}

# ---- GUI-Odyssey -> task_done ------------------------------------------------
def probe_gui_odyssey(rows: list[dict], dest: Path, repo: str) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    files = set()
    for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True):
        files.add(f.path)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "repo_files.txt").write_text("\n".join(sorted(files)), encoding="utf-8")
    clean, reasons, out_rows = 0, Counter(), []
    for r in select_sample(rows):
        row = r["row"]
        try:
            steps = json.loads(row.get("steps") or "[]")
        except json.JSONDecodeError:
            steps = []
        if not steps:
            reasons["no_steps"] += 1
            out_rows.append({"row_idx": r["row_idx"], "clean": False, "reason": "no_steps"})
            continue
        last = steps[-1]
        action = str(last.get("action") or "").upper()
        if action == "COMPLETE":
            label = "true"
        elif action == "IMPOSSIBLE":
            label = "false"
        else:
            reasons["no_terminal_action"] += 1
            out_rows.append({"row_idx": r["row_idx"], "clean": False, "reason": "no_terminal_action"})
            continue
        shot = str(last.get("screenshot") or "").strip()
        if not shot:
            reasons["no_screenshot_ref"] += 1
            out_rows.append({"row_idx": r["row_idx"], "clean": False, "reason": "no_screenshot_ref"})
            continue
        matches = [p for p in files if p.endswith(shot)]
        if not matches:
            reasons["screenshot_missing_in_repo"] += 1
            out_rows.append({"row_idx": r["row_idx"], "clean": False, "reason": "screenshot_missing_in_repo"})
            continue
        clean += 1
        out_rows.append({"row_idx": r["row_idx"], "clean": True, "family": "task_done",
                         "label": label, "screenshot_path": matches[0]})
    return {"clean": clean, "n": len(out_rows), "reasons": dict(reasons), "rows": out_rows}

# ---- main --------------------------------------------------------------------
def main() -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    report = {"criteria": "docs/E38C_PROBE_CRITERIA_2026-10-14.md", "sources": {}}
    for name, (repo, config, split) in SOURCES.items():
        print(f"[{name}] {repo} {config}/{split}", flush=True)
        try:
            rows = fetch_rows(repo, config, split)
        except Exception as exc:
            report["sources"][name] = {"error": f"fetch: {type(exc).__name__}: {str(exc)[:300]}"}
            print(f"  FETCH ERROR: {exc}", flush=True)
            continue
        dest = DATA / name
        if name == "cord":
            res = probe_cord(rows, dest)
        elif name == "doclaynet":
            res = probe_doclaynet(rows, dest)
        else:
            res = probe_gui_odyssey(rows, dest, repo)
        rows_path = dest / "rows.jsonl"
        rows_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in res["rows"]),
                             encoding="utf-8")
        manifest = {"repo": repo, "config": config, "split": split,
                    "rows_fetched": len(rows), "sampled": res["n"],
                    "files_sha256": {p.name: sha256_file(p) for p in sorted(dest.glob("*")) if p.is_file()}}
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        share = res["clean"] / max(res["n"], 1)
        report["sources"][name] = {
            "repo": repo, "config": config, "split": split, "rows_fetched": len(rows),
            "sampled": res["n"], "clean": res["clean"], "clean_share": round(share, 4),
            "passed": share >= PASS_SHARE, "reasons": res["reasons"],
        }
        print(f"  clean {res['clean']}/{res['n']} = {share:.0%} -> {'PASS' if share >= PASS_SHARE else 'FAIL'}",
              flush=True)
    (RUNS / "probe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, s in report["sources"].items():
        if "error" in s:
            print(f"{name}: ERROR {s['error'][:80]}")
        else:
            print(f"{name}: {s['clean']}/{s['sampled']} clean ({s['clean_share']:.0%}) -> "
                  f"{'PASS' if s['passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


