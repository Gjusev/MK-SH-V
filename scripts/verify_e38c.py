"""Independent verifier for E38c (criteria: docs/E38C_CRITERIA_2026-10-14.md).

Does NOT import the generator. Re-derives every label from the original ground
truth carried in the row (CORD gt total snippet; GUI-Odyssey steps verbatim),
checks hashes, contract, group disjunction and the adapted G2 text-only leak
test (NB trained on the synthetic families, evaluated on real rows, positive
control). Writes runs/e38c-verify/verification.json.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from mokka_decisions.contracts import DecisionCase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DATA = ROOT / "data/e38c"
OUT = ROOT / "runs/e38c-verify"
NUM_RE = re.compile(r"\d[\d.,]*\d|\d")

def parse_price(s: str):
    s = re.sub(r"[^\d.,]", "", s.strip())
    if not s or not re.fullmatch(r"[\d.,]+", s):
        return None
    if "." in s and "," in s:
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", s):
            return float(s.replace(",", ""))
        if re.fullmatch(r"\d{1,3}(\.\d{3})+,\d+", s):
            return float(s.replace(".", "").replace(",", "."))
        return None
    n = s.count(".") + s.count(",")
    if n >= 2:
        return float(re.sub(r"[.,]", "", s))
    m = re.fullmatch(r"(\d+)([.,])(\d{3})", s)
    if m:
        return float(m.group(1) + m.group(3))
    return float(s.replace(",", "."))


def tokens(row, bins) -> list:
    text = " ".join([row["state"], row["question"]] + [o["description"] for o in row["options"]])
    toks = re.findall(r"\w+", text.lower())
    m = NUM_RE.findall(row["question"])
    if m and bins:
        v = parse_price(m[0])
        if v is not None:
            toks.append(f"THRBIN_{sum(v > b for b in bins)}")
    return toks


def naive_bayes(train, evalset, bins) -> dict:
    cls_n, tok_n, vocab = Counter(), defaultdict(Counter), set()
    for r in train:
        y = r["target_option"]
        cls_n[y] += 1
        for t in tokens(r, bins):
            tok_n[y][t] += 1
            vocab.add(t)
    tot = {y: sum(c.values()) for y, c in tok_n.items()}
    correct = 0
    for r in evalset:
        ts = tokens(r, bins)
        best = max(cls_n, key=lambda y: math.log(cls_n[y]) + sum(
            math.log((tok_n[y][t] + 1) / (tot[y] + len(vocab))) for t in ts))
        correct += best == r["target_option"]
    maj = max(Counter(r["target_option"] for r in evalset).values()) / len(evalset)
    acc = correct / len(evalset)
    return {"accuracy": round(acc, 4), "majority": round(maj, 4), "bar": round(maj + 0.05, 4),
            "n_eval": len(evalset), "passed": acc <= maj + 0.05}


def derive_cord(row: dict) -> str:
    """Label from the verbatim gt snippet and the row threshold."""
    total = parse_price(row["gt_total_text"])
    assert total is not None and total > 0, f"{row['id']}: gt total not parseable"
    assert abs(total - row["gt_total_value"]) < 1e-6, f"{row['id']}: gt total mismatch"
    thr = float(row["threshold"])
    assert 0 < (thr - total) / total <= 0.5 or 0 < (total - thr) / total <= 0.5, \
        f"{row['id']}: threshold outside delta range"
    return "true" if total >= thr else "false"


def derive_gui(row: dict) -> str:
    """Label from the verbatim steps annotation + stored refs cross-check."""
    steps = row["steps"]
    n = len(steps)
    assert n >= 2 and str(steps[-1].get("action") or "").upper() == "COMPLETE", \
        f"{row['id']}: episode not terminal-COMPLETE"
    eid = str(row["group_id"])
    k = int(hashlib.sha256(eid.encode()).hexdigest()[:15], 16) % (n // 2)
    assert 0 <= k < n // 2, f"{row['id']}: k out of range"
    ref_k = str(steps[k].get("screenshot") or "")
    ref_last = str(steps[-1].get("screenshot") or "")
    assert ref_k != ref_last, f"{row['id']}: negative ref equals final ref"
    assert row.get("neg_ref") == ref_k, f"{row['id']}: stored neg_ref != steps[k]"
    assert row.get("fin_ref") == ref_last, f"{row['id']}: stored fin_ref != steps[-1]"
    assert row["image"] == f"shots/{eid}_{'neg' if row['polarity'] == 'neg' else 'fin'}.png", \
        f"{row['id']}: image path convention violated"
    return "true" if row["polarity"] == "fin" else "false"


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    report = {"criteria": "docs/E38C_CRITERIA_2026-10-14.md", "sources": {}}
    ok_all = True
    syn_train = {
        "cord": load_jsonl(ROOT / "data/e38a/field_threshold/rows.jsonl"),
        "gui": load_jsonl(ROOT / "data/e38b/task_done/rows.jsonl"),
    }
    for name, family in (("cord", "field_threshold"), ("gui_odyssey", "task_done")):
        rows = load_jsonl(DATA / name / "rows.jsonl")
        fails = defaultdict(list)
        groups = defaultdict(set)
        seen_ids = set()
        for row in rows:
            if row["id"] in seen_ids:
                fails["dup_id"].append(row["id"])
            seen_ids.add(row["id"])
            try:
                DecisionCase.from_dict(row)
            except Exception as exc:
                fails["e_contract"].append(f"{row['id']}: {exc}")
            sha = hashlib.sha256((DATA / name / row["image"]).read_bytes()).hexdigest()
            if sha != manifest["files_sha256"].get(f"{name}/{row['image']}".replace("\\\\", "/")):
                fails["e_png_sha"].append(row["id"])
            try:
                derived = derive_cord(row) if name == "cord" else derive_gui(row)
            except AssertionError as exc:
                fails["a_label"].append(str(exc))
                continue
            if derived != row["target_option"]:
                fails["a_label"].append(f"{row['id']}: row {row['target_option']} derived {derived}")
            groups[row["group_id"]].add(row["split"])
        for g, parts in groups.items():
            if len(parts) != 1:
                fails["b_group_cross"].append(g)
        # G2 adapted: NB trained on the synthetic family, evaluated on real rows
        train = syn_train["cord" if name == "cord" else "gui"]
        evalset = rows
        bins = []
        if name == "cord":
            thr_vals = sorted(t for r in train if (m := NUM_RE.search(r["question"]))
                              and (t := parse_price(m.group(0))) is not None)
            bins = [thr_vals[int(len(thr_vals) * q / 10)] for q in range(1, 10)]
        g2 = naive_bayes(train, evalset, bins)
        inject = lambda rs: [dict(r, state=f"{r['state']} leak{r['target_option']}") for r in rs]  # noqa: E731
        control = naive_bayes(inject(train), inject(evalset), bins)
        g2["positive_control_accuracy"] = control["accuracy"]
        g2["passed"] = g2["passed"] and control["accuracy"] > g2["bar"]
        src_ok = not fails and g2["passed"]
        ok_all &= src_ok
        report["sources"][name] = {
            "n_rows": len(rows), "label_failures": len(fails.get("a_label", [])),
            "G1": {"passed": not fails, "failures": {k: (len(v), v[:5]) for k, v in fails.items()}},
            "G2": g2,
        }
        print(name, "G1", "PASS" if not fails else f"FAIL {dict((k, len(v)) for k, v in fails.items())}",
              "| G2", g2, flush=True)
    report["G1_G2_passed"] = ok_all
    (OUT / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("G1+G2", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
