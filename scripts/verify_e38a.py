"""Independent verifier for E38a (criteria: docs/E38A_CRITERIA_2026-10-14.md).

Does NOT import the generator. Re-derives every label from the rendered HTML
sidecars with its own parser and rules, and checks splits, counterfactuals,
counts, hashes, contract validity and text leakage (G1), then runs the
text-only naive Bayes leak test (G2). Writes runs/e38a-verify/verification.json.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path

from mokka_decisions.contracts import DecisionCase

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/e38a"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runs/e38a-verify"
FAMILIES = ("doc_type", "field_threshold", "form_complete")
COUNTS = {"train": 1000, "dev": 100, "cal_temperature": 150, "cal_policy": 150, "test": 200}
LANGS = {"train": ("en", "es", "de"), "dev": ("en", "es", "de"), "cal_temperature": ("en", "es", "de"),
         "cal_policy": ("en", "es", "de"), "test": ("en", "es", "de", "fr")}
HELDOUT = ("dev", "cal_temperature", "cal_policy")
N_CLASSES = {"doc_type": 6, "field_threshold": 2, "form_complete": 3}
NUM_RE = re.compile(r"\d[\d.,]*\d|\d")


class Elements(HTMLParser):
    """Collect every element carrying data-k: its attrs, full text, and parent field index."""

    def __init__(self):
        super().__init__()
        self.stack, self.els, self.template = [], [], None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "data-template" in a:
            self.template = a["data-template"]
        rec = None
        if "data-k" in a:
            field = next((r["i"] for r in reversed([s for s in self.stack if s]) if r["k"] == "field"), None)
            rec = {"k": a["data-k"], "attrs": a, "text": [], "field": field, "i": len(self.els)}
            self.els.append(rec)
        self.stack.append(rec)

    def handle_endtag(self, tag):
        if self.stack:
            self.stack.pop()

    def handle_data(self, data):
        for r in self.stack:
            if r:
                r["text"].append(data)


def parse(html_text: str):
    p = Elements()
    p.feed(html_text)
    for r in p.els:
        r["text"] = "".join(r["text"]).strip()
    return p.els, p.template


def num(s: str, lang: str) -> float:
    s = s.strip()
    s = s.replace(",", "") if lang == "en" else s.replace(".", "").replace(",", ".")
    return float(s)


def derive(fam: str, els, row) -> str:
    ks = Counter(e["k"] for e in els)
    has = lambda k: any(e["k"] == k and e["text"] for e in els)  # noqa: E731
    if fam == "doc_type":
        if ks["line"] >= 2 and has("total") and has("ref") and has("payment") and not ks["para"]:
            return "invoice"
        if ks["line"] >= 1 and has("total") and has("payment") and not has("ref") and not ks["para"]:
            return "receipt"
        if has("title") and ks["para"] >= 2 and ks["sig"] >= 2:
            return "contract"
        if ks["field"] >= 3 and not ks["para"] and not ks["line"]:
            return "form"
        if has("sender") and has("recipient") and ks["para"] and not ks["line"] and not ks["field"] and not ks["sig"]:
            return "letter"
        return "other"
    if fam == "field_threshold":
        totals = [e["text"] for e in els if e["k"] == "total"]
        assert len(totals) == 1, f"{row['id']}: {len(totals)} totals"
        m = re.search(r"(\d[\d.,]*\d|\d)\s*EUR", row["question"])
        return "true" if num(totals[0], row["language"]) >= num(m.group(1), row["language"]) else "false"
    # form_complete
    missing = illegible = False
    for f in (e for e in els if e["k"] == "field"):
        req = f["attrs"].get("data-req") == "1"
        label = next(e for e in els if e["field"] == f["i"] and e["k"] == "flabel")["text"]
        if label.endswith("*") != req:
            raise AssertionError(f"{row['id']}: visible '*' disagrees with required flag")
        val = next(e for e in els if e["field"] == f["i"] and e["k"] == "fval")
        blur = re.search(r"blur\(([\d.]+)px\)", val["attrs"].get("style", ""))
        if req and not val["text"]:
            missing = True
        elif req and blur and float(blur.group(1)) >= 2:
            illegible = True
        if not req and blur:
            raise AssertionError(f"{row['id']}: optional field rendered illegible")
    return "missing_required" if missing else "illegible" if illegible else "complete"


def tokens(row, bins) -> list[str]:
    text = " ".join([row["state"], row["question"]] + [o["description"] for o in row["options"]])
    toks = re.findall(r"\w+", text.lower())
    m = NUM_RE.findall(row["question"])
    if m and bins:
        v = num(m[0], row["language"])
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    report = {"criteria": "docs/E38A_CRITERIA_2026-10-14.md", "generator_commit": manifest["generator_commit"],
              "families": {}}
    ok_all = True
    for fam in FAMILIES:
        rows_path = DATA / fam / "rows.jsonl"
        rows = [json.loads(l) for l in rows_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        fails: dict[str, list] = defaultdict(list)
        if hashlib.sha256(rows_path.read_bytes()).hexdigest() != manifest["families"][fam]["rows_sha256"]:
            fails["e_rows_sha"].append(fam)
        by_id = {r["id"]: r for r in rows}
        derived, opposite, n_ft = {}, 0, 0
        for r in rows:
            try:
                DecisionCase.from_dict(r)
            except Exception as exc:  # (e) contract
                fails["e_contract"].append(f"{r['id']}: {exc}")
            png = (DATA / r["image"]).read_bytes()
            if hashlib.sha256(png).hexdigest() != r["image_sha256"]:  # (e) hashes
                fails["e_png_sha"].append(r["id"])
            els, template = parse((DATA / r["html"]).read_text(encoding="utf-8"))
            if template != r["template"]:
                fails["b_template_attr"].append(r["id"])
            try:
                derived[r["id"]] = derive(fam, els, r)
            except AssertionError as exc:
                fails["a_structure"].append(str(exc))
                continue
            if derived[r["id"]] != r["target_option"]:  # (a)
                fails["a_label"].append(f"{r['id']}: row {r['target_option']} derived {derived[r['id']]}")
            visible_text = " ".join([r["state"], r["question"]] + [o["description"] for o in r["options"]])
            if fam == "field_threshold":  # (f) + distractor statistic
                total_txt = next(e["text"] for e in els if e["k"] == "total")
                if total_txt in visible_text:
                    fails["f_text_leak"].append(r["id"])
                lang = r["language"]
                thr = num(re.search(r"(\d[\d.,]*\d|\d)\s*EUR", r["question"]).group(1), lang)
                side = num(total_txt, lang) >= thr
                # dedicated distractor only (line cells concatenate without separators): lower bound
                amounts = [num(NUM_RE.findall(e["text"])[-1], lang) for e in els if e["k"] == "extra"]
                n_ft += 1
                opposite += any((a >= thr) != side for a in amounts)
            if fam == "form_complete":
                for e in els:
                    if e["k"] == "fval" and e["text"] and e["text"] in visible_text:
                        fails["f_text_leak"].append(r["id"])
        # (b) splits
        g_split, t_parts = {}, defaultdict(set)
        for r in rows:
            if g_split.setdefault(r["group_id"], r["split"]) != r["split"]:
                fails["b_group_cross"].append(r["group_id"])
            t_parts[r["template"]].add("heldout" if r["split"] in HELDOUT else r["split"])
        for t, parts in t_parts.items():
            if len(parts) != 1:
                fails["b_template_cross"].append(f"{t}: {sorted(parts)}")
        # (c) counterfactuals
        n_cf = Counter(r["cf_role"] for r in rows)
        for r in rows:
            if r["cf_role"] == "base":
                continue
            base = by_id.get(r["cf_base_id"])
            if not base or base["split"] != r["split"] or base["group_id"] != r["group_id"]:
                fails["c_cf_link"].append(r["id"])
                continue
            flip = derived.get(r["id"]) != derived.get(base["id"])
            if (r["cf_role"] == "decisive") != flip:
                fails["c_cf_label"].append(r["id"])
        # (d) counts and balance
        cell = defaultdict(Counter)
        for r in rows:
            cell[(r["split"], r["language"])][r["target_option"]] += 1
        for split, langs in LANGS.items():
            for lang in langs:
                c = cell.get((split, lang), Counter())
                if sum(c.values()) != COUNTS[split] or len(c) != N_CLASSES[fam] or max(c.values()) - min(c.values()) > 1:
                    fails["d_counts"].append(f"{split}/{lang}: {dict(c)}")
        if set(cell) != {(s, l) for s, ls in LANGS.items() for l in ls}:
            fails["d_counts"].append(f"unexpected cells {sorted(set(cell))}")
        # G2 text-only leak
        train = [r for r in rows if r["split"] == "train"]
        heldout = [r for r in rows if r["split"] in HELDOUT]
        thr_vals = sorted(num(NUM_RE.findall(r["question"])[0], r["language"]) for r in train
                          if NUM_RE.findall(r["question"]))
        bins = [thr_vals[int(len(thr_vals) * q / 10)] for q in range(1, 10)] if thr_vals else []
        g2 = naive_bayes(train, heldout, bins)
        # positive control: the same test must catch a label token injected into state
        inject = lambda rs: [dict(r, state=f"{r['state']} leak{r['target_option']}") for r in rs]  # noqa: E731
        control = naive_bayes(inject(train), inject(heldout), bins)
        g2["positive_control_accuracy"] = control["accuracy"]
        g2["passed"] = g2["passed"] and not control["passed"]  # an insensitive test cannot pass G2
        g1_pass = not fails
        report["families"][fam] = {
            "n_rows": len(rows), "cf_roles": dict(n_cf),
            "G1": {"passed": g1_pass, "failures": {k: (len(v), v[:5]) for k, v in fails.items()}},
            "G2": g2,
        }
        if fam == "field_threshold":
            report["families"][fam]["distractor_opposite_share"] = round(opposite / max(n_ft, 1), 4)
        ok_all &= g1_pass and g2["passed"]
        print(fam, "G1", "PASS" if g1_pass else f"FAIL {dict((k, len(v)) for k, v in fails.items())}",
              "| G2", g2, flush=True)
    report["G1_G2_passed"] = ok_all
    (OUT / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("G1+G2", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
