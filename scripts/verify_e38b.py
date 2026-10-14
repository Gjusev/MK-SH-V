"""Independent verifier for E38b (criteria: docs/E38B_CRITERIA_2026-10-14.md).

Does NOT import the generator. Re-derives every label from the rendered HTML
sidecars with its own parser and rules, and checks splits, counterfactuals,
counts, hashes, contract validity and text leakage (G1), then runs the
text-only naive Bayes leak test (G2). Writes runs/e38b-verify/verification.json.
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
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/e38b"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runs/e38b-verify"
FAMILIES = ("screen_state", "task_done", "source_conflict", "urgency")
COUNTS = {"train": 1000, "dev": 100, "cal_temperature": 150, "cal_policy": 150, "test": 200}
LANGS = {"train": ("en", "es", "de"), "dev": ("en", "es", "de"), "cal_temperature": ("en", "es", "de"),
         "cal_policy": ("en", "es", "de"), "test": ("en", "es", "de", "fr")}
HELDOUT = ("dev", "cal_temperature", "cal_policy")
N_CLASSES = {"screen_state": 5, "task_done": 2, "source_conflict": 2, "urgency": 5}
NUM_RE = re.compile(r"\d[\d.,]*\d|\d")

# own localized copy of the source_conflict field names (per language)
SC_NAMES = {
    "en": ("Priority", "Owner", "Status", "Category", "Channel"),
    "es": ("Prioridad", "Responsable", "Estado", "Categoría", "Canal"),
    "de": ("Priorität", "Verantwortlich", "Status", "Kategorie", "Kanal"),
    "fr": ("Priorité", "Responsable", "Statut", "Catégorie", "Canal"),
}
# option-description leak pools for G1(f)
IMPACT_WORDS = ("minor", "moderate", "critical", "menor", "moderado", "crítico",
                "gering", "mittel", "kritisch", "mineur", "modéré", "critique")
TASK_TOKEN = re.compile(r"\w+\.(xlsx|csv|tar|png|log|mp4|db|pdf)|[A-Z]+-\d+")


class Elements(HTMLParser):
    """Collect data-k elements with attrs, text and parent field index."""

    def __init__(self):
        super().__init__()
        self.stack, self.els, self.template = [], [], None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "data-template" in a:
            self.template = a["data-template"]
        rec = None
        if "data-k" in a:
            parent = next((r["i"] for r in reversed(self.stack) if r), None)
            rec = {"k": a["data-k"], "attrs": a, "text": [], "field": parent, "i": len(self.els)}
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


def visible_text(row) -> str:
    return " ".join([row["state"], row["question"]] + [o["description"] for o in row["options"]])


def _ss_kind_map() -> dict:
    return {"success": "success", "error": "error", "loading": "loading",
            "form": "needs_input", "blocked": "blocked"}


def derive(fam: str, els, row) -> str:
    ks = Counter(e["k"] for e in els)
    if fam == "screen_state":
        kinds = [k for k in _ss_kind_map() if ks[k]]
        if len(kinds) != 1:
            raise AssertionError(f"{row['id']}: {len(kinds)} state kinds {kinds}")
        kind = kinds[0]
        if kind == "form" and not any(e["k"] == "fin" and not e["text"] for e in els):
            raise AssertionError(f"{row['id']}: form without an empty required input")
        if kind == "error" and not ks["retry"]:
            raise AssertionError(f"{row['id']}: error without retry")
        if kind == "blocked" and not ks["lockglyph"]:
            raise AssertionError(f"{row['id']}: blocked without lock badge")
        return _ss_kind_map()[kind]
    if fam == "task_done":
        m = re.search(r'"([^"]+)"', row["state"])
        assert m, f"{row['id']}: no quoted goal token in state"
        rows = [e for e in els if e["k"] == "taskitem"]
        hits = []
        for t in rows:
            tn = next(e["text"] for e in els if e["field"] == t["i"] and e["k"] == "tname")
            if m.group(1) in tn:
                hits.append(t)
        if len(hits) != 1:
            raise AssertionError(f"{row['id']}: {len(hits)} rows match goal token {m.group(1)!r}")
        t = hits[0]
        return "true" if any(e["k"] == "done-mark" and e["field"] == t["i"] for e in els) else "false"
    if fam == "source_conflict":
        names = SC_NAMES[row["language"]]
        mism = 0
        for name in names:
            m = re.search(re.escape(name) + r":\s*([^.]+)\.", row["state"])
            if not m:
                continue
            dom = [e["text"] for e in els if e["k"] == "sv" and e["attrs"].get("data-f") == fid_of(name, row["language"])]
            if not dom:
                raise AssertionError(f"{row['id']}: claimed field {name} missing on screen")
            if dom[0] != m.group(1).strip():
                mism += 1
        if not any(re.search(re.escape(n) + r":", row["state"]) for n in names):
            raise AssertionError(f"{row['id']}: no parsable claims in state")
        return "conflict" if mism else "agree"
    # urgency
    kinds = [k for k in ("warnbanner", "error", "blocked") if ks[k]]
    if len(kinds) > 1:
        raise AssertionError(f"{row['id']}: multiple banners {kinds}")
    pts = {"warnbanner": 1, "error": 2, "blocked": 3}.get(kinds[0] if kinds else "", 0)
    dl = next(e["text"] for e in els if e["k"] == "deadline")
    h = int(re.search(r"(\d+)\s*h", dl).group(1))
    return str(1 + pts + (1 if h <= 4 else 0))


def fid_of(name: str, lang: str) -> str:
    """Verifier-own field-name -> fid table (no generator import)."""
    table = {
        "en": {"Priority": "priority", "Owner": "owner", "Status": "status", "Category": "category",
               "Channel": "channel"},
        "es": {"Prioridad": "priority", "Responsable": "owner", "Estado": "status",
               "Categoría": "category", "Canal": "channel"},
        "de": {"Priorität": "priority", "Verantwortlich": "owner", "Status": "status",
               "Kategorie": "category", "Kanal": "channel"},
        "fr": {"Priorité": "priority", "Responsable": "owner", "Statut": "status",
               "Catégorie": "category", "Canal": "channel"},
    }
    return table[lang][name]

def tokens(row, bins) -> list[str]:
    text = " ".join([row["state"], row["question"]] + [o["description"] for o in row["options"]])
    toks = re.findall(r"\w+", text.lower())
    m = NUM_RE.findall(row["question"])
    if m and bins:
        v = float(m[0].replace(",", "") if row["language"] == "en" else
                  m[0].replace(".", "").replace(",", "."))
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

def text_leak_checks(fam: str, els, row) -> list[str]:
    """G1(f) per family. Returns failure labels."""
    fails = []
    vt = visible_text(row)
    if fam == "screen_state":
        for e in els:
            if e["k"] in ("msg", "sumline", "ecode") and e["text"] and e["text"] in vt:
                fails.append("f_ss_message_leak")
                break
    elif fam == "task_done":
        if any(TASK_TOKEN.search(o["description"]) for o in row["options"]):
            fails.append("f_td_status_words")
    elif fam == "source_conflict":
        pools = set()
        for name in SC_NAMES[row["language"]]:
            m = re.search(re.escape(name) + r":\s*([^.]+)\.", row["state"])
            if m:
                pools.add(m.group(1).strip())
        for o in row["options"]:
            if any(v and v in o["description"] for v in pools):
                fails.append("f_sc_values_in_options")
                break
    else:  # urgency
        qs = row["question"] + " " + " ".join(o["description"] for o in row["options"])
        if any(t in qs for t in ("warnbanner", "LOCKED", "BLOQUEADO", "GESPERRT", "VERROUILLÉ")):
            fails.append("f_ur_banner_names")
        m = re.search(r"#(\d+)", row["state"])
        if m and m.group(0) in qs:
            fails.append("f_ur_ticket_number")
    if re.search(r"<(img|doc|row|eos|mask|query)>|\[O\d+\]", vt):
        fails.append("f_marker_tokens")
    return fails


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    report = {"criteria": "docs/E38B_CRITERIA_2026-10-14.md", "generator_commit": manifest["generator_commit"],
              "families": {}}
    ok_all = True
    for fam in FAMILIES:
        rows_path = DATA / fam / "rows.jsonl"
        rows = [json.loads(l) for l in rows_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        fails: dict[str, list] = defaultdict(list)
        if hashlib.sha256(rows_path.read_bytes()).hexdigest() != manifest["families"][fam]["rows_sha256"]:
            fails["e_rows_sha"].append(fam)
        by_id = {r["id"]: r for r in rows}
        derived = {}
        for r in rows:
            try:
                DecisionCase.from_dict(r)
            except Exception as exc:
                fails["e_contract"].append(f"{r['id']}: {exc}")
            png = (DATA / r["image"]).read_bytes()
            if hashlib.sha256(png).hexdigest() != r["image_sha256"]:
                fails["e_png_sha"].append(r["id"])
            els, template = parse((DATA / r["html"]).read_text(encoding="utf-8"))
            if template != r["template"]:
                fails["b_template_attr"].append(r["id"])
            pair = f"{r['app']}|{r['template']}"
            try:
                derived[r["id"]] = derive(fam, els, r)
            except AssertionError as exc:
                fails["a_structure"].append(str(exc))
                continue
            if derived[r["id"]] != r["target_option"]:
                fails["a_label"].append(f"{r['id']}: row {r['target_option']} derived {derived[r['id']]}")
            for lab in text_leak_checks(fam, els, r):
                fails["f_leak"].append(f"{r['id']}:{lab}")
        # (b) splits by group and pair
        g_split, p_parts = {}, defaultdict(set)
        for r in rows:
            if g_split.setdefault(r["group_id"], r["split"]) != r["split"]:
                fails["b_group_cross"].append(r["group_id"])
            pair = f"{r['app']}|{r['template']}"
            p_parts[pair].add("heldout" if r["split"] in HELDOUT else r["split"])
        for p, parts in p_parts.items():
            if len(parts) != 1:
                fails["b_pair_cross"].append(f"{p}: {sorted(parts)}")
        # (c) counterfactuals
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
        # G2
        train = [r for r in rows if r["split"] == "train"]
        heldout = [r for r in rows if r["split"] in HELDOUT]
        g2 = naive_bayes(train, heldout, bins=[])
        inject = lambda rs: [dict(r, state=f"{r['state']} leak{r['target_option']}") for r in rs]  # noqa: E731
        control = naive_bayes(inject(train), inject(heldout), bins=[])
        g2["positive_control_accuracy"] = control["accuracy"]
        g2["passed"] = g2["passed"] and not control["passed"]
        g1_pass = not fails
        report["families"][fam] = {
            "n_rows": len(rows), "cf_roles": dict(Counter(r["cf_role"] for r in rows)),
            "G1": {"passed": g1_pass, "failures": {k: (len(v), v[:5]) for k, v in fails.items()}},
            "G2": g2,
        }
        ok_all &= g1_pass and g2["passed"]
        print(fam, "G1", "PASS" if g1_pass else f"FAIL {dict((k, len(v)) for k, v in fails.items())}",
              "| G2", g2, flush=True)
    report["G1_G2_passed"] = ok_all
    (OUT / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("G1+G2", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())

# __APP__
