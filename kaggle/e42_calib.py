"""E42 kernel: calibration + abstention of the two E41 checkpoints (frozen
criteria docs/E42_CRITERIA_2026-10-22.md). One GPU, one shot, no retraining.

E40's protocol with one frozen change: tau is the smallest grid value whose
cal_policy CP95 UPPER bound of selective risk is <= 5 % (E40's point-risk
rule landed on the budget edge; MK-SH-1's pick_threshold stays as a
cross-check). Per seed 42/43: evaluate option probabilities on cal_temperature,
cal_policy, test (non-FR + FR), real (CORD + GUI-Odyssey) and the E35 text
sets; fit T on cal_temperature (frozen NLL grid); pick tau with MK-SH-1's
`policy.pick_threshold` on cal_policy over the frozen grid; held-out tables
with exact Clopper-Pearson 95 % intervals (MK-SH-1 E36 convention, budget on
the upper end) and risk-coverage curves; doc/screen T ablation (descriptive).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

WORK = Path("/kaggle/working")
IMAGE_SIZE = 1024
UPSCALE_TO: int | None = None  # E48: the E46 U1024 input rule; None = E42 pipeline
CRITERIA = "docs/E42_CRITERIA_2026-10-22.md"
SUMMARY_NAME = "e42_summary.json"
EXTRA_ROOTS: dict = {}  # E48: id prefix -> image root of an extra held-out set
MAX_STATE_TOKENS = 700
MAX_OPTION_TOKENS = 96
RISK_BUDGET = 0.05
TAU_GRID = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30..0.95
DOC_FAMILIES = ("doc_type", "field_threshold", "form_complete")

SEED_CKPTS = {  # E41 selected checkpoints (docs/E42_CRITERIA_2026-10-22.md)
    "42": ("mkshv-e41-runs", "a1v_real_s42",
           "e4b4f6f5a56fedae18886e378cd9fb8f84afb1a77536b168a2fdfb990d9641a5"),
    "43": ("mkshv-e41-runs", "a1v_real_s43",
           "30f2eb429c5b25dca00e1ef53fd509232997c34799a5a22fa1c77200619c3819"),
}
SYNTH_DATASETS = {"E38a": "mkshv-e38a", "E38b": "mkshv-e38b"}  # manifest-verified
REAL_DATASET = "mkshv-e38c"
REAL_ROWS_SHA = {  # same frozen pins as E39 (the E38c manifest has no families key)
    "cord/rows.jsonl": "5d0dbc9e7ba3b3301bbcc1d52ecc2fb352b281dcb72c76bbe21ad34e23ab3f09",
    "gui_odyssey/rows.jsonl": "dcdbe6170752738cbb4a72a7f4cafbb56f4b30177cdb79add4c0e2c5a3c2a8da",
}


def log(msg: str) -> None:
    print(f"[e42 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dataset_base(slug: str) -> Path | None:
    """Mount dir of an attached dataset (/kaggle/input/<slug> or nested)."""
    inp = Path("/kaggle/input")
    for base in [inp / slug, *inp.glob(f"*/{slug}"), *inp.glob(f"*/*/{slug}")]:
        if base.is_dir():
            return base
    return None


def find_dir(base: Path, wanted: str) -> Path | None:
    """First dir under base (following symlinks) holding the file `wanted`."""
    for dirpath, _dirs, files in os.walk(base, followlinks=True):
        if wanted in files:
            return Path(dirpath)
    return None


def find_run(base: Path, tag: str) -> Path | None:
    """Dir named `tag` holding best.pt, wherever the upload put it."""
    for dirpath, _dirs, files in os.walk(base, followlinks=True):
        if Path(dirpath).name == tag and "best.pt" in files:
            return Path(dirpath)
    return None


def pip_install() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/Gjusev/MK-SH-1@9e1a226e",
                    "transformers==5.19.0", "pillow>=10.1"], check=True)


def load_rows() -> tuple[dict[str, list[dict]], dict[str, Path]]:
    """({split: rows}, {repo key: root}); every rows file sha-verified."""
    splits: dict[str, list[dict]] = {}
    roots: dict[str, Path] = {}
    for key, slug in SYNTH_DATASETS.items():
        base = dataset_base(slug)
        root = find_dir(base, "manifest.json") if base else None
        if root is None:
            raise SystemExit(f"E42 ABORT: dataset {slug} not attached")
        roots[key] = root
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for family, spec in manifest["families"].items():
            path = root / spec["rows"]
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            if got != spec["rows_sha256"]:
                raise SystemExit(f"E42 ABORT: rows sha mismatch {slug}/{family}: {got}")
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r["split"] in ("cal_temperature", "cal_policy", "test"):
                        splits.setdefault(r["split"], []).append(r)
    base = dataset_base(REAL_DATASET)
    root = find_dir(base, "manifest.json") if base else None
    if root is None:
        raise SystemExit(f"E42 ABORT: dataset {REAL_DATASET} not attached")
    roots["E38c"] = root
    for rel, want in REAL_ROWS_SHA.items():
        got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"E42 ABORT: rows sha mismatch {REAL_DATASET}/{rel}: {got}")
        fam_dir = rel.rsplit("/", 1)[0]
        for line in (root / rel).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["image"] = f"{fam_dir}/{r['image']}"  # E38c stores family-relative paths
                splits.setdefault("real", []).append(r)
    log("rows: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))
    return splits, roots


def extra_sets() -> dict:
    """Extra held-out sets {name: rows} (E48 overrides; E42 has none)."""
    return {}


def image_root(row: dict, roots: dict[str, Path]) -> Path:
    for prefix, root in EXTRA_ROOTS.items():
        if row["id"].startswith(prefix):
            return root
    if row["id"].startswith("e38a"):
        return roots["E38a"]
    if row["id"].startswith("e38b"):
        return roots["E38b"]
    return roots["E38c"]


def load_text_sets() -> dict[str, list[dict]]:
    bundle = None
    for cand in Path("/kaggle/input").rglob("BUNDLE_MANIFEST.json"):
        bundle = cand.parent
        break
    if bundle is None:
        raise SystemExit("E42 ABORT: gjusev/mokka-decisions-bundle not attached")
    man29 = json.loads((bundle / "e29-mix/manifest.json").read_text(encoding="utf-8"))
    want = set(man29["original_dev_ids"])
    task = [r for r in (json.loads(l) for l in (bundle / "data/processed/v1/eval/dev_eval.jsonl")
                        .read_text(encoding="utf-8").splitlines() if l.strip())
            if r["id"] in want]
    testc = [json.loads(l) for l in (bundle / "e26-order/test_c_rows.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"testC": testc, "task": task}


# ---------------------------------------------------------------- math -------
def temp_probs(probs: list[float], t: float) -> list[float]:
    """Softmax temperature applied to probabilities (log p is logits + const)."""
    logs = [math.log(max(p, 1e-12)) / t for p in probs]
    m = max(logs)
    ex = [math.exp(x - m) for x in logs]
    s = sum(ex)
    return [e / s for e in ex]


def nll(cases: list[dict], t: float) -> float:
    return -sum(math.log(max(temp_probs(c["probs"], t)[c["gold"]], 1e-12))
                for c in cases) / len(cases)


def fit_T(cases: list[dict]) -> float:
    """Frozen procedure: grid 0.50..3.00 step 0.01, then ternary refinement
    within +-0.01 (NLL is convex in 1/T, so the search is well posed)."""
    best = min((round(0.50 + 0.01 * i, 2) for i in range(251)), key=lambda t: nll(cases, t))
    lo, hi = best - 0.01, best + 0.01
    for _ in range(30):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(cases, m1) < nll(cases, m2):
            hi = m2
        else:
            lo = m1
    return round((lo + hi) / 2, 4)


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    """Exact two-sided 95 % interval (MK-SH-1 E36 convention)."""
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def risk_at(cases: list[dict], tau: float, t: float) -> dict:
    acc = [c for c in cases if max(temp_probs(c["probs"], t)) >= tau]
    if not acc:
        return {"n": len(cases), "n_acc": 0, "coverage": 0.0, "risk": None,
                "cp95": None, "meets_budget": False}
    k = sum(1 for c in acc if c["pred"] != c["gold"])
    lo, hi = clopper_pearson(k, len(acc))
    return {"n": len(cases), "n_acc": len(acc), "errors": k,
            "coverage": round(len(acc) / len(cases), 4), "risk": round(k / len(acc), 4),
            "cp95": [round(lo, 4), round(hi, 4)], "meets_budget": hi <= RISK_BUDGET}


def pick_tau(cases: list[dict], t: float) -> dict:
    """E42 frozen rule: smallest grid tau whose cal_policy CP95 UPPER bound
    of selective risk is <= RISK_BUDGET (E40's point-risk tau landed on the
    budget edge); none qualifies -> 0.95 with met_target False."""
    for tau in TAU_GRID:
        r = risk_at(cases, tau, t)
        if r["cp95"] is not None and r["cp95"][1] <= RISK_BUDGET:
            return {"threshold": tau, "rule": "cp95_upper", "cal_risk": r["risk"],
                    "cal_cp95": r["cp95"], "met_target": True}
    return {"threshold": 0.95, "rule": "cp95_upper", "met_target": False}


def pick_tau_point_crosscheck(cases: list[dict], t: float) -> dict:
    """MK-SH-1's own rule (point risk), E40's operating point: cross-check only."""
    from mokka_decisions.policy import pick_threshold

    shim = [SimpleNamespace(target_option=str(c["gold"])) for c in cases]
    rows = [{str(i): p for i, p in enumerate(temp_probs(c["probs"], t))} for c in cases]
    return pick_threshold(shim, rows, target_risk=RISK_BUDGET, grid=TAU_GRID)


def mokka_T(cases: list[dict]) -> float:
    """Cross-check only: MK-SH-1's golden-section T on the same split."""
    from mokka_decisions.calibrate import fit_temperature

    shim = [SimpleNamespace(option_ids=[str(i) for i in range(len(c["probs"]))],
                            target_option=str(c["gold"])) for c in cases]
    rows = [{str(i): math.log(max(p, 1e-12)) for i, p in enumerate(c["probs"])} for c in cases]
    return fit_temperature(shim, rows)


def curve(cases: list[dict], t: float) -> list[dict]:
    return [{"tau": tau, **risk_at(cases, tau, t)} for tau in TAU_GRID]


def ece15(cases: list[dict], t: float) -> float:
    bins: list[list[tuple[float, int]]] = [[] for _ in range(15)]
    for c in cases:
        p = temp_probs(c["probs"], t)
        top = max(range(len(p)), key=lambda i: p[i])
        bins[min(14, int(p[top] * 15))].append((p[top], int(top == c["gold"])))
    n = sum(len(b) for b in bins)
    return round(sum(len(b) / n * abs(sum(x for x, _ in b) / len(b) - sum(h for _, h in b) / len(b))
                     for b in bins if b), 4)


# --------------------------------------------------------------- run ---------
def evaluate_set(model, rows: list[dict], roots: dict[str, Path] | None, device) -> list[dict]:
    from mksh_v.arch_a1v import evaluate_a1v
    from mksh_v.train_compare import case_of_row
    from PIL import Image

    out = []
    for s in range(0, len(rows), 400):
        part = rows[s:s + 400]
        cases = [case_of_row(r) for r in part]
        imgs = None
        if roots is not None:
            imgs = [Image.open(image_root(r, roots) / r["image"]).convert("RGB")
                    if r.get("image") else None for r in part]
        ev = evaluate_a1v(model, cases, imgs, device=device, max_side=IMAGE_SIZE, upscale_to=UPSCALE_TO,
                          batch_decisions=16, max_state_tokens=MAX_STATE_TOKENS,
                          max_option_tokens=MAX_OPTION_TOKENS, collect_rows=True)
        for r, er in zip(part, ev["rows"]):
            if er["id"] != r["id"]:
                raise SystemExit(f"E42 ABORT: eval row order {er['id']} != {r['id']}")
            out.append({"id": r["id"], "family": r.get("family", ""),
                        "group_id": r.get("group_id", r["id"]),
                        "probs": er["probabilities"], "gold": er["gold_idx"],
                        "pred": er["pred_idx"]})
    return out


def main() -> None:
    t_start = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    pip_install()
    import torch
    from mksh_v.arch_a1v import NeoMMEMarkerScorer

    if not torch.cuda.is_available():
        raise SystemExit("E42 ABORT: needs a GPU")
    device = "cuda:0"
    torch.manual_seed(42)

    splits, roots = load_rows()
    text = load_text_sets()
    sets = {"cal_temperature": (splits["cal_temperature"], roots),
            "cal_policy": (splits["cal_policy"], roots),
            "test_nonfr": ([r for r in splits["test"] if r["language"] != "fr"], roots),
            "test_fr": ([r for r in splits["test"] if r["language"] == "fr"], roots),
            "cord": ([r for r in splits["real"] if r["id"].startswith("e38c-cord")], roots),
            "gui": ([r for r in splits["real"] if not r["id"].startswith("e38c-cord")], roots),
            "testC": (text["testC"], None), "task": (text["task"], None),
            **{k: (v, roots) for k, v in extra_sets().items()}}
    log("sets: " + ", ".join(f"{k} {len(v[0])}" for k, v in sets.items()))

    summary = {"status": "running", "criteria": CRITERIA, "upscale_to": UPSCALE_TO,
               "tau_grid": TAU_GRID, "risk_budget": RISK_BUDGET,
               "interval": "Clopper-Pearson exact, 95 % two-sided (MK-SH-1 E36); budget on upper end",
               "set_sizes": {k: len(v[0]) for k, v in sets.items()}, "seeds": {}}
    for seed, (slug, tag, sha) in SEED_CKPTS.items():
        base = dataset_base(slug)
        run = find_run(base, tag) if base else None
        if run is None:
            raise SystemExit(f"E42 ABORT: {slug}/{tag} not attached")
        got = hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()
        if got != sha:
            raise SystemExit(f"E42 ABORT: {tag} sha {got} != pinned {sha}")
        model = NeoMMEMarkerScorer().to(device)
        model.ensure_markers(32)  # E39 tables are 32-marker wide
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu",
                                         weights_only=True)["model"])
        model.to(device).eval()
        log(f"s{seed}: {tag} loaded (sha {got[:16]})")

        evald: dict[str, list[dict]] = {}
        for name, (rows, rts) in sets.items():
            t0 = time.time()
            evald[name] = evaluate_set(model, rows, rts, device)
            (WORK / f"percase_s{seed}__{name}.jsonl").write_text(
                "".join(json.dumps({**c, "probs": [round(p, 6) for p in c["probs"]]}) + "\n"
                        for c in evald[name]), encoding="utf-8")
            log(f"s{seed} {name}: {len(rows)} rows in {time.time() - t0:.0f}s")
        del model
        torch.cuda.empty_cache()

        # frozen order: T on cal_temperature -> tau on cal_policy -> held-out tables
        cal = evald["cal_temperature"]
        t_star = fit_T(cal)
        tau = pick_tau(evald["cal_policy"], t_star)
        tau_star = tau["threshold"]
        tables, curves = {}, {}
        for name, cs in evald.items():
            if name in ("cal_temperature", "cal_policy"):
                continue
            tables[name] = {"overall": risk_at(cs, tau_star, t_star),
                            "ece15_raw": ece15(cs, 1.0), "ece15_T": ece15(cs, t_star),
                            "nll_raw": round(nll(cs, 1.0), 4), "nll_T": round(nll(cs, t_star), 4)}
            curves[name] = curve(cs, t_star)
            if name in ("test_nonfr", "test_fr"):
                for f in sorted({c["family"] for c in cs}):
                    sub = [c for c in cs if c["family"] == f]
                    tables[name][f"family:{f}"] = risk_at(sub, tau_star, t_star)
                    curves[f"{name}|{f}"] = curve(sub, t_star)
        # ablation (descriptive, never used for claims)
        t_doc = fit_T([c for c in cal if c["family"] in DOC_FAMILIES])
        t_screen = fit_T([c for c in cal if c["family"] not in DOC_FAMILIES])
        tn = evald["test_nonfr"]
        doc = [c for c in tn if c["family"] in DOC_FAMILIES]
        scr = [c for c in tn if c["family"] not in DOC_FAMILIES]
        ablation = {"T_doc": t_doc, "T_screen": t_screen,
                    "test_nonfr_doc": {"ece15_T": ece15(doc, t_star), "ece15_T_doc": ece15(doc, t_doc),
                                       "nll_T": round(nll(doc, t_star), 4), "nll_T_doc": round(nll(doc, t_doc), 4)},
                    "test_nonfr_screen": {"ece15_T": ece15(scr, t_star), "ece15_T_screen": ece15(scr, t_screen),
                                          "nll_T": round(nll(scr, t_star), 4),
                                          "nll_T_screen": round(nll(scr, t_screen), 4)}}
        summary["seeds"][seed] = {
            "checkpoint": f"{slug}/{tag}/best.pt", "sha256": got,
            "T": t_star, "T_mokka_crosscheck": mokka_T(cal), "tau": tau,
            "tau_point_crosscheck": pick_tau_point_crosscheck(evald["cal_policy"], t_star),
            "cal_policy_at_tau": risk_at(evald["cal_policy"], tau_star, t_star),
            "raw_acc": {n: round(sum(c["pred"] == c["gold"] for c in cs) / len(cs), 4)
                        for n, cs in evald.items()},
            "tables": tables, "curves": curves, "ablation_T_by_modality": ablation,
            "budget_flags": {n: tables[n]["overall"]["meets_budget"] for n in tables},
        }
        log(f"s{seed}: T*={t_star} tau={tau} flags={summary['seeds'][seed]['budget_flags']}")
        (WORK / SUMMARY_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    names = next(iter(summary["seeds"].values()))["budget_flags"]
    summary["claimable_sets_all_seeds"] = sorted(
        n for n in names if all(s["budget_flags"][n] for s in summary["seeds"].values()))
    summary["status"] = "completed"
    summary["wall_s"] = round(time.time() - t_start, 1)
    (WORK / SUMMARY_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"E42 done in {summary['wall_s']}s; claimable: {summary['claimable_sets_all_seeds']}")


if __name__ == "__main__":
    main()
