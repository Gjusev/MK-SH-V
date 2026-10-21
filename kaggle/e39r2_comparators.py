"""E39-r2 kernel: re-run the two reported E39 comparators with their
implementation defects fixed (frozen criteria docs/E39R2_CRITERIA_2026-10-22.md).

SmolVLM2-500M (task-trained, E39 recipe) trains on GPU 0 through the E39
worker while jev-vlm-4b (zero-shot, its native `decide()` API) runs on GPU 1.
Writes per-case rows; the paired comparison against the official E39 v7
A1-V per-case files is computed post hoc with the frozen `paired_delta_ci`.
The E39 kernel ships next to `mksh_v` as the module `e39_train`.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

WORK = Path("/kaggle/working")
WALL_BUDGET_S = 6 * 3600  # quota guard: stop waiting / sampling after 6 h
JEV_SAMPLE_SHA = "12791f79e26eb9d0196bf1c72b08fd5236476c150ae07250d446786db475c592"
NOUL_IDS = ["false", "true"]


def log(msg: str) -> None:
    print(f"[e39r2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cap(img, side: int):
    """Longest side <= side (the A1-V E39 visual budget)."""
    w, h = img.size
    s = min(1.0, side / max(w, h))
    if s < 1.0:
        from PIL import Image

        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return img


def jev_question(case) -> tuple[dict, str]:
    """Map a case onto jev's typed question (criteria section of E39-r2)."""
    ids = list(case.option_ids)
    if sorted(ids) == NOUL_IDS:
        return {"type": "noul", "instructions": case.question}, "noul"
    return {"type": "choice", "instructions": case.question,
            "criteria": {o.id: o.description for o in case.options}}, "choice"


def jev_probs(case, answer: dict, kind: str) -> list[float]:
    """Probabilities aligned with case.option_ids from a decide() answer."""
    if kind == "noul":
        p_true = float(answer["probability"])
        return [p_true if oid == "true" else 1.0 - p_true for oid in case.option_ids]
    dist = answer["probabilities"]
    return [float(dist[oid]) for oid in case.option_ids]


def row_out(r: dict, case, probs: list[float] | None, error: str | None = None) -> dict:
    gold = case.option_ids.index(case.target_option)
    if probs is None:
        return {"id": r["id"], "group_id": r.get("group_id", r["id"]),
                "family": r.get("family", ""), "probabilities": None, "gold_idx": gold,
                "pred_idx": None, "correct": 0, "answered": False, "error": error}
    pred = max(range(len(probs)), key=lambda i: probs[i])
    return {"id": r["id"], "group_id": r.get("group_id", r["id"]),
            "family": r.get("family", ""), "probabilities": [round(p, 6) for p in probs],
            "gold_idx": gold, "pred_idx": pred, "correct": int(pred == gold), "answered": True}


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    return {"n": n, "answered": sum(r["answered"] for r in rows),
            "accuracy": round(sum(r["correct"] for r in rows) / n, 4) if n else None}


def write_rows(name: str, rows: list[dict]) -> None:
    (WORK / f"{name}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def load_real_rows(T) -> tuple[list[dict], Path]:
    root = T.kaggle_copy(T.REAL_EVAL_REPO)
    if root is None:
        raise SystemExit("E39R2 ABORT: mkshv-e38c not attached")
    rows = []
    for rel, want in T.REAL_EVAL_ROWS_SHA.items():
        got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"E39R2 ABORT: rows sha mismatch {rel}: {got}")
        fam = rel.rsplit("/", 1)[0]
        for line in (root / rel).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["image"] = f"{fam}/{r['image']}"
                rows.append(r)
    return rows, root


def run_jev(T, sample, real_rows, roots, real_root, t_start) -> dict:
    import torch
    from mksh_v.train_compare import case_of_row
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(T.JEV_ID, revision=T.JEV_REVISION,
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(T.JEV_ID, revision=T.JEV_REVISION,
                                      trust_remote_code=True, dtype=torch.float16).to("cuda:1")
    model.eval()
    log("jev loaded on cuda:1")
    out = {}
    for name, rows, opener in (
            ("jev_sample", sample, lambda r: T.open_image(r, roots)),
            ("jev_real", real_rows, lambda r: Image.open(real_root / r["image"]).convert("RGB"))):
        res = []
        for i, r in enumerate(rows):
            case = case_of_row(r)
            if time.time() - t_start > WALL_BUDGET_S:
                res.append(row_out(r, case, None, "wall budget"))
                continue
            q, kind = jev_question(case)
            try:
                ans = model.decide(processor, state=case.state, questions={"q": q},
                                   images=[cap(opener(r), T.IMAGE_SIZE)])["q"]
                res.append(row_out(r, case, jev_probs(case, ans, kind)))
            except Exception as exc:  # recorded as unanswered, never dropped
                res.append(row_out(r, case, None, f"{type(exc).__name__}: {exc}"[:300]))
            if (i + 1) % 200 == 0:
                write_rows(name, res)
                log(f"{name}: {i + 1}/{len(rows)} {summarise(res)}")
        write_rows(name, res)
        out[name] = summarise(res)
        log(f"{name} done: {out[name]}")
    del model
    torch.cuda.empty_cache()
    return out


def eval_smol(T, smol, rows, opener) -> list[dict]:
    import torch
    from mksh_v.train_compare import case_of_row

    out = []
    smol.model.eval()
    for s in range(0, len(rows), 16):
        sub = rows[s:s + 16]
        cases = [case_of_row(r) for r in sub]
        imgs = [opener(r) for r in sub] if opener else None
        with torch.no_grad():
            logits = smol.batch_logits(cases, imgs)
        for j, (r, c) in enumerate(zip(sub, cases)):
            out.append(row_out(r, c, logits[j, :len(c.options)].softmax(-1).tolist()))
    return out


def main() -> None:
    t_start = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    import e39_train as T  # shipped next to mksh_v (pkg root)

    T.pip_install()
    import torch

    if torch.cuda.device_count() < 2:
        raise SystemExit("E39R2 ABORT: needs the T4x2 session (2 GPUs)")
    from mksh_v.train_compare import sha_sample
    from PIL import Image

    roots = T.download_data()
    T.verify_rows_sha(roots)
    splits = T.load_e38_rows(roots)
    test_nonfr = [r for r in splits["test"] if r["language"] != "fr"]
    test_fr = [r for r in splits["test"] if r["language"] == "fr"]
    real_rows, real_root = load_real_rows(T)
    sample = sha_sample(test_nonfr, per_family=200)
    got = hashlib.sha256("\n".join(r["id"] for r in sample).encode()).hexdigest()
    if got != JEV_SAMPLE_SHA:
        raise SystemExit(f"E39R2 ABORT: jev sample != E39 v7 sample ({got})")
    log(f"rows: test non-FR {len(test_nonfr)}, FR {len(test_fr)}, real {len(real_rows)}, "
        f"jev sample {len(sample)} (== v7)")

    summary = {"status": "running", "criteria": "docs/E39R2_CRITERIA_2026-10-22.md",
               "smol": {"status": "not_run"}, "jev": {"status": "not_run"}}

    def save():
        (WORK / "e39r2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # SmolVLM2: smoke, then the 2,000-step run in the background on GPU 0
    smol_proc = None
    try:
        T.run_and_log(T.spawn("smol", 42, 0, T.SMOKE_STEPS), "smoke-smol")
        smol_proc = T.spawn("smol", 42, 0, T.STEPS)
        log("smol 2000-step run started on GPU 0")
    except SystemExit as exc:
        summary["smol"] = {"status": "not_run", "reason": f"smoke failed: {exc}"[:400]}
    save()

    # jev on GPU 1 meanwhile
    try:
        summary["jev"] = {"status": "completed", "regimen": "zero-shot", "model": T.JEV_ID,
                          "revision": T.JEV_REVISION, **run_jev(T, sample, real_rows, roots,
                                                                real_root, t_start)}
    except Exception as exc:  # noqa: BLE001 - recorded
        summary["jev"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:400]}
    save()

    if smol_proc is not None:
        proc = smol_proc[0]
        try:
            proc.wait(timeout=max(60, WALL_BUDGET_S - (time.time() - t_start)))
            T.run_and_log(smol_proc, "smol-s42")  # prints the worker log; raises on failure
            smol = T.SmolScorer("cuda:0")
            T.load_best_weights(WORK / "runs/smol_s42/best.pt", smol.model)
            res = {}
            for name, rows, opener in (
                    ("smol_test_nonfr", test_nonfr, lambda r: T.open_image(r, roots)),
                    ("smol_test_fr", test_fr, lambda r: T.open_image(r, roots)),
                    ("smol_real", real_rows,
                     lambda r: Image.open(real_root / r["image"]).convert("RGB"))):
                rows_out = eval_smol(T, smol, rows, opener)
                write_rows(name, rows_out)
                res[name] = summarise(rows_out)
                log(f"{name}: {res[name]}")
            bundle = T.find_bundle()
            text = {"testC": [json.loads(l) for l in (bundle / "e26-order/test_c_rows.jsonl")
                              .read_text(encoding="utf-8").splitlines() if l.strip()]}
            man29 = json.loads((bundle / "e29-mix/manifest.json").read_text(encoding="utf-8"))
            want = set(man29["original_dev_ids"])
            text["task"] = [r for r in (json.loads(l) for l in
                                        (bundle / "data/processed/v1/eval/dev_eval.jsonl")
                                        .read_text(encoding="utf-8").splitlines() if l.strip())
                            if r["id"] in want]
            for name, rows in text.items():
                rows_out = eval_smol(T, smol, rows, None)
                write_rows(f"smol_{name}", rows_out)
                res[f"smol_{name}"] = summarise(rows_out)
            summary["smol"] = {"status": "completed", "regimen": "task-trained",
                               "train_summary": json.loads((WORK / "runs/smol_s42/summary.json")
                                                           .read_text(encoding="utf-8")), **res}
        except Exception as exc:  # noqa: BLE001 - incl. timeout and worker failure
            if proc.poll() is None:
                proc.kill()
            summary["smol"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:400]}
        except SystemExit as exc:
            summary["smol"] = {"status": "failed", "error": str(exc)[:400]}

    summary["status"] = "completed"
    summary["wall_s"] = round(time.time() - t_start, 1)
    save()
    log(f"E39-r2 done in {summary['wall_s']}s: smol {summary['smol'].get('status')}, "
        f"jev {summary['jev'].get('status')}")


if __name__ == "__main__":
    main()
