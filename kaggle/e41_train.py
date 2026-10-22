"""E41 kernel: the E39 A1-V recipe + CC-BY real training data
(frozen criteria docs/E41_CRITERIA_2026-10-22.md).

Seeds 42 (GPU 0) and 43 (GPU 1) train in parallel as subprocess workers.
Every 4th microbatch is E35 text replay (slot % 4 == 3, as in E39) and slot
% 4 == 1 draws from the real-train pool; the rest is E39's synthetic pool.
Each worker evaluates its selected checkpoint once and writes per-case
files; the gates are computed post hoc against the official E39 v7 files.
The E39 kernel ships next to `mksh_v` as the module `e39_train`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REALTRAIN_SLUG = "mkshv-e41-realtrain"
REALTRAIN_ROWS_SHA = {  # data/e41/manifest.json (generated after the freeze)
    "cord": "015cce63913a4fdc6ccb5b861387d531991123ee2d20c736055d77afb462ee6a",
    "gui_odyssey": "c33d366beaf3116cf4e2428f119880f91cb71d6da05cb5969a9da75ffedf6f51",
}
SEEDS = {42: 0, 43: 1}  # seed -> GPU
REAL_SLOT = 1  # microbatch slot % 4 == 1 -> real-train pool
TRAIN_DEADLINE_S = int(3.5 * 3600)  # stop training (keep best) so the eval always runs
WALL_BUDGET_S = int(4.2 * 3600)  # orchestrator kills workers after this
SMOKE_STEPS = 20
SMOKE_EVAL_ROWS = 24


def log(msg: str) -> None:
    print(f"[e41 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def realtrain_root() -> Path:
    inp = Path("/kaggle/input")
    for base in [inp / REALTRAIN_SLUG, *inp.glob(f"*/{REALTRAIN_SLUG}"),
                 *inp.glob(f"*/*/{REALTRAIN_SLUG}")]:
        for dirpath, _dirs, files in os.walk(base, followlinks=True):
            if "manifest.json" in files and Path(dirpath, "cord").is_dir():
                return Path(dirpath)
    raise SystemExit(f"E41 ABORT: {REALTRAIN_SLUG} not attached")


def load_realtrain(root: Path) -> tuple[list[dict], list[dict]]:
    rows = []
    for name, want in REALTRAIN_ROWS_SHA.items():
        p = root / name / "rows.jsonl"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"E41 ABORT: realtrain rows sha mismatch {name}: {got}")
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "dev"]


def load_real_eval(T) -> tuple[list[dict], Path]:
    root = T.kaggle_copy(T.REAL_EVAL_REPO)
    if root is None:
        raise SystemExit("E41 ABORT: mkshv-e38c not attached")
    rows = []
    for rel, want in T.REAL_EVAL_ROWS_SHA.items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != want:
            raise SystemExit(f"E41 ABORT: E38c rows sha mismatch {rel}")
        fam = rel.rsplit("/", 1)[0]
        for line in (root / rel).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["image"] = f"{fam}/{r['image']}"
                rows.append(r)
    return rows, root


def load_text_sets(T) -> dict[str, list[dict]]:
    bundle = T.find_bundle()
    man29 = json.loads((bundle / "e29-mix/manifest.json").read_text(encoding="utf-8"))
    want = set(man29["original_dev_ids"])
    task = [r for r in (json.loads(l) for l in (bundle / "data/processed/v1/eval/dev_eval.jsonl")
                        .read_text(encoding="utf-8").splitlines() if l.strip()) if r["id"] in want]
    testc = [json.loads(l) for l in (bundle / "e26-order/test_c_rows.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"testC": testc, "task": task}


def slot_pool(slot: int) -> str:
    """Microbatch source by slot (E39: slot % 4 == 3 is replay)."""
    if slot % 4 == 3:
        return "replay"
    return "real" if slot % 4 == REAL_SLOT else "synthetic"


def selection_key(synth_macro: float, real_macro: float, synth_acc: float) -> tuple:
    """Frozen selection: mean of synthetic dev-half macro and real-dev macro;
    tie -> higher synthetic dev-half accuracy."""
    return ((synth_macro + real_macro) / 2, synth_acc)


def evaluate_rows(model, rows, opener, device) -> list[dict]:
    from mksh_v.arch_a1v import evaluate_a1v
    from mksh_v.train_compare import case_of_row

    import e39_train as T

    out = []
    for s in range(0, len(rows), 400):
        part = rows[s:s + 400]
        imgs = [opener(r) for r in part] if opener else None
        ev = evaluate_a1v(model, [case_of_row(r) for r in part], imgs, device=device,
                          max_side=T.IMAGE_SIZE, batch_decisions=16,
                          max_state_tokens=T.MAX_STATE_TOKENS,
                          max_option_tokens=T.MAX_OPTION_TOKENS, collect_rows=True)
        for r, er in zip(part, ev["rows"]):
            if er["id"] != r["id"]:
                raise SystemExit(f"E41 ABORT: eval row order {er['id']} != {r['id']}")
            out.append({"id": r["id"], "group_id": r.get("group_id", r["id"]),
                        "family": r.get("family", ""), "probabilities": er["probabilities"],
                        "gold_idx": er["gold_idx"], "pred_idx": er["pred_idx"],
                        "correct": int(er["pred_idx"] == er["gold_idx"])})
    return out


def run_worker(seed: int, steps: int, smoke: bool, t0: float) -> None:
    import torch
    from mksh_v.arch_a1v import forward_a1v
    from mksh_v.train_compare import case_of_row, fixed_dev_half, macro_family_accuracy
    from PIL import Image

    import e39_train as T

    device = "cuda"
    tag = f"a1v_real_s{seed}" + ("_smoke" if smoke else "")
    out_dir = WORK / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    roots = T.download_data()
    T.verify_rows_sha(roots)
    splits = T.load_e38_rows(roots)
    synth = splits["train"]
    dev_half = fixed_dev_half(splits["dev"], seed=T.DEV_HALF_SEED, fraction=0.5)
    dev_sel = [r for r in splits["dev"] if r["id"] in dev_half]
    rt_root = realtrain_root()
    real, real_dev = load_realtrain(rt_root)
    bundle = T.find_bundle()
    replay = [case_of_row(r) for r in T.load_replay_mix(bundle)]
    if smoke:
        dev_sel, real_dev = dev_sel[:SMOKE_EVAL_ROWS], real_dev[:SMOKE_EVAL_ROWS]
    log(f"[{tag}] synthetic {len(synth)}, real-train {len(real)}, real-dev {len(real_dev)}, "
        f"replay {len(replay)}, dev-half {len(dev_sel)}")

    def open_synth(r):
        return T.open_image(r, roots)

    def open_real(r):
        return Image.open(rt_root / r["image"]).convert("RGB")

    model = T.build_a1v(device)
    kw = dict(max_state_tokens=T.MAX_STATE_TOKENS, max_option_tokens=T.MAX_OPTION_TOKENS)
    pools = {"synthetic": (model.encode_cases([case_of_row(r) for r in synth], **kw), synth, open_synth),
             "real": (model.encode_cases([case_of_row(r) for r in real], **kw), real, open_real),
             "replay": (model.encode_cases(replay, **kw), None, None)}
    rng = random.Random(seed)
    orders = {k: list(range(len(v[0]))) for k, v in pools.items()}
    for o in orders.values():
        rng.shuffle(o)
    cursors = {k: 0 for k in pools}

    def take(pool: str, m: int) -> list[int]:
        if cursors[pool] + m > len(orders[pool]):
            rng.shuffle(orders[pool])
            cursors[pool] = 0
        idx = orders[pool][cursors[pool]:cursors[pool] + m]
        cursors[pool] += m
        return idx

    def eval_dev():
        s_rows = evaluate_rows(model, dev_sel, open_synth, device)
        r_rows = evaluate_rows(model, real_dev, open_real, device)
        s_macro = macro_family_accuracy(s_rows)
        s_acc = sum(r["correct"] for r in s_rows) / len(s_rows)
        r_macro = macro_family_accuracy(r_rows)
        return s_macro, r_macro, s_acc

    micro = T.ARM_MICROBATCH["a1v"]
    accum = T.EFFECTIVE_BATCH // micro
    opt = torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, T.lr_lambda_of(steps))
    best, curves, stopped_at = None, [], None
    best_path = out_dir / "best.pt"
    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        for slot in range(accum):
            pool = slot_pool(slot)
            enc, rows, opener = pools[pool]
            idx = take(pool, micro)
            imgs = [None] * len(idx) if opener is None else [opener(rows[i]) for i in idx]
            batch = model.build_batch([enc[i] for i in idx], imgs, device, max_side=T.IMAGE_SIZE)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = forward_a1v(model, batch).float()
                loss = torch.nn.functional.cross_entropy(
                    logits, batch["targets"], ignore_index=-100) / accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        late = time.time() - t0 > TRAIN_DEADLINE_S
        if step % T.DEV_EVERY == 0 or step == steps or late:
            with torch.no_grad():
                s_macro, r_macro, s_acc = eval_dev()
            key = selection_key(s_macro, r_macro, s_acc)
            curves.append({"step": step, "synthetic_dev_half_macro": round(s_macro, 4),
                           "real_dev_macro": round(r_macro, 4), "synthetic_dev_half_acc": round(s_acc, 4)})
            log(f"[{tag}] step {step}/{steps} synth {s_macro:.4f} real {r_macro:.4f}")
            if best is None or key >= best["key"]:
                best = {"key": key, "step": step}
                T.save_best_weights(best_path, model)
            model.train()
        if late:
            stopped_at = step
            log(f"[{tag}] train deadline reached at step {step}: evaluating best so far")
            break
    (out_dir / "curves.jsonl").write_text("".join(json.dumps(c) + "\n" for c in curves), encoding="utf-8")

    T.load_best_weights(best_path, model)
    model.to(device)
    test = splits["test"]
    real_eval, real_root = load_real_eval(T)
    sets = {"test_nonfr": ([r for r in test if r["language"] != "fr"], open_synth),
            "test_fr": ([r for r in test if r["language"] == "fr"], open_synth),
            "real": (real_eval, lambda r: Image.open(real_root / r["image"]).convert("RGB")),
            **{k: (v, None) for k, v in load_text_sets(T).items()}}
    res = {}
    with torch.no_grad():
        for name, (rows, opener) in sets.items():
            rows = rows[:SMOKE_EVAL_ROWS] if smoke else rows
            out = evaluate_rows(model, rows, opener, device)
            (out_dir / f"eval_{name}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
            res[name] = {"n": len(out), "accuracy": round(sum(r["correct"] for r in out) / len(out), 4)}
            log(f"[{tag}] {name}: {res[name]}")
    sha = hashlib.sha256(best_path.read_bytes()).hexdigest()
    (out_dir / "summary.json").write_text(json.dumps(
        {"seed": seed, "steps_planned": steps, "stopped_at": stopped_at,
         "best_step": best["step"], "best_selection": [round(x, 4) for x in best["key"]],
         "best_pt_sha256": sha, "eval": res}, indent=2), encoding="utf-8")
    log(f"[{tag}] done: best step {best['step']}, {res}")


def spawn(seed: int, gpu: int, steps: int, smoke: bool, t0: float):
    tag = f"a1v_real_s{seed}" + ("_smoke" if smoke else "")
    (WORK / "runs" / tag).mkdir(parents=True, exist_ok=True)
    log_path = WORK / "runs" / tag / f"worker_gpu{gpu}.log"
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--seed", str(seed),
           "--steps", str(steps), "--t0", str(t0)] + (["--smoke"] if smoke else [])
    fh = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                            stdout=fh, stderr=subprocess.STDOUT, text=True)
    return proc, fh, log_path


def finish(spawned, label: str, timeout: float) -> str:
    proc, fh, log_path = spawned
    try:
        proc.wait(timeout=max(60.0, timeout))
        status = "completed" if proc.returncode == 0 else f"failed (exit {proc.returncode})"
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        status = "failed (wall budget)"
    fh.close()
    for line in log_path.read_text(encoding="utf-8").splitlines()[-60:]:
        print(f"[{label}] {line}", flush=True)
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.worker:
        run_worker(args.seed, args.steps, args.smoke, args.t0)
        return

    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    import e39_train as T

    T.pip_install()
    import torch

    if torch.cuda.device_count() < 2:
        raise SystemExit("E41 ABORT: needs the T4x2 session (2 GPUs)")
    summary = {"status": "running", "criteria": "docs/E41_CRITERIA_2026-10-22.md", "seeds": {}}

    def save():
        (WORK / "e41_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    smoke = finish(spawn(42, 0, SMOKE_STEPS, True, t0), "smoke", 3600)
    summary["smoke"] = smoke
    save()
    if smoke != "completed":
        summary["status"] = "failed (smoke)"
        save()
        raise SystemExit(f"E41 ABORT: smoke {smoke}")
    procs = {seed: spawn(seed, gpu, 2000, False, t0) for seed, gpu in SEEDS.items()}
    for seed, sp in procs.items():
        status = finish(sp, f"s{seed}", WALL_BUDGET_S - (time.time() - t0))
        sfile = WORK / "runs" / f"a1v_real_s{seed}" / "summary.json"
        summary["seeds"][seed] = {"status": status, **(json.loads(sfile.read_text(encoding="utf-8"))
                                                        if sfile.exists() else {})}
        save()
    summary["status"] = "completed"
    summary["wall_s"] = round(time.time() - t0, 1)
    save()
    log(f"E41 done in {summary['wall_s']}s: " + ", ".join(
        f"s{s}: {v['status']}" for s, v in summary["seeds"].items()))


if __name__ == "__main__":
    main()
