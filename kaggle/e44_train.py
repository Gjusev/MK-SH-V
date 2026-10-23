"""E44 kernel: fine-tune the two E41 checkpoints with the bind pool
(frozen criteria docs/E44_CRITERIA_2026-10-23.md).

Same orchestration as E41 (seeds 42/43 on GPUs 0/1, smoke first). Microbatch
slots of 8: % 4 == 3 replay, == 1 E41 real, == 2 bind (two whole pairs),
== 0 E39 synthetic. E41 helpers ship as the root module `e41_train`.
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
BIND_SLUG = "mkshv-e44-bind"
BIND_ROWS_SHA = "733b051288df5ff1ecc311b8270e6ed0c70fa9427460b8dc3007d9f4a0e9941f"
BIND_WEIGHTS = {"num_syn": 0.35, "num_cord": 0.20, "num_text": 0.10, "range": 0.05, "goal": 0.30}
E41_CKPTS = {42: ("a1v_real_s42", "e4b4f6f5a56fedae18886e378cd9fb8f84afb1a77536b168a2fdfb990d9641a5"),
             43: ("a1v_real_s43", "30f2eb429c5b25dca00e1ef53fd509232997c34799a5a22fa1c77200619c3819")}
SEEDS = {42: 0, 43: 1}
STEPS = 800
DEV_EVERY = 200
TRAIN_DEADLINE_S = int(2.5 * 3600)
WALL_BUDGET_S = int(3.2 * 3600)
SMOKE_STEPS = 20
SMOKE_EVAL_ROWS = 24


def log(msg: str) -> None:
    print(f"[e44 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_file_dir(slug: str, wanted: str, dirname: str | None = None) -> Path:
    inp = Path("/kaggle/input")
    for base in [inp / slug, *inp.glob(f"*/{slug}"), *inp.glob(f"*/*/{slug}")]:
        for dirpath, _dirs, files in os.walk(base, followlinks=True):
            if wanted in files and (dirname is None or Path(dirpath).name == dirname):
                return Path(dirpath)
    raise SystemExit(f"E44 ABORT: {slug} ({dirname or wanted}) not attached")


def slot_pool(slot: int) -> str:
    return {3: "replay", 1: "real", 2: "bind", 0: "synthetic"}[slot % 4]


def selection_key(synth_macro: float, real_macro: float, bind_acc: float, synth_acc: float) -> tuple:
    """Frozen: mean of synthetic dev-half macro, real-dev macro, bind-dev accuracy;
    tie -> higher synthetic dev-half accuracy."""
    return ((synth_macro + real_macro + bind_acc) / 3, synth_acc)


def load_bind() -> tuple[dict[str, list[list[dict]]], list[dict]]:
    """({kind: [pair rows]}, dev rows); rows sha-pinned."""
    p = find_file_dir(BIND_SLUG, "rows.jsonl") / "rows.jsonl"
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != BIND_ROWS_SHA:
        raise SystemExit(f"E44 ABORT: bind rows sha {got}")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    pairs: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        if r["split"] == "train":
            pairs.setdefault(r["bind"], {}).setdefault(r["pair_id"], []).append(r)
    out = {k: sorted(v.values(), key=lambda x: x[0]["pair_id"]) for k, v in pairs.items()}
    assert set(out) == set(BIND_WEIGHTS) and all(len(p) == 2 for v in out.values() for p in v)
    return out, [r for r in rows if r["split"] == "dev"]


def run_worker(seed: int, steps: int, smoke: bool, t0: float) -> None:
    import torch
    from mksh_v.arch_a1v import NeoMMEMarkerScorer, forward_a1v
    from mksh_v.train_compare import case_of_row, fixed_dev_half, macro_family_accuracy
    from PIL import Image

    import e39_train as T
    import e41_train as E41

    device = "cuda"
    tag = f"a1v_e44_s{seed}" + ("_smoke" if smoke else "")
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
    rt_root = E41.realtrain_root()
    real, real_dev = E41.load_realtrain(rt_root)
    bind, bind_dev = load_bind()
    replay = [case_of_row(r) for r in T.load_replay_mix(T.find_bundle())]
    if smoke:
        dev_sel, real_dev, bind_dev = dev_sel[:SMOKE_EVAL_ROWS], real_dev[:SMOKE_EVAL_ROWS], bind_dev[:SMOKE_EVAL_ROWS]
    log(f"[{tag}] synthetic {len(synth)}, real {len(real)}, bind pairs "
        f"{ {k: len(v) for k, v in bind.items()} }, bind-dev {len(bind_dev)}, replay {len(replay)}")

    def open_synth(r):
        return T.open_image(r, roots)

    def open_real(r):
        return Image.open(rt_root / r["image"]).convert("RGB")

    def open_bind(r):
        base = roots["E38a"] if r["img_src"] == "e38a" else rt_root
        return Image.open(base / r["image"]).convert("RGB")

    ck_tag, ck_sha = E41_CKPTS[seed]
    run = find_file_dir("mkshv-e41-runs", "best.pt", ck_tag)
    got = hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()
    if got != ck_sha:
        raise SystemExit(f"E44 ABORT: {ck_tag} sha {got}")
    model = NeoMMEMarkerScorer().to(device)
    model.ensure_markers(32)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"])
    model.to(device)
    log(f"[{tag}] init from {ck_tag} (sha {got[:16]})")

    kw = dict(max_state_tokens=T.MAX_STATE_TOKENS, max_option_tokens=T.MAX_OPTION_TOKENS)
    pools = {"synthetic": (model.encode_cases([case_of_row(r) for r in synth], **kw), synth, open_synth),
             "real": (model.encode_cases([case_of_row(r) for r in real], **kw), real, open_real),
             "replay": (model.encode_cases(replay, **kw), None, None)}
    bind_enc = {k: [model.encode_cases([case_of_row(r) for r in p], **kw) for p in v]
                for k, v in bind.items()}
    rng = random.Random(seed)
    orders = {k: list(range(len(v[0]))) for k, v in pools.items()}
    for o in orders.values():
        rng.shuffle(o)
    cursors = {k: 0 for k in pools}
    kinds, weights = list(BIND_WEIGHTS), list(BIND_WEIGHTS.values())

    def take(pool: str, m: int) -> list[int]:
        if cursors[pool] + m > len(orders[pool]):
            rng.shuffle(orders[pool])
            cursors[pool] = 0
        idx = orders[pool][cursors[pool]:cursors[pool] + m]
        cursors[pool] += m
        return idx

    def take_bind(m: int) -> tuple[list, list[dict]]:
        enc, rows = [], []
        for _ in range(m // 2):
            k = rng.choices(kinds, weights)[0]
            j = rng.randrange(len(bind[k]))
            enc += bind_enc[k][j]
            rows += bind[k][j]
        return enc, rows

    def eval_dev():
        s_rows = E41.evaluate_rows(model, dev_sel, open_synth, device)
        r_rows = E41.evaluate_rows(model, real_dev, open_real, device)
        b_rows = E41.evaluate_rows(model, bind_dev, open_bind, device)
        return (macro_family_accuracy(s_rows), macro_family_accuracy(r_rows),
                sum(r["correct"] for r in b_rows) / len(b_rows),
                sum(r["correct"] for r in s_rows) / len(s_rows))

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
            if pool == "bind":
                encs, brows = take_bind(micro)
                imgs = [open_bind(r) for r in brows]
            else:
                enc, rows, opener = pools[pool]
                idx = take(pool, micro)
                encs = [enc[i] for i in idx]
                imgs = [None] * len(idx) if opener is None else [opener(rows[i]) for i in idx]
            batch = model.build_batch(encs, imgs, device, max_side=T.IMAGE_SIZE)
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
        if step % DEV_EVERY == 0 or step == steps or late:
            with torch.no_grad():
                s_macro, r_macro, b_acc, s_acc = eval_dev()
            key = selection_key(s_macro, r_macro, b_acc, s_acc)
            curves.append({"step": step, "synthetic_dev_half_macro": round(s_macro, 4),
                           "real_dev_macro": round(r_macro, 4), "bind_dev_acc": round(b_acc, 4),
                           "synthetic_dev_half_acc": round(s_acc, 4)})
            log(f"[{tag}] step {step}/{steps} synth {s_macro:.4f} real {r_macro:.4f} bind {b_acc:.4f}")
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
    real_eval, real_root = E41.load_real_eval(T)
    sets = {"test_nonfr": ([r for r in test if r["language"] != "fr"], open_synth),
            "test_fr": ([r for r in test if r["language"] == "fr"], open_synth),
            "real": (real_eval, lambda r: Image.open(real_root / r["image"]).convert("RGB")),
            "bind_dev": (bind_dev, open_bind),
            **{k: (v, None) for k, v in E41.load_text_sets(T).items()}}
    res = {}
    with torch.no_grad():
        for name, (rows, opener) in sets.items():
            rows = rows[:SMOKE_EVAL_ROWS] if smoke else rows
            out = E41.evaluate_rows(model, rows, opener, device)
            (out_dir / f"eval_{name}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
            res[name] = {"n": len(out), "accuracy": round(sum(r["correct"] for r in out) / len(out), 4)}
            log(f"[{tag}] {name}: {res[name]}")
    sha = hashlib.sha256(best_path.read_bytes()).hexdigest()
    (out_dir / "summary.json").write_text(json.dumps(
        {"seed": seed, "init": ck_tag, "init_sha256": ck_sha, "steps_planned": steps,
         "stopped_at": stopped_at, "best_step": best["step"],
         "best_selection": [round(x, 4) for x in best["key"]],
         "best_pt_sha256": sha, "eval": res}, indent=2), encoding="utf-8")
    log(f"[{tag}] done: best step {best['step']}, {res}")


def spawn(seed: int, gpu: int, steps: int, smoke: bool, t0: float):
    tag = f"a1v_e44_s{seed}" + ("_smoke" if smoke else "")
    (WORK / "runs" / tag).mkdir(parents=True, exist_ok=True)
    log_path = WORK / "runs" / tag / f"worker_gpu{gpu}.log"
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--seed", str(seed),
           "--steps", str(steps), "--t0", str(t0)] + (["--smoke"] if smoke else [])
    fh = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                            stdout=fh, stderr=subprocess.STDOUT, text=True)
    return proc, fh, log_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.worker:
        run_worker(args.seed, args.steps, args.smoke, args.t0)
        return

    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    import e39_train as T
    import e41_train as E41

    T.pip_install()
    import torch

    if torch.cuda.device_count() < 2:
        raise SystemExit("E44 ABORT: needs the T4x2 session (2 GPUs)")
    summary = {"status": "running", "criteria": "docs/E44_CRITERIA_2026-10-23.md", "seeds": {}}

    def save():
        (WORK / "e44_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    smoke = E41.finish(spawn(42, 0, SMOKE_STEPS, True, t0), "smoke", 3600)
    summary["smoke"] = smoke
    save()
    if smoke != "completed":
        summary["status"] = "failed (smoke)"
        save()
        raise SystemExit(f"E44 ABORT: smoke {smoke}")
    procs = {seed: spawn(seed, gpu, STEPS, False, t0) for seed, gpu in SEEDS.items()}
    for seed, sp in procs.items():
        status = E41.finish(sp, f"s{seed}", WALL_BUDGET_S - (time.time() - t0))
        sfile = WORK / "runs" / f"a1v_e44_s{seed}" / "summary.json"
        summary["seeds"][seed] = {"status": status, **(json.loads(sfile.read_text(encoding="utf-8"))
                                                        if sfile.exists() else {})}
        save()
    summary["status"] = "completed"
    summary["wall_s"] = round(time.time() - t0, 1)
    save()
    log(f"E44 done in {summary['wall_s']}s: " + ", ".join(
        f"s{s}: {v['status']}" for s, v in summary["seeds"].items()))


if __name__ == "__main__":
    main()
