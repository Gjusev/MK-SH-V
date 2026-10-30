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
# per experiment: bind dataset, pinned rows, pool weights, init checkpoints (slug, tag, sha)
CONFIGS = {
    "e44": {"criteria": "docs/E44_CRITERIA_2026-10-23.md", "bind_slug": "mkshv-e44-bind",
            "bind_sha": "733b051288df5ff1ecc311b8270e6ed0c70fa9427460b8dc3007d9f4a0e9941f",
            "weights": {"num_syn": 0.35, "num_cord": 0.20, "num_text": 0.10, "range": 0.05, "goal": 0.30},
            "init": {42: ("mkshv-e41-runs", "a1v_real_s42",
                          "e4b4f6f5a56fedae18886e378cd9fb8f84afb1a77536b168a2fdfb990d9641a5"),
                     43: ("mkshv-e41-runs", "a1v_real_s43",
                          "30f2eb429c5b25dca00e1ef53fd509232997c34799a5a22fa1c77200619c3819")}},
    "e45": {"criteria": "docs/E45_CRITERIA_2026-10-24.md", "bind_slug": "mkshv-e45-bind",
            "bind_sha": "78c82e21d97f868dd38ff5a81f39453a9c521e715099dea46bbfdb6e01783aab",
            "weights": {"num_syn": 0.20, "num_cord": 0.10, "cmp_text": 0.15, "digits": 0.20,
                        "range": 0.05, "goal": 0.15, "goal_match": 0.15},
            "init": {42: ("mkshv-e44-train", "a1v_e44_s42",
                          "bb4f659f3773e59e60d2d724f6ac6ab19442e795a767b76754c5331b4deb49bf"),
                     43: ("mkshv-e44-train", "a1v_e44_s43",
                          "59b1cd820c96804a898fdbb22bf37c7c9b0ad6d945c175d2be82fc385ad67b78")}},
}
# E46 (docs/E46_CRITERIA_2026-10-25.md): E45 bind pool, new weights, init from E45,
# images upscaled to `side` before the processor; per-arm deadlines
_E46 = {"criteria": "docs/E46_CRITERIA_2026-10-25.md", "bind_slug": "mkshv-e45-bind",
        "bind_sha": CONFIGS["e45"]["bind_sha"],
        "weights": {"num_syn": 0.20, "num_cord": 0.10, "cmp_text": 0.05, "digits": 0.20,
                    "range": 0.05, "goal": 0.20, "goal_match": 0.20},
        "init": {42: ("mkshv-e45-train", "a1v_e45_s42",
                      "a360c96154bba5c9b9a038ba8a8fc43b13d0f6d3cfe77f958e32bab57f77fa77"),
                 43: ("mkshv-e45-train", "a1v_e45_s43",
                      "c729a142350d2b8cb0bff5856bc9906631b18e528c8b5165fd9b397205ac1999")}}
CONFIGS["e46a"] = dict(_E46, side=1024, train_deadline_h=2.5, wall_h=3.2)
CONFIGS["e46b"] = dict(_E46, side=1536, train_deadline_h=4.0, wall_h=4.8)
# E47 (docs/E47_CRITERIA_2026-10-25.md): bind pool v3 (digits at every position), init from E46 U1024
CONFIGS["e47"] = dict(CONFIGS["e46a"], criteria="docs/E47_CRITERIA_2026-10-25.md",
                      bind_slug="mkshv-e47-bind",
                      bind_sha="d3236aed192a4990b208729e95f49c4c4de0615b95c86e61b7f914dae09deac6",
                      init={42: ("mkshv-e46a-train", "a1v_e46a_s42",
                                 "724b81b80b3805211e07c3e5131898ffe42f39a52db5f574ef1909841deb9e13"),
                            43: ("mkshv-e46a-train", "a1v_e46a_s43",
                                 "169004453125d703a2e2c946158bb6f74f0be3a53b185ad6a6a5e903d24133b7")},
                      alloc_conf="expandable_segments:True")
# E49 (docs/E49_CRITERIA_2026-10-26.md): E47 rows + goal / goal_match rows of 2,000 new GUI
# episodes (images inside the bind dataset, img_src "bind"), init from the E47 checkpoints
CONFIGS["e49"] = dict(CONFIGS["e47"], criteria="docs/E49_CRITERIA_2026-10-26.md",
                      bind_slug="mkshv-e49-bind",
                      bind_sha="bc526c84d0b8007e23b7a9ab8b256d6b753f7dc46e2a720c96903d1862cef928",
                      init={42: ("mkshv-e47-train", "a1v_e47_s42",
                                 "8b166fb995069d0f80f95aec9fd759ef47a72fc0cde1338ca453b8adb6d19452"),
                            43: ("mkshv-e47-train", "a1v_e47_s43",
                                 "afcbeaa45a4c033b5e7d3f033b85fdda2642c0fef78397fcecc5bef53cd83c29")})
# E52 (docs/E52_CRITERIA_2026-10-29.md): pool v4 = E47 rows byte-identical + receipt-format
# family (own thermal renders; idr dot-thousands / comma decimals / eur control, 4-8-digit
# totals), G0 probe dropped the crop arm; init from the E47 checkpoints
CONFIGS["e52"] = dict(CONFIGS["e47"], criteria="docs/E52_CRITERIA_2026-10-29.md",
                      bind_slug="mkshv-e52-bind",
                      bind_sha="351b5505b4f5362866c1f116c30232a46b35b9e21e8c23a9c5d7c84555bb5704",
                      weights={"num_syn": 0.10, "num_cord": 0.10, "cmp_text": 0.05, "digits": 0.15,
                               "receipt": 0.25, "range": 0.05, "goal": 0.15, "goal_match": 0.15},
                      init={42: ("mkshv-e47-train", "a1v_e47_s42",
                                 "8b166fb995069d0f80f95aec9fd759ef47a72fc0cde1338ca453b8adb6d19452"),
                            43: ("mkshv-e47-train", "a1v_e47_s43",
                                 "afcbeaa45a4c033b5e7d3f033b85fdda2642c0fef78397fcecc5bef53cd83c29")})

# E54 (docs/E54_CRITERIA_2026-10-30.md): aux digit heads on the E52 checkpoints; own kernel
# module (kaggle/e54_train.py) carries the worker; this config only pins data + init + budget
CONFIGS["e54"] = dict(CONFIGS["e52"], criteria="docs/E54_CRITERIA_2026-10-30.md",
                      bind_slug="mkshv-e52-bind",
                      bind_sha="351b5505b4f5362866c1f116c30232a46b35b9e21e8c23a9c5d7c84555bb5704",
                      train_deadline_h=3.9, wall_h=5.2,
                      init={42: ("mkshv-e52-train", "a1v_e52_s42",
                                 "3d2ffdbcfe6eb9e851716079796cf9a2a04e1e0e90f30b9ffacecf66093c9aa3"),
                            43: ("mkshv-e52-train", "a1v_e52_s43",
                                 "dfc6d60e15247c9e4b5902f6e84cbca9c70eba8a5489a06a580b6b43330de37d")})

EXP = "e44"  # kernel default; the E45 kernel sets it before main()
SMOKE_ONLY: tuple[str, ...] = ()  # E46 smoke kernel: these arms, one GPU each, 20 steps, then stop
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


def load_bind(exp: str = "e44") -> tuple[dict[str, list[list[dict]]], list[dict]]:
    """({kind: [pair rows]}, dev rows); rows sha-pinned."""
    cfg = CONFIGS[exp]
    p = find_file_dir(cfg["bind_slug"], "rows.jsonl") / "rows.jsonl"
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != cfg["bind_sha"]:
        raise SystemExit(f"E44 ABORT: bind rows sha {got}")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    missing = [r["image"] for r in rows if r.get("img_src") == "bind" and not (p.parent / r["image"]).is_file()]
    if missing:  # E49: images ship inside the bind dataset; fail before any GPU work
        raise SystemExit(f"E44 ABORT: {len(missing)} bind images missing, first {missing[0]}")
    pairs: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        if r["split"] == "train":
            pairs.setdefault(r["bind"], {}).setdefault(r["pair_id"], []).append(r)
    out = {k: sorted(v.values(), key=lambda x: x[0]["pair_id"]) for k, v in pairs.items()}
    assert set(out) == set(cfg["weights"]) and all(len(p) == 2 for v in out.values() for p in v)
    return out, [r for r in rows if r["split"] == "dev"]


def _watchdog(seconds: float, why: str) -> None:
    """E49 incident (5h35m hung run): hard-exit after `seconds` no matter where the
    process is (a hung pip/dataloader/orphaned worker must never outlive its budget)."""
    import os
    import threading

    def die():
        import time as _t

        _t.sleep(max(60.0, seconds))
        print(f"[watchdog] {why}: exiting after {seconds:.0f}s", flush=True)
        os._exit(0)

    threading.Thread(target=die, daemon=True).start()


def run_worker(seed: int, steps: int, smoke: bool, t0: float, exp: str = "e44") -> None:
    import time as _t

    cfg0 = CONFIGS[exp]
    if not smoke:
        _watchdog(cfg0.get("wall_h", WALL_BUDGET_S / 3600) * 3600 - (_t.time() - t0) + 900,
                   f"worker {exp} s{seed}")
    import torch
    from mksh_v.arch_a1v import NeoMMEMarkerScorer, forward_a1v
    from mksh_v.train_compare import case_of_row, fixed_dev_half, macro_family_accuracy
    from PIL import Image

    import e39_train as T
    import e41_train as E41

    device = "cuda"
    cfg = CONFIGS[exp]
    tag = f"a1v_{exp}_s{seed}" + ("_smoke" if smoke else "")
    side = cfg.get("side")  # E46 input rule; None = the E41..E45 pipeline
    max_side = side or T.IMAGE_SIZE
    deadline_s = cfg.get("train_deadline_h", TRAIN_DEADLINE_S / 3600) * 3600
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
    bind, bind_dev = load_bind(exp)
    bind_root = find_file_dir(cfg["bind_slug"], "rows.jsonl")  # E49 images (img_src "bind")
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
        if r["img_src"] is None:  # text-only row (e45 cmp_text)
            return None
        base = {"e38a": roots["E38a"], "bind": bind_root}.get(r["img_src"], rt_root)
        return Image.open(base / r["image"]).convert("RGB")

    ck_slug, ck_tag, ck_sha = cfg["init"][seed]
    run = find_file_dir(ck_slug, "best.pt", ck_tag)
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
    kinds, weights = list(cfg["weights"]), list(cfg["weights"].values())

    def take(pool: str, m: int) -> list[int]:
        if cursors[pool] + m > len(orders[pool]):
            rng.shuffle(orders[pool])
            cursors[pool] = 0
        idx = orders[pool][cursors[pool]:cursors[pool] + m]
        cursors[pool] += m
        return idx

    def take_bind(m: int) -> tuple[list, list[dict]]:
        enc, rows = [], []
        k0 = rng.choices(kinds, weights)[0]
        for _ in range(m // 2):
            # e44 drew the kind per pair (as run); e45 keeps one kind per microbatch
            k = rng.choices(kinds, weights)[0] if exp == "e44" else k0
            j = rng.randrange(len(bind[k]))
            enc += bind_enc[k][j]
            rows += bind[k][j]
        return enc, rows

    def eval_dev():
        s_rows = E41.evaluate_rows(model, dev_sel, open_synth, device, side=side)
        r_rows = E41.evaluate_rows(model, real_dev, open_real, device, side=side)
        b_rows = E41.evaluate_rows(model, bind_dev, open_bind, device, side=side)
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
    torch.cuda.reset_peak_memory_stats()
    step_s = []
    for step in range(1, steps + 1):
        t_step = time.time()
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
            batch = model.build_batch(encs, imgs, device, max_side=max_side, upscale_to=side)
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
        step_s.append(time.time() - t_step)
        late = time.time() - t0 > deadline_s
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
    gb = 1024 ** 3
    measure = {"side": side, "max_side": max_side,
               "peak_mem_alloc_gb": round(torch.cuda.max_memory_allocated() / gb, 2),
               "peak_mem_reserved_gb": round(torch.cuda.max_memory_reserved() / gb, 2),
               "gpu_mem_total_gb": round(torch.cuda.get_device_properties(0).total_memory / gb, 2),
               "first_step_s": round(step_s[0], 2) if step_s else None,
               "s_per_step": round(sum(step_s[1:]) / max(1, len(step_s) - 1), 2) if step_s else None}
    log(f"[{tag}] measure {measure}")
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
            out = E41.evaluate_rows(model, rows, opener, device, side=side)
            (out_dir / f"eval_{name}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
            res[name] = {"n": len(out), "accuracy": round(sum(r["correct"] for r in out) / len(out), 4)}
            log(f"[{tag}] {name}: {res[name]}")
    sha = hashlib.sha256(best_path.read_bytes()).hexdigest()
    (out_dir / "summary.json").write_text(json.dumps(
        {"seed": seed, "init": ck_tag, "init_sha256": ck_sha, "steps_planned": steps,
         "stopped_at": stopped_at, "best_step": best["step"],
         "best_selection": [round(x, 4) for x in best["key"]],
         "best_pt_sha256": sha, "eval": res, "measure": measure}, indent=2), encoding="utf-8")
    log(f"[{tag}] done: best step {best['step']}, {res}")


def spawn(seed: int, gpu: int, steps: int, smoke: bool, t0: float, exp: str | None = None):
    exp = exp or EXP
    tag = f"a1v_{exp}_s{seed}" + ("_smoke" if smoke else "")
    (WORK / "runs" / tag).mkdir(parents=True, exist_ok=True)
    log_path = WORK / "runs" / tag / f"worker_gpu{gpu}.log"
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--seed", str(seed),
           "--steps", str(steps), "--t0", str(t0), "--exp", exp] + (["--smoke"] if smoke else [])
    fh = open(log_path, "a", encoding="utf-8")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    if CONFIGS[exp].get("alloc_conf"):  # allocation only (E47); computation unchanged
        env["PYTORCH_CUDA_ALLOC_CONF"] = CONFIGS[exp]["alloc_conf"]
    proc = subprocess.Popen(cmd, env=env,
                            stdout=fh, stderr=subprocess.STDOUT, text=True)
    return proc, fh, log_path


def smoke_only(t0: float) -> None:
    """E46: smoke each arm in SMOKE_ONLY on its own GPU (seed 42), record memory and
    s/step, never start a full run. An arm that fails (e.g. out of memory) is reported
    with its log tail; the kernel still exits cleanly."""
    import e41_train as E41

    out = {"status": "running", "criteria": CONFIGS[SMOKE_ONLY[0]]["criteria"], "arms": {}}
    procs = {exp: spawn(42, gpu, SMOKE_STEPS, True, t0, exp) for gpu, exp in enumerate(SMOKE_ONLY)}
    for exp, sp in procs.items():
        status = E41.finish(sp, f"smoke {exp}", 3600)
        run = WORK / "runs" / f"a1v_{exp}_s42_smoke"
        sfile = run / "summary.json"
        arm = {"status": status, "side": CONFIGS[exp].get("side"),
               "train_deadline_h": CONFIGS[exp].get("train_deadline_h")}
        if sfile.exists():
            arm["measure"] = json.loads(sfile.read_text(encoding="utf-8")).get("measure")
            sps = (arm["measure"] or {}).get("s_per_step")
            if sps:
                arm["projected_train_h"] = round(sps * STEPS * 1.15 / 3600, 2)
        logs = sorted(run.glob("worker_gpu*.log"))
        if logs:
            arm["log_tail"] = logs[0].read_text(encoding="utf-8", errors="replace")[-1500:]
        out["arms"][exp] = arm
        (WORK / "e46_smoke_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["status"] = "completed"
    out["wall_s"] = round(time.time() - t0, 1)
    (WORK / "e46_smoke_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"E46 smoke done: { {k: (v['status'], v.get('measure')) for k, v in out['arms'].items()} }")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exp", default=EXP, choices=sorted(CONFIGS))
    args = ap.parse_args()
    if args.worker:
        run_worker(args.seed, args.steps, args.smoke, args.t0, args.exp)
        return

    t0 = time.time()
    _watchdog(CONFIGS[EXP].get("wall_h", WALL_BUDGET_S / 3600) * 3600 + 1800, "parent")
    print(f"[parent {time.strftime('%H:%M:%S')}] E44 start, exp={EXP}, heartbeat before pip", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    import e39_train as T
    import e41_train as E41

    T.pip_install()
    print(f"[parent {time.strftime('%H:%M:%S')}] pip done", flush=True)
    import torch

    if torch.cuda.device_count() < 2:
        raise SystemExit("E44 ABORT: needs the T4x2 session (2 GPUs)")
    if SMOKE_ONLY:
        smoke_only(t0)
        return
    summary = {"status": "running", "criteria": CONFIGS[EXP]["criteria"], "seeds": {}}

    def save():
        (WORK / f"{EXP}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[parent {time.strftime('%H:%M:%S')}] smoke next", flush=True)
    smoke = E41.finish(spawn(42, 0, SMOKE_STEPS, True, t0), "smoke", 3600)
    summary["smoke"] = smoke
    save()
    if smoke != "completed":
        summary["status"] = "failed (smoke)"
        save()
        raise SystemExit(f"E44 ABORT: smoke {smoke}")
    print(f"[parent {time.strftime('%H:%M:%S')}] workers next", flush=True)
    procs = {seed: spawn(seed, gpu, STEPS, False, t0) for seed, gpu in SEEDS.items()}
    for seed, sp in procs.items():
        wall_s = CONFIGS[EXP].get("wall_h", WALL_BUDGET_S / 3600) * 3600
        status = E41.finish(sp, f"s{seed}", wall_s - (time.time() - t0))
        sfile = WORK / "runs" / f"a1v_{EXP}_s{seed}" / "summary.json"
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
