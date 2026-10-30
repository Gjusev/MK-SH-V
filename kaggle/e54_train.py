"""E54 kernel: aux digit heads on the E52 checkpoints (criteria
docs/E54_CRITERIA_2026-10-30.md).

The E44 training loop with three deltas: the model is NeoMMEAuxScorer
(train-time-only [DIGIT_1..8]+[LEN] heads, src/mksh_v/arch_a1v_aux.py), bind
encodings carry aux labels from each row's printed total, and the loss is
CE(choice) + 0.5 * CE(aux). Init from the pinned E52 checkpoints; pool v4,
weights, U1024, 800 steps and the frozen selection key are identical to E52.
This module carries its own spawn/main (E44's spawn re-executes e44_train.py,
which would run the non-aux worker).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time

import e39_train as T
import e41_train as E41
import e44_train as K
from e44_train import CONFIGS, DEV_EVERY, SEEDS, SMOKE_EVAL_ROWS, SMOKE_STEPS, STEPS, WORK, find_file_dir, load_bind, log, selection_key, slot_pool

AUX_W = 0.5  # frozen in the E54 criteria


def _sig_digits(total: float, idr: bool) -> str:
    """gen_e45.sig_digits inlined (scripts/ modules do not ship in the kernel)."""
    return str(round(total)) if idr else f"{total:.2f}".replace(".", "")


def aux_labels_of(row: dict) -> tuple[list[int] | None, int | None]:
    total = row.get("total")
    if total is None:
        return None, None
    idr = row.get("img_src") == "e41" or row.get("fmt") == "idr"
    d = _sig_digits(float(total), idr)
    return [int(c) for c in d][:8], len(str(int(float(total))))


def run_worker(seed: int, steps: int, smoke: bool, t0: float) -> None:
    exp = "e54"
    cfg = CONFIGS[exp]
    if not smoke:
        K._watchdog(cfg.get("wall_h", 5.2) * 3600 - (time.time() - t0) + 900, f"worker {exp} s{seed}")
    import torch
    from PIL import Image
    from mksh_v.arch_a1v_aux import NeoMMEAuxScorer, aux_loss, forward_a1v_aux
    from mksh_v.train_compare import case_of_row, fixed_dev_half, macro_family_accuracy

    device = "cuda"
    tag = f"a1v_{exp}_s{seed}" + ("_smoke" if smoke else "")
    side = cfg.get("side")
    max_side = side or T.IMAGE_SIZE
    deadline_s = cfg.get("train_deadline_h", 3.9) * 3600
    out_dir = WORK / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
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
    bind_root = find_file_dir(cfg["bind_slug"], "rows.jsonl")
    replay = [case_of_row(r) for r in T.load_replay_mix(T.find_bundle())]
    if smoke:
        dev_sel = dev_sel[:SMOKE_EVAL_ROWS]
        real_dev = real_dev[:SMOKE_EVAL_ROWS]
        bind_dev = bind_dev[:SMOKE_EVAL_ROWS]
    log(f"[{tag}] synth {len(synth)} real {len(real)} bind { {k: len(v) for k, v in bind.items()} } bind-dev {len(bind_dev)}")

    def open_synth(r):
        return T.open_image(r, roots)

    def open_real(r):
        return Image.open(rt_root / r["image"]).convert("RGB")

    def open_bind(r):
        if r["img_src"] is None:
            return None
        base = {"e38a": roots["E38a"], "bind": bind_root}.get(r["img_src"], rt_root)
        return Image.open(base / r["image"]).convert("RGB")

    ck_slug, ck_tag, ck_sha = cfg["init"][seed]
    run = find_file_dir(ck_slug, "best.pt", ck_tag)
    got = hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()
    if got != ck_sha:
        raise SystemExit(f"E54 ABORT: {ck_tag} sha {got}")
    model = NeoMMEAuxScorer().to(device)
    model.ensure_markers(32)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"], strict=False)
    model.ensure_aux()
    model.to(device)
    log(f"[{tag}] init from {ck_tag} (sha {got[:16]}) + aux heads")

    kw = dict(max_state_tokens=T.MAX_STATE_TOKENS, max_option_tokens=T.MAX_OPTION_TOKENS)

    def enc_rows(rows):
        cases = [r if hasattr(r, "options") else case_of_row(r) for r in rows]
        encs = model.encode_cases(cases, **kw)
        out = []
        for r, e in zip(rows, encs):
            d, L = aux_labels_of(r) if isinstance(r, dict) else (None, None)
            out.append(dict(e, aux_digits=d, aux_len=L) if d else e)
        return out

    pools = {"synthetic": (enc_rows(synth), synth, open_synth),
             "real": (enc_rows(real), real, open_real),
             "replay": (enc_rows(replay), None, None)}
    bind_enc = {k: [enc_rows(p) for p in v] for k, v in bind.items()}
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

    def take_bind(m: int):
        enc, rows = [], []
        k0 = rng.choices(kinds, weights)[0]
        for _ in range(m // 2):
            j = rng.randrange(len(bind[k0]))
            enc += bind_enc[k0][j]
            rows += bind[k0][j]
        return enc, rows

    def eval_aux(rows, encs) -> float:
        sel = [(r, e) for r, e in zip(rows, encs) if e.get("aux_digits")]
        if not sel:
            return -1.0
        good = tot = 0
        with torch.no_grad():
            for s in range(0, len(sel), 16):
                part = sel[s:s + 16]
                imgs = [open_bind(r) for r, _ in part]
                batch = model.build_batch([e for _, e in part], imgs, device,
                                          max_side=max_side, upscale_to=side)
                _, d_logits, _ = forward_a1v_aux(model, batch)
                pred = d_logits.argmax(-1)
                for (_, e), p in zip(part, pred.tolist()):
                    lab = e["aux_digits"]
                    for pos in range(min(4, len(lab))):
                        tot += 1
                        good += int(p[pos] == lab[pos])
        return good / tot if tot else -1.0

    def eval_dev():
        s_rows = E41.evaluate_rows(model, dev_sel, open_synth, device, side=side)
        r_rows = E41.evaluate_rows(model, real_dev, open_real, device, side=side)
        b_rows = E41.evaluate_rows(model, bind_dev, open_bind, device, side=side)
        aux_acc = eval_aux(bind_dev, enc_rows(bind_dev))
        return (macro_family_accuracy(s_rows), macro_family_accuracy(r_rows),
                sum(r["correct"] for r in b_rows) / len(b_rows),
                sum(r["correct"] for r in s_rows) / len(s_rows), aux_acc)

    micro = T.ARM_MICROBATCH["a1v"]
    accum = T.EFFECTIVE_BATCH // micro
    opt = torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best, curves, step_s, stopped_at = None, [], [], steps
    log(f"[{tag}] steps {steps} micro {micro} x accum {accum}, aux_w {AUX_W}")

    t_step = time.time()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        for slot in range(accum):
            pool = slot_pool(slot)
            aux_on = pool == "bind"
            if pool == "bind":
                encs, brows = take_bind(micro)
                imgs = [open_bind(r) for r in brows]
            else:
                enc, rows_, opener = pools[pool]
                idx = take(pool, micro)
                encs = [enc[i] for i in idx]
                imgs = [None] * len(idx) if opener is None else [opener(rows_[i]) for i in idx]
            batch = model.build_batch(encs, imgs, device, max_side=max_side, upscale_to=side)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, d_logits, l_logits = forward_a1v_aux(model, batch)
                loss = torch.nn.functional.cross_entropy(
                    logits.float(), batch["targets"], ignore_index=-100)
                if aux_on:
                    loss = loss + AUX_W * aux_loss(batch, d_logits.float(), l_logits.float())
                loss = loss / accum
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
                key = eval_dev()
            sel_key = selection_key(*key[:4])
            curves.append({"step": step, "synthetic_dev_half_macro": round(key[0], 4),
                           "real_dev_macro": round(key[1], 4), "bind_dev_acc": round(key[2], 4),
                           "aux_pos1_4_dev": round(key[4], 4)})
            log(f"[{tag}] step {step}/{steps} synth {key[0]:.4f} real {key[1]:.4f} bind {key[2]:.4f} aux {key[4]:.4f}")
            if best is None or sel_key >= best["key"]:
                best = {"key": sel_key, "step": step, "aux": key[4]}
                T.save_best_weights(best_path, model)
            model.train()
            if late:
                stopped_at = step
                log(f"[{tag}] train deadline at step {step}")
                break
        t_step = time.time()

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
    aux_final = eval_aux(bind_dev, enc_rows(bind_dev))
    sha = hashlib.sha256(best_path.read_bytes()).hexdigest()
    (out_dir / "summary.json").write_text(json.dumps(
        {"seed": seed, "init": ck_tag, "init_sha256": ck_sha, "steps_planned": steps,
         "stopped_at": stopped_at, "best_step": best["step"],
         "best_selection": [round(x, 4) for x in best["key"]],
         "aux_pos1_4_dev": round(best["aux"], 4), "aux_pos1_4_final": round(aux_final, 4),
         "best_pt_sha256": sha, "eval": res, "measure": measure}, indent=2), encoding="utf-8")
    log(f"[{tag}] done: best step {best['step']}, aux {aux_final:.4f}, {res}")


def spawn(seed: int, gpu: int, steps: int, smoke: bool, t0: float):
    tag = f"a1v_e54_s{seed}" + ("_smoke" if smoke else "")
    (WORK / "runs" / tag).mkdir(parents=True, exist_ok=True)
    log_path = WORK / "runs" / tag / f"worker_gpu{gpu}.log"
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--seed", str(seed),
           "--steps", str(steps), "--t0", str(t0)] + (["--smoke"] if smoke else [])
    fh = open(log_path, "a", encoding="utf-8")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               PYTORCH_CUDA_ALLOC_CONF=CONFIGS["e54"].get("alloc_conf", "expandable_segments:True"))
    return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True), fh, log_path


def main() -> None:
    import argparse

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
    K._watchdog(CONFIGS["e54"].get("wall_h", 5.2) * 3600 + 1800, "parent")
    print(f"[parent {time.strftime('%H:%M:%S')}] E54 start, heartbeat before pip", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    T.pip_install()
    print(f"[parent {time.strftime('%H:%M:%S')}] pip done", flush=True)
    import torch

    if torch.cuda.device_count() < 2:
        raise SystemExit("E54 ABORT: needs the T4x2 session (2 GPUs)")
    summary = {"status": "running", "criteria": CONFIGS["e54"]["criteria"], "seeds": {}}

    def save():
        (WORK / "e54_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[parent {time.strftime('%H:%M:%S')}] smoke next", flush=True)
    smoke = E41.finish(spawn(42, 0, SMOKE_STEPS, True, t0), "smoke", 3600)
    summary["smoke"] = smoke
    save()
    if smoke != "completed":
        summary["status"] = "failed (smoke)"
        save()
        raise SystemExit(f"E54 ABORT: smoke {smoke}")
    print(f"[parent {time.strftime('%H:%M:%S')}] workers next", flush=True)
    procs = {seed: spawn(seed, gpu, STEPS, False, t0) for seed, gpu in SEEDS.items()}
    for seed, sp in procs.items():
        wall_s = CONFIGS["e54"]["wall_h"] * 3600
        status = E41.finish(sp, f"s{seed}", wall_s - (time.time() - t0))
        sfile = WORK / "runs" / f"a1v_e54_s{seed}" / "summary.json"
        seeds = summary["seeds"].setdefault(str(seed), {})
        seeds["status"] = status
        if sfile.exists():
            s = json.loads(sfile.read_text(encoding="utf-8"))
            seeds.update({k: s.get(k) for k in ("best_step", "best_pt_sha256", "stopped_at", "aux_pos1_4_final")})
        save()
    summary["status"] = "completed" if all(
        v["status"] == "completed" for v in summary["seeds"].values()) else "partial"
    summary["wall_s"] = round(time.time() - t0, 1)
    save()
    log(f"E54 done: {summary['status']} in {summary['wall_s']}s")


if __name__ == "__main__":
    main()
