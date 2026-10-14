"""E37 C4: visual smoke test.

Trains A1-V on 2,004 synthetic doc_type cases (12 train templates, 2 per
class) for 500 optimizer steps and evaluates on 402 dev cases from 6 held-out
templates (1 per class). Passes when dev accuracy beats the majority-class
rate AND swapping images between dev cases drops accuracy back to ~majority
(if swapping does not hurt, the generator leaks the label through text and
must be fixed before E38).

Image size via env MKSHV_SIZE (default 1024; set from the C1 measurement).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

WORK = Path("/kaggle/working")
TRAIN_CASES = 2004
DEV_CASES = 402
STEPS = 500
EFFECTIVE_BATCH = 32
TRAIN_SEED = 42
DEV_SEED = 43
SWAP_MARGIN = 0.03


def log(msg: str) -> None:
    print(f"[c4 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pip_install() -> None:
    pkgs = ["git+https://github.com/Gjusev/MK-SH-1@9e1a226e", "transformers==5.19.0", "pillow>=10.1"]
    wheel = sorted(Path("/kaggle/input").rglob("mksh_v-*.whl"))
    if wheel:
        pkgs.append(str(wheel[0]))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def main() -> None:
    size = int(os.environ.get("MKSHV_SIZE", "1024"))
    microbatch = int(os.environ.get("MKSHV_MICROBATCH", "4"))
    accumulation = EFFECTIVE_BATCH // microbatch
    pip_install()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("C4 ABORT: no CUDA device assigned")
    torch.manual_seed(TRAIN_SEED)    # head init + dropout reproducible
    torch.cuda.manual_seed_all(TRAIN_SEED)
    log(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
        f"size {size}, microbatch {microbatch} x {accumulation}")
    device = "cuda"

    from mksh_v.arch_a1v import NeoMMEMarkerScorer, evaluate_a1v, forward_a1v
    from mksh_v.splitting import assert_no_leakage, stratified_template_split
    from mksh_v.synth_doc_type import ALL_TEMPLATES, generate_doc_type_cases, majority_rate

    train_templates, dev_templates = stratified_template_split(ALL_TEMPLATES)
    log(f"templates: train {train_templates} / dev {dev_templates}")
    train_rows = generate_doc_type_cases(train_templates, n=TRAIN_CASES, seed=TRAIN_SEED, split="train")
    dev_rows = generate_doc_type_cases(dev_templates, n=DEV_CASES, seed=DEV_SEED, split="dev")
    assert_no_leakage([r["case"] for r in train_rows + dev_rows])
    dev_cases = [r["case"] for r in dev_rows]
    dev_images = [r["image"] for r in dev_rows]
    majority = majority_rate(dev_cases)
    log(f"train {len(train_rows)} dev {len(dev_rows)} majority {majority:.4f}")

    model = NeoMMEMarkerScorer().to(device)
    model.ensure_markers(6)
    model.train()
    cases = [r["case"] for r in train_rows]
    images = [r["image"] for r in train_rows]
    encoded = model.encode_cases(cases)
    opt = torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    warmup = max(int(STEPS * 0.05), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        pr = (step - warmup) / max(STEPS - warmup, 1)
        return max(0.05, 0.95 * (1 - pr) + 0.05)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    rng = random.Random(TRAIN_SEED)
    order = list(range(len(cases)))
    rng.shuffle(order)
    cursor = 0

    def next_chunk(m):
        nonlocal cursor
        if cursor + m > len(order):
            rng.shuffle(order)
            cursor = 0
        idx = order[cursor : cursor + m]
        cursor += m
        return idx

    t0 = time.time()
    step = 0
    while step < STEPS:
        opt.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            idx = next_chunk(microbatch)
            batch = model.build_batch(
                [encoded[i] for i in idx], [images[i] for i in idx], device, max_side=size
            )
            with torch.autocast("cuda", dtype=torch.float16):
                logits = forward_a1v(model, batch).float()
                loss = torch.nn.functional.cross_entropy(
                    logits, batch["targets"], ignore_index=-100
                ) / accumulation
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if step % 50 == 0:
            log(f"step {step}/{STEPS} loss {float(loss.item()) * accumulation:.4f} "
                f"({time.time() - t0:.0f}s)")

    # -- eval: clean, then images swapped between dev cases --------------------
    clean = evaluate_a1v(model, dev_cases, dev_images, device=device, max_side=size,
                         batch_decisions=16, collect_rows=True)
    swapped_images = dev_images[1:] + dev_images[:1]  # every case gets a foreign image
    swapped = evaluate_a1v(model, dev_cases, swapped_images, device=device, max_side=size,
                           batch_decisions=16, collect_rows=True)
    per_class_pred = Counter(
        dev_cases[i].options[r["pred_idx"]].id
        for i, r in enumerate(clean["rows"])
    )
    per_class_gold = Counter(c.target_option for c in dev_cases)
    verdict = {
        "status": "completed",
        "size": size,
        "steps": STEPS,
        "majority_rate": round(majority, 4),
        "dev_accuracy": round(clean["accuracy"], 4),
        "dev_accuracy_n": clean["n"],
        "swap_accuracy": round(swapped["accuracy"], 4),
        "per_class_gold": dict(per_class_gold),
        "per_class_pred": dict(per_class_pred),
        "wall_s": round(time.time() - t0, 1),
    }
    verdict["passed"] = (
        clean["accuracy"] > majority
        and swapped["accuracy"] <= majority + SWAP_MARGIN
    )
    (WORK / "c4_summary.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    with open(WORK / "c4_dev_rows.jsonl", "w", encoding="utf-8") as fh:
        for row in clean["rows"]:
            fh.write(json.dumps(row) + "\n")
    with open(WORK / "c4_swap_rows.jsonl", "w", encoding="utf-8") as fh:
        for row in swapped["rows"]:
            fh.write(json.dumps(row) + "\n")
    log(f"C4 verdict: {verdict['passed']} dev {verdict['dev_accuracy']} vs majority "
        f"{verdict['majority_rate']}, swap {verdict['swap_accuracy']}")


if __name__ == "__main__":
    main()
