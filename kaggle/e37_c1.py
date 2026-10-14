"""E37 C1: hardware compatibility on ONE Kaggle T4 (the T4x2 shape is used,
only cuda:0). 100 optimizer steps of A1-V on synthetic doc_type data at image
longest side 768 / 1024 / 1536 px; microbatch found by measurement (start 8,
halve on OOM); records peak memory, seconds per step, loss finiteness.

Effective batch is pinned at 32 (microbatch x grad-accumulation = 32).
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SIZES = (768, 1024, 1536)
N_CASES = 512
STEPS = 100
EFFECTIVE_BATCH = 32
SEED = 42


def log(msg: str) -> None:
    print(f"[c1 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pip_install() -> None:
    pkgs = ["git+https://github.com/Gjusev/MK-SH-1@9e1a226e", "transformers==5.19.0", "pillow>=10.1"]
    wheel = sorted(Path("/kaggle/input").rglob("mksh_v-*.whl"))
    if wheel:
        pkgs.append(str(wheel[0]))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def build_cases():
    from mksh_v.splitting import stratified_template_split
    from mksh_v.synth_doc_type import ALL_TEMPLATES, generate_doc_type_cases

    train_templates, _ = stratified_template_split(ALL_TEMPLATES)
    rows = generate_doc_type_cases(train_templates, n=N_CASES, seed=SEED, split="train")
    log(f"data: {len(rows)} doc_type cases over {len(train_templates)} train templates")
    return rows


def run_size(size: int, rows) -> dict:
    import torch

    from mksh_v.arch_a1v import NeoMMEMarkerScorer, forward_a1v

    device = "cuda"
    model = NeoMMEMarkerScorer().to(device)
    model.ensure_markers(6)
    model.train()
    cases = [r["case"] for r in rows]
    images = [r["image"] for r in rows]
    encoded = model.encode_cases(cases)

    def microbatch_probe(m: int) -> bool:
        """Worst-case forward/backward on m copies of the LARGEST image.

        The 100-step loop pads every microbatch to its own longest sequence,
        so the honest probe is a batch of the biggest page, not the first m
        cases (page sizes differ per template: receipts 400x620 vs contracts
        794x1100; at 1536 px the big pages are not resized at all).
        """
        biggest = max(range(len(images)), key=lambda i: images[i].width * images[i].height)
        try:
            batch = model.build_batch([encoded[biggest]] * m, [images[biggest]] * m,
                                       device, max_side=size)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = forward_a1v(model, batch).float()
                loss = torch.nn.functional.cross_entropy(
                    logits, batch["targets"], ignore_index=-100
                )
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            return True
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False

    microbatch = 8
    while microbatch >= 1 and not microbatch_probe(microbatch):
        microbatch //= 2
        log(f"size {size}: OOM probe -> microbatch {microbatch or 'FAILED'}")
    if microbatch < 1:
        return {"size": size, "status": "failed", "reason": "OOM at microbatch 1"}

    accumulation = EFFECTIVE_BATCH // microbatch
    opt = torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(SEED)
    order = list(range(len(cases)))
    rng.shuffle(order)
    cursor = 0
    torch.cuda.reset_peak_memory_stats()
    step_times, losses, finite = [], [], True

    def next_chunk(m: int):
        nonlocal cursor
        if cursor + m > len(order):
            rng.shuffle(order)
            cursor = 0
        idx = order[cursor : cursor + m]
        cursor += m
        return idx

    t_start = time.time()
    step = 0
    while step < STEPS:
        torch.cuda.synchronize()
        t_step = time.time()
        opt.zero_grad(set_to_none=True)
        chunk_loss = 0.0
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
            chunk_loss += float(loss.item())  # loss already scaled by 1/accumulation
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], 1.0
        )
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        if step >= 10:  # skip warmup steps in the timing mean
            step_times.append(time.time() - t_step)
        step += 1
        if not math.isfinite(chunk_loss):
            finite = False
        losses.append(chunk_loss)
        if step % 10 == 0:
            log(f"size {size}: step {step}/{STEPS} loss {chunk_loss:.4f}")
    wall = time.time() - t_start
    result = {
        "size": size,
        "status": "completed",
        "microbatch": microbatch,
        "grad_accumulation": accumulation,
        "steps": STEPS,
        "sec_per_step_mean": round(sum(step_times) / len(step_times), 4) if step_times else None,
        "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "loss_finite_every_step": finite,
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "wall_s": round(wall, 1),
        "gpu": torch.cuda.get_device_name(0),
    }
    del model, opt
    torch.cuda.empty_cache()
    return result


def main() -> None:
    pip_install()
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("C1 ABORT: no CUDA device assigned")
    torch.manual_seed(SEED)          # head init + dropout reproducible
    torch.cuda.manual_seed_all(SEED)
    log(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB), torch {torch.__version__}")
    if "T4" not in torch.cuda.get_device_name(0):
        raise SystemExit(f"C1 ABORT: expected a T4, got {torch.cuda.get_device_name(0)}")

    rows = build_cases()
    results = []
    for size in SIZES:
        log(f"=== C1 size {size} ===")
        results.append(run_size(size, rows))
        (WORK / "c1_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    passed = [r for r in results if r["status"] == "completed" and r["loss_finite_every_step"]]
    log(f"C1 verdict: {len(passed)}/3 sizes ran 100 steps with finite loss "
        f"({[r['size'] for r in passed]})")


if __name__ == "__main__":
    main()
