"""E37 C2: marker tokens on NeoMME's factorized embeddings.

Checks, in order:
  A. resize correctness - 32 markers added, word AND value tables grew
  B. gradients reach the new marker rows in BOTH tables
  C. marker hidden states differ across options
  D. 200-case overfit test (loss decreases, train accuracy >= 0.95 within 40
     epochs at image size 768)

MICROBATCH is passed via env MKSHV_MICROBATCH (from the C1 measurement at
768 px; default 8).
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SIZE = 768
OVERFIT_CASES = 200
OVERFIT_MAX_EPOCHS = 40
PASS_ACC = 0.95
SEED = 42


def log(msg: str) -> None:
    print(f"[c2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pip_install() -> None:
    pkgs = ["git+https://github.com/Gjusev/MK-SH-1@9e1a226e", "transformers==5.19.0", "pillow>=10.1"]
    wheel = sorted(Path("/kaggle/input").rglob("mksh_v-*.whl"))
    if wheel:
        pkgs.append(str(wheel[0]))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def build_cases(n, seed):
    from mksh_v.splitting import stratified_template_split
    from mksh_v.synth_doc_type import ALL_TEMPLATES, generate_doc_type_cases

    train_templates, _ = stratified_template_split(ALL_TEMPLATES)
    return generate_doc_type_cases(train_templates, n=n, seed=seed, split="train")


def main() -> None:
    pip_install()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("C2 ABORT: no CUDA device assigned")
    torch.manual_seed(SEED)          # head init + dropout reproducible
    torch.cuda.manual_seed_all(SEED)
    log(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    device = "cuda"

    from mksh_v.arch_a1v import NeoMMEMarkerScorer, forward_a1v

    model = NeoMMEMarkerScorer().to(device)

    # -- A: resize correctness ------------------------------------------------
    vocab_before = model.encoder.config.vocab_size
    model.ensure_markers(32)
    vocab_after = model.encoder.config.vocab_size
    backbone = getattr(model.encoder, model.encoder.base_model_prefix, model.encoder)
    word_rows = backbone.embed_tokens.word_embeddings.weight.shape[0]
    value_rows = backbone.value_embeddings.weight.shape[0]
    marker_ids = model.marker_ids()
    check_a = {
        "vocab_before": vocab_before,
        "vocab_after": vocab_after,
        "word_rows": word_rows,
        "value_rows": value_rows,
        "marker_ids_distinct": len(set(marker_ids)) == 32,
        "passed": vocab_after == word_rows == value_rows == vocab_before + 32
        and len(set(marker_ids)) == 32,
    }
    log(f"A resize: {check_a}")
    if not check_a["passed"]:
        (WORK / "c2_summary.json").write_text(
            json.dumps({"status": "failed", "stage": "A-resize", "detail": check_a}, indent=2),
            encoding="utf-8",
        )
        raise SystemExit("C2 FAILED at A (resize)")

    # -- B: gradients reach marker rows in both tables -------------------------
    rows = build_cases(4, seed=SEED)
    cases = [r["case"] for r in rows]
    images = [r["image"] for r in rows]
    model.train()
    batch = model.build_batch(model.encode_cases(cases), images, device, max_side=SIZE)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = forward_a1v(model, batch).float()
        loss = torch.nn.functional.cross_entropy(logits, batch["targets"], ignore_index=-100)
    loss.backward()
    word_grad = backbone.embed_tokens.word_embeddings.weight.grad[marker_ids]
    value_grad = backbone.value_embeddings.weight.grad[marker_ids]
    check_b = {
        "word_marker_grad_abs_max": float(word_grad.abs().max().item()),
        "value_marker_grad_abs_max": float(value_grad.abs().max().item()),
        "passed": float(word_grad.abs().max().item()) > 0.0
        and float(value_grad.abs().max().item()) > 0.0,
    }
    log(f"B gradients: {check_b}")
    model.zero_grad(set_to_none=True)

    # -- C: marker hidden states differ across options -------------------------
    model.eval()
    with torch.no_grad():
        one = model.build_batch(model.encode_cases(cases[:1]), images[:1], device, max_side=SIZE)
        out = model.encoder(
            input_ids=one["input_ids"], attention_mask=one["attention_mask"],
            position_ids=one["position_ids"], pixel_values=one["pixel_values"],
        )
        hidden = out.last_hidden_state[0]
        marker_hidden = hidden[one["marker_rows"][:, 1]]
        dists = torch.cdist(marker_hidden.float(), marker_hidden.float())
        off_diag = dists[~torch.eye(dists.shape[0], dtype=torch.bool, device=dists.device)]
        # any off-diagonal 0.0 means two markers collapsed to identical states -> fail
        min_pairwise = float(off_diag.min().item()) if off_diag.numel() else 0.0
    check_c = {
        "n_markers_in_case": int(one["marker_rows"].shape[0]),
        "min_pairwise_l2": min_pairwise,
        "passed": min_pairwise > 0.0,
    }
    log(f"C hidden distinctness: {check_c}")

    # -- D: 200-case overfit ---------------------------------------------------
    microbatch = int(os.environ.get("MKSHV_MICROBATCH", "8"))
    accumulation = 32 // microbatch
    rows = build_cases(OVERFIT_CASES, seed=SEED + 100)
    cases = [r["case"] for r in rows]
    images = [r["image"] for r in rows]
    encoded = model.encode_cases(cases)
    model.train()
    opt = torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(SEED)
    total_steps = OVERFIT_MAX_EPOCHS * ((len(cases) + 31) // 32)  # exact optimizer steps/epoch (E35 formula)
    warmup = max(int(total_steps * 0.05), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        pr = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.05, 0.95 * (1 - pr) + 0.05)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    curve = []
    passed_d = False
    for epoch in range(OVERFIT_MAX_EPOCHS):
        order = list(range(len(cases)))
        rng.shuffle(order)
        correct = seen = 0
        epoch_losses = []
        for cs in range(0, len(order), microbatch):
            idx = order[cs : cs + microbatch]
            batch = model.build_batch(
                [encoded[i] for i in idx], [images[i] for i in idx], device, max_side=SIZE
            )
            with torch.autocast("cuda", dtype=torch.float16):
                logits = forward_a1v(model, batch).float()
                loss = torch.nn.functional.cross_entropy(
                    logits, batch["targets"], ignore_index=-100
                ) / accumulation
            scaler.scale(loss).backward()
            epoch_losses.append(float(loss.item()) * accumulation)
            preds = logits.argmax(-1)
            correct += int((preds == batch["targets"]).sum().item())
            seen += len(idx)
            if (cs // microbatch + 1) % accumulation == 0 or cs + microbatch >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], 1.0
                )
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
        train_acc = correct / seen
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        curve.append({"epoch": epoch, "train_acc": round(train_acc, 4),
                      "mean_loss": round(mean_loss, 4)})
        log(f"D overfit epoch {epoch}: acc {train_acc:.4f} loss {mean_loss:.4f}")
        if train_acc >= PASS_ACC:
            break
    # frozen pass rule: accuracy target reached AND loss decreased overall
    if curve:
        passed_d = (
            any(c["train_acc"] >= PASS_ACC for c in curve)
            and curve[-1]["mean_loss"] < curve[0]["mean_loss"]
        )
    check_d = {
        "n_cases": OVERFIT_CASES,
        "microbatch": microbatch,
        "epochs_run": len(curve),
        "first_loss": curve[0]["mean_loss"] if curve else None,
        "last_loss": curve[-1]["mean_loss"] if curve else None,
        "best_acc": max((c["train_acc"] for c in curve), default=None),
        "passed": passed_d,
    }

    summary = {
        "status": "completed",
        "A_resize": check_a,
        "B_gradients": check_b,
        "C_hidden_distinct": check_c,
        "D_overfit": check_d,
        "c2_passed": check_a["passed"] and check_b["passed"] and check_c["passed"] and passed_d,
        "curve": curve,
    }
    (WORK / "c2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"C2 verdict: {summary['c2_passed']}")


if __name__ == "__main__":
    main()
