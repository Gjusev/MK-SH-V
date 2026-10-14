"""E37 C3: text parity - NeoMME + marker head, TEXT ONLY, MK-SH-1 v1.1 recipe.

Faithful port of the E35 stage (mokka-decisions kaggle/typed_pilot.py,
stage_typed_e35_extremes) with the declared differences:

- backbone: pretrained NeoMME-260M + marker head (MK-SH-1's E30-final
  checkpoint cannot initialise a different architecture);
- sequence: A1-V tail (state <eos> question [Oi] option_i), state capped at
  700 tokens, option descriptions at 96 (E35 used julia_sequence's own
  1024/256 budgets);
- rows are consumed with their own options/state verbatim (E35 re-dumped
  noul states; whitespace-equivalent).

Everything else mirrors E35 seed 42: mix = 248 extremes + 944 standard grid
replay + 944 original task replay; 3 phases x 5 epochs; fresh AdamW
(2e-5/1e-4, wd 0.01) per phase; warmup 5%; clip 1.0; fp16; microbatch 8,
grad accumulation 4 (effective 32); shuffle rng seed^0xE35^phase. Evals at
t0 and after every phase on the frozen sets: known(testC) new(testD) mono e
(=testE with interp/extrap slices) task. One shot, no retries.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SEED = 42
MICROBATCH = 8
ACCUMULATION = 4
PHASES = 3
EPOCHS_PER_PHASE = 5
MAX_STATE_TOKENS = 700
MAX_OPTION_TOKENS = 96


def log(msg: str) -> None:
    print(f"[c3 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pip_install() -> None:
    pkgs = ["git+https://github.com/Gjusev/MK-SH-1@9e1a226e", "transformers==5.19.0", "pillow>=10.1"]
    wheel = sorted(Path("/kaggle/input").rglob("mksh_v-*.whl"))
    if wheel:
        pkgs.append(str(wheel[0]))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def find_bundle() -> Path:
    root = Path("/kaggle/input")
    for candidate in root.rglob("BUNDLE_MANIFEST.json"):
        return candidate.parent
    for candidate in root.rglob("e35-extremes/train_rows.jsonl"):
        return candidate.parent.parent
    raise SystemExit(f"mokka bundle not found under {root}: " + ", ".join(p.name for p in root.iterdir()))


def rows_from(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def rows_with_ids(path: Path, ids) -> list[dict]:
    want = set(ids)
    picked = [r for r in rows_from(path) if r["id"] in want]
    if len(picked) != len(want):
        raise SystemExit(f"id mismatch: {len(picked)}/{len(want)}")
    return picked


def case_of(row: dict):
    from mokka_decisions.contracts import DecisionCase, Option

    return DecisionCase(
        id=row["id"], group_id=row.get("group_id", row["id"]), source="e35",
        language=row.get("language", "en"), domain=row.get("domain", "task"),
        state=row["state"], question=row["question"],
        options=tuple(Option(id=o["id"], description=o["description"]) for o in row["options"]),
        target_kind="single", target_option=row["target_option"],
    )


def main() -> None:
    pip_install()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("C3 ABORT: no CUDA device assigned")
    log(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    device = "cuda"

    bundle = find_bundle()
    e26 = bundle / "e26-order"
    e27 = bundle / "e27-grid"
    e28 = bundle / "e28-ext"
    e29 = bundle / "e29-mix"
    e35 = bundle / "e35-extremes"

    ext_rows = rows_from(e35 / "train_rows.jsonl")
    grid_rows = rows_from(e28 / "train_rows.jsonl")
    man29 = json.loads((e29 / "manifest.json").read_text(encoding="utf-8"))
    task_rows = rows_with_ids(bundle / "data/processed/v1/instances/train.jsonl",
                              man29["original_train_ids"])
    dev_rows = rows_from(e26 / "dev_rows.jsonl")
    eval_specs = {
        "known": rows_from(e26 / "test_c_rows.jsonl"),
        "new": rows_from(e26 / "test_d_rows.jsonl"),
        "mono": rows_from(e26 / "mono_rows.jsonl"),
        "e": rows_from(e27 / "test_e_rows.jsonl"),
        "task": rows_with_ids(bundle / "data/processed/v1/eval/dev_eval.jsonl",
                              man29["original_dev_ids"]),
    }
    eval_cases = {k: [case_of(r) for r in v] for k, v in eval_specs.items()}
    # E-extrap slice (frozen reference 0.875): thr_range == "extrap" rows of testE
    eval_cases["e_extrap"] = [case_of(r) for r in eval_specs["e"] if r.get("thr_range") == "extrap"]
    all_mix = ext_rows + grid_rows + task_rows
    mix_cases = [case_of(r) for r in all_mix]
    log(f"mix: ext {len(ext_rows)} + grid {len(grid_rows)} + task {len(task_rows)} "
        f"= {len(mix_cases)}; dev curve {len(dev_rows)}; "
        f"evals { {k: len(v) for k, v in eval_cases.items()} }")

    from mksh_v.arch_a1v import NeoMMEMarkerScorer, evaluate_a1v, forward_a1v

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = NeoMMEMarkerScorer().to(device)
    model.ensure_markers(32)  # task rows have up to 8 options; 32 upfront (as C2)
    model.train()

    encoded = model.encode_cases(
        mix_cases, max_state_tokens=MAX_STATE_TOKENS, max_option_tokens=MAX_OPTION_TOKENS
    )
    dev_cases = [case_of(r) for r in dev_rows]

    def weights_sha() -> str:
        h = hashlib.sha256()
        for k in sorted(model.state_dict()):
            t = model.state_dict()[k].detach().to(torch.float32).cpu().contiguous()
            h.update(k.encode())
            h.update(t.numpy().tobytes())
        return h.hexdigest()

    def eval_all(tag: str, final_gate: bool = False) -> dict:
        """Per-phase monitoring on GPU fp32; the FINAL gate numbers on CPU
        float32 (E32 discipline: the bars come from a CPU fp32 harness)."""
        out = {"tag": tag, "eval_weights_sha256": weights_sha(), "sets": {}, "device": "cpu" if final_gate else "cuda"}
        eval_device = "cpu"
        if final_gate:
            model.to("cpu")
        else:
            model.to(device)
        model.eval()
        for name, cases in eval_cases.items():
            ev = evaluate_a1v(
                model, cases, None, device=eval_device if final_gate else device,
                batch_decisions=16,
                max_state_tokens=MAX_STATE_TOKENS, max_option_tokens=MAX_OPTION_TOKENS,
                collect_rows=True,
            )
            out["sets"][name] = {"n": ev["n"], "accuracy": round(ev["accuracy"], 4)}
            (WORK / f"eval_{name}__{tag}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in ev["rows"]), encoding="utf-8"
            )
            log(f"  {tag} {name}: {ev['accuracy']:.4f} ({ev['n']})")
        if final_gate:
            model.to(device)  # back for any further use
        else:
            model.train()
        return out

    history: list[dict] = []
    checkpoints = [eval_all("t0")]
    for phase in range(1, PHASES + 1):
        opt = torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4), weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda")
        rng = random.Random(SEED ^ 0xE35 ^ phase)
        total = max((len(encoded) + 31) // 32 * EPOCHS_PER_PHASE, 1)
        warmup = max(int(total * 0.05), 1)

        def lr_lambda(step, warmup=warmup, total=total):
            if step < warmup:
                return step / warmup
            pr = (step - warmup) / max(total - warmup, 1)
            return max(0.05, 0.95 * (1 - pr) + 0.05)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        for epoch in range(EPOCHS_PER_PHASE):
            order = list(range(len(encoded)))
            rng.shuffle(order)
            correct = seen = 0
            for cs in range(0, len(order), MICROBATCH):
                idx = order[cs : cs + MICROBATCH]
                batch = model.build_batch([encoded[i] for i in idx], [None] * len(idx), device)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = forward_a1v(model, batch).float()
                    loss = torch.nn.functional.cross_entropy(
                        logits, batch["targets"], ignore_index=-100
                    ) / ACCUMULATION
                scaler.scale(loss).backward()
                preds = logits.argmax(-1)
                correct += int((preds == batch["targets"]).sum().item())
                seen += len(idx)
                if (cs // MICROBATCH + 1) % ACCUMULATION == 0 or cs + MICROBATCH >= len(order):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in opt.param_groups for p in g["params"]], 1.0
                    )
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
            # dev curve on the E35 numeric dev (drift indicator, same as E35)
            model.eval()
            dev_ev = evaluate_a1v(model, dev_cases, None, device=device, batch_decisions=16,
                                  max_state_tokens=MAX_STATE_TOKENS,
                                  max_option_tokens=MAX_OPTION_TOKENS)
            model.train()
            history.append({"phase": phase, "epoch": epoch,
                            "train_acc": round(correct / seen, 4),
                            "dev_acc": round(dev_ev["accuracy"], 4)})
            log(f"ph{phase} ep{epoch}: train {correct / seen:.4f} dev {dev_ev['accuracy']:.4f}")
        checkpoints.append(eval_all(f"post_ph{phase}", final_gate=(phase == PHASES)))

    from safetensors.torch import save_file

    model.eval()
    save_file(dict(model.state_dict()), str(WORK / "c3_final.safetensors"), metadata={"format": "pt"})
    final = checkpoints[-1]["sets"]
    summary = {
        "status": "completed",
        "seed": SEED,
        "recipe": "E35 data mix + steps (3x5 phases, mb8 x acc4, AdamW 2e-5/1e-4, wd 0.01, "
                  "warmup 5%, clip 1.0, fp16), text-only A1-V tail",
        "checkpoints": checkpoints,
        "history": history,
        "final": final,
        "gate": {
            "testC_bar": 0.929,   # MK-SH-1 v1.1 0.979 - 5 pp
            "task_bar": 0.750,    # MK-SH-1 v1.1 0.800 - 5 pp
            "testC": final.get("known", {}).get("accuracy"),
            "task": final.get("task", {}).get("accuracy"),
            "e_extrap_reference": 0.875,
            "e_extrap": final.get("e_extrap", {}).get("accuracy"),
        },
    }
    summary["gate"]["within_5pp"] = (
        (summary["gate"]["testC"] or 0.0) >= 0.929
        and (summary["gate"]["task"] or 0.0) >= 0.750
    )
    (WORK / "c3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"C3 gate: {summary['gate']}")


if __name__ == "__main__":
    main()
