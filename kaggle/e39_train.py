"""E39: compared training, synthetic-only (criteria: docs/E39_CRITERIA_*.md at
freeze). ONE GPU orchestrator + self-spawning per-GPU workers. NOT LAUNCHED
until the criteria are frozen and the user gives GO.

Arms (seed 42 head-to-head; the finalist also runs seeds 43/44):
  a1v   NeoMME + marker head, init = E37 C3 checkpoint, 1024 px, mb 4 x 8
  ocr   precomputed Tesseract text (gjusev/mkshv-e39-ocr) -> MK-SH-1, mb 8 x 4
  smol  SmolVLM2-500M, first-token option-letter logits, mb 4 x 8 (smoke-gated)

Every arm mixes E35 text replay: 1 of every 4 microbatches, steps unchanged.
Selection is dev-only (frozen half, best checkpoint restored); test opens
once at the end. G2 anchors to frozen MK-SH-1 v1.1 (testC 0.959 / task 0.780
after -2 pp), not to a retrained arm. Quota bills the SESSION wall at double
rate: phase 1 runs a1v on gpu0 while ocr+smol run on gpu1.

Expected attachments: gjusev/mkshv-e37-ckpt (c3_final.safetensors,
mksh1_v12.safetensors), gjusev/mokka-decisions-bundle (E35 replay + G2),
gjusev/mkshv-e39-ocr (ocr.jsonl). Secret HF_TOKEN checked FIRST, per
notebook; never printed, logged or written to files.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
E38_REPOS = {"E38a": "Gjusev/MK-SH-V-E38a", "E38b": "Gjusev/MK-SH-V-E38b"}
REAL_EVAL_REPO = "Gjusev/MK-SH-V-E38c"  # provisional; skipped cleanly if absent
SMOL_ID = "HuggingFaceTB/SmolVLM2-500M-Video"  # exact id pinned at freeze
STEPS = 2000
SMOKE_STEPS = 20
EFFECTIVE_BATCH = 32
ARM_MICROBATCH = {"a1v": 4, "ocr": 8, "smol": 4}
IMAGE_SIZE = 1024
MAX_STATE_TOKENS = 700
MAX_OPTION_TOKENS = 96
DEV_EVERY = 500
DEV_HALF_SEED = 39
TEXT_BARS = {"testC": 0.959, "task": 0.780}  # frozen MK-SH-1 v1.1 - 2 pp
DELTA_LO_MARGIN = 0.02
SWAP_DROP_BAR = 0.15
REAL_EVAL_KILL = 0.95
REPLAY_EVERY = 4  # every 4th microbatch is E35 text replay
CKPT_SHA_PREFIXES = {
    "c3_final.safetensors": "bac7c72242d236f8",
    "mksh1_v12.safetensors": "ea66cdde",
}
LETTERS = "ABCDEFGH"


def log(msg: str) -> None:
    print(f"[e39 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------------
def check_secret_and_repos() -> None:
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("HF_TOKEN")  # never printed/logged
    if not token:
        raise SystemExit("E39 ABORT: HF_TOKEN secret missing on this notebook "
                         "(Add-ons -> Secrets). No quota spent.")
    os.environ["HF_TOKEN"] = token
    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    for repo in E38_REPOS.values():
        if not any(f.endswith("rows.jsonl")
                   for f in api.list_repo_files(repo, repo_type="dataset")):
            raise SystemExit(f"E39 ABORT: {repo} readable but has no rows.jsonl")
    log(f"HF auth OK (user {who.get('name', '?')}); E38 repos readable")


def pip_install() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/Gjusev/MK-SH-1@9e1a226e",
                    "transformers==5.19.0", "pillow>=10.1",
                    "huggingface_hub>=0.34"], check=True)


def find_bundle() -> Path:
    for cand in Path("/kaggle/input").rglob("BUNDLE_MANIFEST.json"):
        return cand.parent
    raise SystemExit("E39 ABORT: gjusev/mokka-decisions-bundle not attached "
                     "(E35 replay mix + G2 text sets)")


def find_ocr_cache() -> Path:
    for cand in Path("/kaggle/input").rglob("ocr.jsonl"):
        return cand
    raise SystemExit("E39 ABORT: ocr.jsonl not attached "
                     "(expected from the gjusev/mkshv-e39-ocr dataset)")


def expected_checkpoint(name: str) -> Path | None:
    for cand in sorted(Path("/kaggle/input").rglob(name)):
        got = hashlib.sha256(cand.read_bytes()).hexdigest()
        if got.startswith(CKPT_SHA_PREFIXES[name]):
            return cand
        log(f"sha mismatch for {name} at {cand}: {got[:16]}...")
    return None


def download_data() -> dict:
    from huggingface_hub import snapshot_download

    roots = {}
    for key, repo in E38_REPOS.items():
        t0 = time.time()
        roots[key] = Path(snapshot_download(
            repo_id=repo, repo_type="dataset",
            allow_patterns=["manifest.json", "*/rows.jsonl", "*/train/*.png",
                            "*/dev/*.png", "*/test/*.png"]))
        log(f"downloaded {repo} ({time.time() - t0:.0f}s)")
    return roots


def verify_rows_sha(roots: dict) -> None:
    for key, root in roots.items():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for family, spec in manifest["families"].items():
            got = hashlib.sha256((root / spec["rows"]).read_bytes()).hexdigest()
            if got != spec["rows_sha256"]:
                raise SystemExit(f"E39 ABORT: rows sha mismatch {key}/{family}: {got}")
    log("rows.jsonl sha256 verified against both repo manifests")


def image_key(row: dict, roots: dict) -> str:
    repo = "E38a" if row["id"].startswith("e38a") else "E38b"
    return f"{E38_REPOS[repo]}|{row['image']}"


def image_path_of(row: dict, roots: dict) -> Path:
    repo = "E38a" if row["id"].startswith("e38a") else "E38b"
    return roots[repo] / row["image"]


def open_image(row: dict, roots: dict):
    from PIL import Image

    return Image.open(image_path_of(row, roots)).convert("RGB")


def load_ocr_cache() -> dict[str, str]:
    cache = {}
    for line in find_ocr_cache().read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            cache[r["key"]] = r["text"]
    log(f"OCR cache loaded: {len(cache)} entries")
    return cache


def load_replay_mix(bundle: Path) -> list[dict]:
    """E35 text mix (the C3 mix): extremes 248 + grid 944 + task 944."""
    rows = []
    for rel in ("e35-extremes/train_rows.jsonl", "e28-ext/train_rows.jsonl"):
        rows += [json.loads(l) for l in (bundle / rel).read_text(encoding="utf-8")
                 .splitlines() if l.strip()]
    man = json.loads((bundle / "e29-mix/manifest.json").read_text(encoding="utf-8"))
    want = set(man["original_train_ids"])
    rows += [json.loads(l) for l in (bundle / "data/processed/v1/instances/train.jsonl")
             .read_text(encoding="utf-8").splitlines()
             if l.strip() and json.loads(l)["id"] in want]
    return rows


def load_e38_rows(roots: dict) -> dict:
    from mksh_v.train_compare import (E38A_FAMILIES, E38B_FAMILIES,
                                      load_e38_rows, rows_by_split)

    rows = (load_e38_rows(roots["E38a"], families=E38A_FAMILIES)
            + load_e38_rows(roots["E38b"], families=E38B_FAMILIES))
    return {sp: rows_by_split(rows, sp, drop_fr=False)
            for sp in ("train", "dev", "test")}


def lr_lambda_of(total_steps: int):
    warmup = max(int(total_steps * 0.05), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        pr = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.05, 0.95 * (1 - pr) + 0.05)

    return lr_lambda


def save_full_checkpoint(path: Path, model, opt, sched, scaler, step, cursor, rng_state) -> None:
    import torch

    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "data_cursor": cursor, "rng_python": rng_state,
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all()}, path)


def save_best_weights(path: Path, model) -> None:
    import torch

    torch.save({"model": model.state_dict()}, path)


def load_best_weights(path: Path, model) -> None:
    import torch

    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])


# ---------------------------------------------------------------------------
# arm: a1v
# ---------------------------------------------------------------------------
def build_a1v(device):
    from mksh_v.arch_a1v import NeoMMEMarkerScorer

    model = NeoMMEMarkerScorer().to(device)
    model.ensure_markers(32)  # the C3 checkpoint's tables are 32-marker wide
    ckpt = expected_checkpoint("c3_final.safetensors")
    if ckpt:
        from safetensors.torch import load_file

        model.load_state_dict(load_file(str(ckpt)), strict=True)
        log(f"A1-V init: E37 C3 checkpoint (sha verified) {ckpt}")
    else:
        log("A1-V init: c3_final absent -> C3-rerun fallback (pre-declared)")
        c3_rerun(model, device)
    return model


def c3_rerun(model, device) -> None:
    import torch

    from mksh_v.arch_a1v import forward_a1v
    from mksh_v.train_compare import case_of_row

    bundle = find_bundle()
    cases = [case_of_row(r) for r in load_replay_mix(bundle)]
    encoded = model.encode_cases(cases, max_state_tokens=MAX_STATE_TOKENS,
                                 max_option_tokens=MAX_OPTION_TOKENS)
    model.train()
    for phase in (1, 2, 3):
        opt = torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4), weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda")
        rng = random.Random(42 ^ 0xE35 ^ phase)
        total = 5 * ((len(cases) + 31) // 32)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_of(total))
        order, cursor = list(range(len(cases))), 0
        rng.shuffle(order)
        for _ in range(total):
            opt.zero_grad(set_to_none=True)
            for _ in range(4):
                if cursor + 8 > len(order):
                    rng.shuffle(order)
                    cursor = 0
                idx = order[cursor:cursor + 8]
                cursor += 8
                batch = model.build_batch([encoded[i] for i in idx], [None] * len(idx), device)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = forward_a1v(model, batch).float()
                    loss = torch.nn.functional.cross_entropy(
                        logits, batch["targets"], ignore_index=-100) / 4
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        t = model.state_dict()[k].detach().to(torch.float32).cpu().contiguous()
        h.update(k.encode())
        h.update(t.numpy().tobytes())
    (WORK / "e39_c3_rerun_sha.txt").write_text(h.hexdigest(), encoding="utf-8")
    log(f"C3 rerun done; weights sha256 {h.hexdigest()[:16]} (declared divergence)")


# ---------------------------------------------------------------------------
# arm: smol (SmolVLM2 first-token letter scoring; smoke-gated)
# ---------------------------------------------------------------------------
class SmolScorer:
    """Image + options prompt; scores options by first-token letter logits."""

    def __init__(self, device, model_id: str = SMOL_ID):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)
        self.model.float()
        tok = self.processor.tokenizer
        self.letter_ids = [tok(c, add_special_tokens=False)["input_ids"][0]
                           for c in LETTERS]

    def prompt_of(self, case, k: int) -> str:
        opts = "\n".join(f"{LETTERS[i]}) {case.options[i].description}"
                         for i in range(k))
        return (f"{case.state}\nQuestion: {case.question}\nOptions:\n{opts}\n"
                "Answer with the letter only. Answer:")

    def batch_logits(self, cases, images):
        """(B, len(LETTERS)) letter logits at the answer position (fp32)."""
        import torch

        texts = [self.prompt_of(c, len(c.options)) for c in cases]
        if images and images[0] is not None:
            feats = self.processor(text=texts, images=list(images),
                                   padding=True, return_tensors="pt").to(self.device)
        else:
            feats = self.processor(text=texts, padding=True,
                                   return_tensors="pt").to(self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = self.model(**feats)
        last = out.logits[:, -1, :].float()  # (B, vocab)
        ids = torch.tensor(self.letter_ids, device=last.device)
        return last[:, ids]

    def parameters(self):
        return self.model.parameters()


# ---------------------------------------------------------------------------
# worker: trains ONE arm+seed on its GPU
# ---------------------------------------------------------------------------
def ocr_case_of(row: dict, roots: dict | None, ocr: dict):
    from mksh_v.train_compare import case_of_row

    c = case_of_row(row)
    if roots is None or not row.get("image"):
        return c
    text = ocr.get(image_key(row, roots))
    return dataclasses.replace(c, state=text) if text is not None else c


def run_worker(args) -> None:
    """--arm --seed --gpu --steps; writes runs/<arm>_s<seed>/{best.pt,...}."""
    device = "cuda"
    out_dir = WORK / "runs" / f"{args.arm}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_seed = args.seed
    import torch

    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    random.seed(torch_seed)

    roots = download_data()
    verify_rows_sha(roots)
    splits = load_e38_rows(roots)
    bundle = find_bundle()
    ocr = load_ocr_cache() if args.arm in ("ocr",) else {}
    from mksh_v.train_compare import case_of_row, fixed_dev_half, macro_family_accuracy, qtype_of_row

    dev_half = fixed_dev_half(splits["dev"], seed=DEV_HALF_SEED, fraction=0.5)
    dev_sel = [r for r in splits["dev"] if r["id"] in dev_half]
    replay_rows = load_replay_mix(bundle)
    replay_cases = [case_of_row(r) for r in replay_rows]
    replay_qt = {c.id: qtype_of_row(r) for c, r in zip(replay_cases, replay_rows)}
    train_rows = splits["train"]

    micro = ARM_MICROBATCH[args.arm]
    accum = EFFECTIVE_BATCH // micro

    # ---- build the arm -------------------------------------------------------
    if args.arm == "a1v":
        model = build_a1v(device)  # ensure_markers(32) already done inside
        train_cases = [case_of_row(r) for r in train_rows]
        enc = model.encode_cases(train_cases, max_state_tokens=MAX_STATE_TOKENS,
                                 max_option_tokens=MAX_OPTION_TOKENS)
        rep_enc = model.encode_cases(replay_cases, max_state_tokens=MAX_STATE_TOKENS,
                                     max_option_tokens=MAX_OPTION_TOKENS)
        tok = None
        opt_groups = model.parameter_groups(2e-5, 1e-4)
    elif args.arm == "ocr":
        from mokka_decisions.arch_julia import JuliaArchScorer
        from safetensors.torch import load_file

        mk1_path = expected_checkpoint("mksh1_v12.safetensors")
        if mk1_path is None:
            raise SystemExit("worker ABORT: mksh1_v12.safetensors not attached")
        model = JuliaArchScorer("jhu-clsp/mmBERT-small").to(device)
        model.load_state_dict(load_file(str(mk1_path)))
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
        train_cases = [ocr_case_of(r, roots, ocr) for r in train_rows]
        enc = None
        rep_enc = None
        opt_groups = model.parameter_groups(2e-5, 1e-4)
    else:  # smol
        model = SmolScorer(device)
        train_cases = [case_of_row(r) for r in train_rows]
        enc = rep_enc = tok = None
        opt_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                       "lr": 2e-5}]

    # ---- eval helper for the dev half ---------------------------------------
    def eval_dev():
        from mksh_v.arch_a1v import evaluate_a1v
        from mokka_decisions.arch_julia import evaluate_julia_arch

        model.eval() if not isinstance(model, SmolScorer) else model.model.eval()
        recs = []
        with torch.no_grad():
            if args.arm == "a1v":
                ev = evaluate_a1v(model, [case_of_row(r) for r in dev_sel],
                                  [open_image(r, roots) for r in dev_sel],
                                  device=device, max_side=IMAGE_SIZE, batch_decisions=16,
                                  max_state_tokens=MAX_STATE_TOKENS,
                                  max_option_tokens=MAX_OPTION_TOKENS, collect_rows=True)
                rows = ev["rows"]
            elif args.arm == "ocr":
                cases = [ocr_case_of(r, roots, ocr) for r in dev_sel]
                qts = {c.id: qtype_of_row(r) for c, r in zip(cases, dev_sel)}
                ev = evaluate_julia_arch(model, tok, cases, qts, device=device,
                                         batch_decisions=32, collect_rows=True)
                rows = ev["rows"]
            else:
                rows = []
                for s in range(0, len(dev_sel), 16):
                    part = dev_sel[s:s + 16]
                    cases = [case_of_row(r) for r in part]
                    imgs = [open_image(r, roots) for r in part]
                    logits = model.batch_logits(cases, imgs)
                    probs = logits.softmax(-1)
                    for j, r in enumerate(part):
                        k = len(cases[j].options)
                        gold = cases[j].option_ids.index(cases[j].target_option)
                        pred = int(probs[j, :k].argmax().item())
                        rows.append({"correct": int(pred == gold)})
        for r, rr in zip(rows, dev_sel):
            recs.append(dict(r, family=rr["family"]))
        return macro_family_accuracy(recs), sum(r["correct"] for r in rows) / len(rows)

    # ---- training loop with E35 replay --------------------------------------
    opt = torch.optim.AdamW(opt_groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_of(args.steps))
    rng = random.Random(args.seed)
    order = list(range(len(train_rows)))
    rng.shuffle(order)
    rep_order = list(range(len(replay_cases)))
    rng.shuffle(rep_order)
    cursor = rep_cursor = 0
    best = {"dev_half_macro": -1.0, "dev_half_acc": -1.0, "step": 0}
    best_path = out_dir / "best.pt"
    full_path = out_dir / "ckpt_latest.pt"
    curves = []
    is_smol = isinstance(model, SmolScorer)
    if not is_smol:
        model.train()

    from mokka_decisions.arch_julia import collate_julia

    keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")

    def next_synthetic(m):
        nonlocal cursor
        if cursor + m > len(order):
            rng.shuffle(order)
            cursor = 0
        idx = order[cursor:cursor + m]
        cursor += m
        return idx

    def next_replay(m):
        nonlocal rep_cursor
        if rep_cursor + m > len(rep_order):
            rng.shuffle(rep_order)
            rep_cursor = 0
        idx = rep_order[rep_cursor:rep_cursor + m]
        rep_cursor += m
        return idx

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        for slot in range(accum):
            use_replay = (slot % REPLAY_EVERY) == (REPLAY_EVERY - 1)
            if args.arm == "a1v":
                if use_replay:
                    idx = next_replay(micro)
                    batch = model.build_batch([rep_enc[i] for i in idx],
                                              [None] * len(idx), device)
                else:
                    idx = next_synthetic(micro)
                    batch = model.build_batch(
                        [enc[i] for i in idx],
                        [open_image(train_rows[i], roots) for i in idx],
                        device, max_side=IMAGE_SIZE)
                from mksh_v.arch_a1v import forward_a1v

                with torch.autocast("cuda", dtype=torch.float16):
                    logits = forward_a1v(model, batch).float()
                    loss = torch.nn.functional.cross_entropy(
                        logits, batch["targets"], ignore_index=-100) / accum
            elif args.arm == "ocr":
                if use_replay:
                    idx = next_replay(micro)
                    chunk = [replay_cases[i] for i in idx]
                    qts = {c.id: replay_qt[c.id] for c in chunk}
                else:
                    idx = next_synthetic(micro)
                    chunk = [train_cases[i] for i in idx]
                    qts = {c.id: qtype_of_row(train_rows[i]) for i in idx}
                batch = collate_julia(tok, chunk, qts, 1024, 256, device)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**{k: batch[k] for k in keys})
                    loss = torch.nn.functional.cross_entropy(
                        logits.float(), batch["targets"], ignore_index=-100) / accum
            else:  # smol
                if use_replay:
                    idx = next_replay(micro)
                    chunk = [replay_cases[i] for i in idx]
                else:
                    idx = next_synthetic(micro)
                    chunk = [train_cases[i] for i in idx]
                imgs = None if use_replay else [
                    open_image(train_rows[i], roots) for i in idx]
                logits = model.batch_logits(chunk, imgs)
                golds = torch.tensor(
                    [c.option_ids.index(c.target_option) for c in chunk],
                    device=logits.device)
                loss = torch.nn.functional.cross_entropy(logits, golds) / accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % DEV_EVERY == 0 or step == args.steps:
            macro, acc = eval_dev()
            curves.append({"step": step, "dev_half_macro": round(macro, 4),
                           "dev_half_acc": round(acc, 4)})
            log(f"{args.arm}[s{args.seed}] step {step}/{args.steps} "
                f"dev-half macro {macro:.4f}")
            if (macro, acc) >= (best["dev_half_macro"], best["dev_half_acc"]):
                best = {"dev_half_macro": macro, "dev_half_acc": acc, "step": step}
                save_best_weights(best_path, model.model if is_smol else model)
            if not is_smol:
                model.train()
        if step % 500 == 0:
            save_full_checkpoint(full_path, model.model if is_smol else model,
                                 opt, sched, scaler, step, cursor, rng.getstate())
    (out_dir / "curves.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in curves), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(
        {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()},
         "arm": args.arm, "seed": args.seed, "steps": args.steps}, indent=2),
        encoding="utf-8")
    log(f"worker done: {args.arm} s{args.seed} best {best}")


def spawn(arm: str, seed: int, gpu: int, steps: int) -> tuple:
    """Worker as subprocess with output to a FILE (a PIPE nobody reads while
    another proc runs would fill its buffer and deadlock the worker)."""
    log_dir = WORK / "runs" / f"{arm}_s{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"worker_gpu{gpu}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    fh = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--worker",
         "--arm", arm, "--seed", str(seed), "--gpu", str(gpu),
         "--steps", str(steps)],
        env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
    return proc, fh, log_path


def run_and_log(spawned: tuple, label: str) -> None:
    proc, fh, log_path = spawned
    proc.wait()
    fh.close()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        print(f"[{label}] {line}", flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"E39 ABORT: worker {label} exited {proc.returncode}")


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=("a1v", "ocr", "smol"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
        return

    t_start = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    check_secret_and_repos()  # FIRST: no quota burnt on a bad secret
    pip_install()
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("E39 ABORT: needs the T4x2 session (2 GPUs)")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    import transformers

    env = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "transformers": transformers.__version__,
           "gpu_count": torch.cuda.device_count(),
           "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)}
    (WORK / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    log(f"env: {env}")
    find_bundle()
    find_ocr_cache()

    # warm the HF cache once so workers start fast
    roots = download_data()
    verify_rows_sha(roots)
    splits = load_e38_rows(roots)
    log(f"rows: train {len(splits['train'])} dev {len(splits['dev'])} "
        f"test {len(splits['test'])}")
    from mksh_v.train_compare import fixed_dev_half

    dev_half = fixed_dev_half(splits["dev"], seed=DEV_HALF_SEED, fraction=0.5)
    (WORK / "dev_half_ids.json").write_text(json.dumps(
        {"seed": DEV_HALF_SEED, "n": len(dev_half), "ids": sorted(dev_half)}),
        encoding="utf-8")

    # smoke-first: 20 steps of each arm on gpu1 while gpu0 idles briefly
    log("smoke: 20 steps per arm (gpu0 a1v; gpu1 ocr then smol)")
    run_and_log(spawn("a1v", 42, 0, SMOKE_STEPS), "smoke-a1v")
    run_and_log(spawn("ocr", 42, 1, SMOKE_STEPS), "smoke-ocr")
    run_and_log(spawn("smol", 42, 1, SMOKE_STEPS), "smoke-smol")

    # phase 1: compared runs, seed 42
    log("phase 1: a1v s42 on gpu0; ocr s42 then smol s42 on gpu1")
    p_a1v = spawn("a1v", 42, 0, STEPS)
    run_and_log(spawn("ocr", 42, 1, STEPS), "ocr-s42")
    run_and_log(spawn("smol", 42, 1, STEPS), "smol-s42")
    run_and_log(p_a1v, "a1v-s42")

    def summary_of(arm: str, seed: int) -> dict:
        return json.loads((WORK / "runs" / f"{arm}_s{seed}" / "summary.json")
                          .read_text(encoding="utf-8"))

    s42 = {arm: summary_of(arm, 42) for arm in ("a1v", "ocr", "smol")}
    non_visual = max(("ocr", "smol"), key=lambda a: s42[a]["dev_half_macro"])
    finalist = ("a1v" if s42["a1v"]["dev_half_macro"] >= s42[non_visual]["dev_half_macro"]
                else non_visual)
    log(f"finalist={finalist} (dev macro a1v {s42['a1v']['dev_half_macro']} vs "
        f"{non_visual} {s42[non_visual]['dev_half_macro']})")

    # phase 2: finalist seeds 43/44, one per GPU
    log(f"phase 2: finalist {finalist} seeds 43/44 in parallel")
    run_and_log(spawn(finalist, 43, 0, STEPS), f"{finalist}-s43")
    run_and_log(spawn(finalist, 44, 1, STEPS), f"{finalist}-s44")

    # phase 3: evals + gates (this process)
    from mksh_v.arch_a1v import NeoMMEMarkerScorer, evaluate_a1v
    from mksh_v.train_compare import (case_of_row, gate_report,
                                      paired_delta_ci, qtype_of_row)

    ocr = load_ocr_cache()
    device = "cuda:0"
    a1v = NeoMMEMarkerScorer().to(device)
    a1v.ensure_markers(16)
    load_best_weights(WORK / "runs/a1v_s42/best.pt", a1v)
    a1v.to(device)

    from mokka_decisions.arch_julia import JuliaArchScorer, evaluate_julia_arch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    mk1 = JuliaArchScorer("jhu-clsp/mmBERT-small").to(device)
    mk1_path = expected_checkpoint("mksh1_v12.safetensors")
    from safetensors.torch import load_file

    mk1.load_state_dict(load_file(str(mk1_path)))
    load_best_weights(WORK / "runs/ocr_s42/best.pt", mk1)

    smol = SmolScorer(device)
    load_best_weights(WORK / "runs/smol_s42/best.pt", smol.model)

    test_nonfr = [r for r in splits["test"] if r["language"] != "fr"]
    test_fr = [r for r in splits["test"] if r["language"] == "fr"]

    def eval_rows(arm, rows, *, transform=None, image_index_map=None):
        """Per-case rows + accuracy of one arm on rows (chunked, RAM-safe)."""
        out_rows, correct, n = [], 0, 0
        chunk = 400 if arm != "ocr" else 800
        for s in range(0, len(rows), chunk):
            part = rows[s:s + chunk]
            if arm == "a1v":
                if image_index_map is not None:
                    imgs = [open_image(rows[image_index_map[s + j]], roots)
                            for j in range(len(part))]
                else:
                    imgs = [open_image(r, roots) for r in part]
                if transform is not None:
                    imgs = [transform(im) for im in imgs]
                ev = evaluate_a1v(a1v, [case_of_row(r) for r in part], imgs,
                                  device=device, max_side=IMAGE_SIZE, batch_decisions=16,
                                  max_state_tokens=MAX_STATE_TOKENS,
                                  max_option_tokens=MAX_OPTION_TOKENS,
                                  collect_rows=True)
                rows_out, hits, cnt = ev["rows"], sum(r["correct"] for r in ev["rows"]), ev["n"]
            elif arm == "ocr":
                cases = [ocr_case_of(r, roots, ocr) for r in part]
                qts = {c.id: qtype_of_row(r) for c, r in zip(cases, part)}
                ev = evaluate_julia_arch(mk1, tok, cases, qts, device=device,
                                         batch_decisions=32, collect_rows=True)
                rows_out, hits, cnt = ev["rows"], sum(r["correct"] for r in ev["rows"]), ev["n"]
            else:
                rows_out = []
                hits = cnt = 0
                import torch as _t

                with _t.no_grad():
                    for s2 in range(0, len(part), 16):
                        sub = part[s2:s2 + 16]
                        cases = [case_of_row(r) for r in sub]
                        if image_index_map is not None:
                            imgs = [open_image(rows[image_index_map[s + s2 + j]], roots)
                                    for j in range(len(sub))]
                        else:
                            imgs = [open_image(r, roots) for r in sub]
                        logits = smol.batch_logits(cases, imgs)
                        probs = logits.softmax(-1)
                        for j, r in enumerate(sub):
                            k = len(cases[j].options)
                            gold = cases[j].option_ids.index(cases[j].target_option)
                            pred = int(probs[j, :k].argmax().item())
                            ok = int(pred == gold)
                            rows_out.append({"correct": ok})
                            hits += ok
                            cnt += 1
            out_rows += [dict(r, family=rr.get("family", "text"),
                              language=rr.get("language", "en"),
                              group_id=rr.get("group_id", rr["id"]))
                         for r, rr in zip(rows_out, part)]
            correct += hits
            n += cnt
        return out_rows, (correct / n if n else None)

    test_rows = {}
    test_acc = {}
    for arm in ("a1v", non_visual):
        test_rows[arm], test_acc[arm] = eval_rows(arm, test_nonfr)
        (WORK / f"eval_test_nonfr__{arm}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in test_rows[arm]), encoding="utf-8")
    for arm in ("a1v", "ocr", "smol"):
        fr_rows, fr_acc = eval_rows(arm, test_fr)
        (WORK / f"eval_test_fr__{arm}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in fr_rows), encoding="utf-8")
        test_acc[f"{arm}_fr"] = fr_acc
    # finalist seeds on non-FR test (reported); gates always use s42 bests
    seed_reports = [dict(s42[a], arm=a) for a in ("a1v", "ocr", "smol")]
    for seed in (43, 44):
        if finalist == "a1v":
            load_best_weights(WORK / f"runs/a1v_s{seed}/best.pt", a1v)
            a1v.to(device)
        elif finalist == "ocr":
            load_best_weights(WORK / f"runs/ocr_s{seed}/best.pt", mk1)
            mk1.to(device)
        else:
            load_best_weights(WORK / f"runs/smol_s{seed}/best.pt", smol.model)
            smol.model.to(device)
        _, acc_seed = eval_rows(finalist, test_nonfr)
        seed_reports.append({"arm": finalist, "seed": seed,
                             "test_nonfr_acc": round(acc_seed, 4),
                             **summary_of(finalist, seed)})
    # restore s42-best for the gates
    load_best_weights(WORK / "runs/a1v_s42/best.pt", a1v)
    a1v.to(device)
    load_best_weights(WORK / "runs/ocr_s42/best.pt", mk1)
    mk1.to(device)

    log(f"test non-FR: " + " ".join(f"{a}={test_acc[a]:.4f}" for a in test_rows))

    # G1: paired-by-episode bootstrap, a1v vs best non-visual arm
    by_id = {r["id"]: r for r in test_rows[non_visual]}
    delta_ci = paired_delta_ci(
        [r["correct"] for r in test_rows["a1v"]],
        [int(by_id[r["id"]]["correct"]) for r in test_rows["a1v"]],
        [r["group_id"] for r in test_rows["a1v"]], n_boot=1000, seed=42)

    # G2: E35 text sets, CPU float32, every arm vs frozen bars
    bundle = find_bundle()
    man29 = json.loads((bundle / "e29-mix/manifest.json").read_text(encoding="utf-8"))
    want = set(man29["original_dev_ids"])
    task_rows = [json.loads(l) for l in (bundle / "data/processed/v1/eval/dev_eval.jsonl")
                 .read_text(encoding="utf-8").splitlines()
                 if l.strip() and json.loads(l)["id"] in want]
    a1v.to("cpu"); a1v.eval()
    mk1.to("cpu"); mk1.eval()
    smol.model.to("cpu"); smol.model.eval()
    text_sets = {}
    for name, rr in (("testC", [json.loads(l) for l in (bundle / "e26-order/test_c_rows.jsonl")
                                 .read_text(encoding="utf-8").splitlines() if l.strip()]),
                     ("task", task_rows)):
        accs = {}
        cases = [case_of_row(r) for r in rr]
        ev = evaluate_a1v(a1v, cases, None, device="cpu", batch_decisions=16,
                          max_state_tokens=MAX_STATE_TOKENS,
                          max_option_tokens=MAX_OPTION_TOKENS)
        accs["a1v"] = round(ev["accuracy"], 4)
        qts = {c.id: qtype_of_row(r) for c, r in zip(cases, rr)}
        ev2 = evaluate_julia_arch(mk1, tok, cases, qts, device="cpu", batch_decisions=32)
        accs["ocr"] = round(ev2["accuracy"], 4)
        import torch as _t

        with _t.no_grad():
            hits = cnt = 0
            for s2 in range(0, len(cases), 16):
                sub = cases[s2:s2 + 16]
                logits = smol.batch_logits(sub, [None] * len(sub))
                for j, c in enumerate(sub):
                    k = len(c.options)
                    gold = c.option_ids.index(c.target_option)
                    hits += int(int(logits[j, :k].argmax().item()) == gold)
                    cnt += 1
        accs["smol"] = round(hits / cnt, 4)
        text_sets[name] = accs
    gates = gate_report(delta_ci=delta_ci, text_sets=text_sets, text_bars=TEXT_BARS,
                        delta_lo_margin=DELTA_LO_MARGIN)

    # G3: ablation tripwire on the visual arm (a1v s42 best)
    from mksh_v.train_compare import blank_like, degrade, swap_map_within_blocks

    a1v.to(device)
    swap_map = swap_map_within_blocks(
        [f'{r["family"]}|{r["language"]}' for r in test_nonfr])
    swap_rows, acc_swap = eval_rows("a1v", test_nonfr, image_index_map=swap_map)
    _, acc_blank = eval_rows("a1v", test_nonfr, transform=blank_like)
    _, acc_degr = eval_rows("a1v", test_nonfr, transform=degrade)
    g3 = {"bar": f"swap drop >= {SWAP_DROP_BAR:.2f}",
          "swap_drop": round(test_acc["a1v"] - acc_swap, 4),
          "swap_acc": round(acc_swap, 4), "blank_acc": round(acc_blank, 4),
          "degraded_acc": round(acc_degr, 4)}
    g3["passed"] = (test_acc["a1v"] - acc_swap) >= SWAP_DROP_BAR
    (WORK / "eval_ablations__a1v.json").write_text(json.dumps(g3, indent=2), encoding="utf-8")

    # G4: real eval (repo may not exist; skipped cleanly)
    g4 = {"status": "not_run", "reason": f"{REAL_EVAL_REPO} not available (E38c pending)"}
    try:
        from huggingface_hub import list_repo_files, snapshot_download

        files = list_repo_files(REAL_EVAL_REPO, repo_type="dataset")
        rows_file = next((f for f in files if f.endswith("rows.jsonl")), None)
    except Exception as exc:
        rows_file = None
        log(f"real eval skipped: {REAL_EVAL_REPO} not readable ({type(exc).__name__})")
    if rows_file:
        real_root_path = Path(snapshot_download(
            repo_id=REAL_EVAL_REPO, repo_type="dataset",
            allow_patterns=["*.jsonl", "*.png", "*.jpg", "*.jpeg"]))
        real_rows = [json.loads(l) for l in (real_root_path / rows_file)
                     .read_text(encoding="utf-8").splitlines() if l.strip()]
        real_root = {"E38a": real_root_path, "E38b": real_root_path}
        # OCR the real images (small set) on CPUs, then swap the roots/ocr
        # context so eval_rows resolves paths against the real snapshot
        from concurrent.futures import ProcessPoolExecutor

        _TESS = {"en": "eng", "de": "deu", "es": "spa", "fr": "fra"}

        def _ocr_job(p_lang):
            p, lang = p_lang
            import pytesseract
            from PIL import Image

            return pytesseract.image_to_string(
                Image.open(p), lang=_TESS.get(lang, "eng"))

        real_ocr = {}
        with ProcessPoolExecutor(max_workers=3) as pool:
            for r, text in zip(real_rows, pool.map(
                    _ocr_job, [(str(real_root_path / r["image"]), r.get("language", "en"))
                               for r in real_rows], chunksize=8)):
                real_ocr[r["id"]] = text

        def eval_real(arm, rows):
            """eval_rows against the real snapshot: monkey-context, no globals."""
            out_rows, correct, n = [], 0, 0
            for s in range(0, len(rows), 200):
                part = rows[s:s + 200]
                if arm == "a1v":
                    imgs = [open_image(r, real_root) for r in part]
                    ev = evaluate_a1v(a1v, [case_of_row(r) for r in part], imgs,
                                      device=device, max_side=IMAGE_SIZE,
                                      batch_decisions=16,
                                      max_state_tokens=MAX_STATE_TOKENS,
                                      max_option_tokens=MAX_OPTION_TOKENS,
                                      collect_rows=True)
                    out_rows += ev["rows"]
                    correct += sum(r["correct"] for r in ev["rows"])
                    n += ev["n"]
                else:
                    cases = []
                    for r in part:
                        c = case_of_row(r)
                        text = real_ocr.get(r["id"])
                        if text is not None:
                            c = dataclasses.replace(c, state=text)
                        cases.append(c)
                    qts = {c.id: qtype_of_row(r) for c, r in zip(cases, part)}
                    ev = evaluate_julia_arch(mk1.to(device), tok, cases, qts,
                                             device=device, batch_decisions=32,
                                             collect_rows=True)
                    out_rows += ev["rows"]
                    correct += sum(r["correct"] for r in ev["rows"])
                    n += ev["n"]
            return out_rows, (correct / n if n else None)

        accs_real = {}
        for arm in ("a1v", non_visual):
            real_rows_arm, acc = eval_real(arm, real_rows)
            accs_real[arm] = round(acc, 4)
            (WORK / f"eval_real__{arm}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in real_rows_arm), encoding="utf-8")
        g4 = {"status": "completed", "repo": REAL_EVAL_REPO, **accs_real,
              "kill_bar": REAL_EVAL_KILL,
              "non_discriminating": all(v >= REAL_EVAL_KILL for v in accs_real.values())}
    (WORK / "e39_real_eval.json").write_text(json.dumps(g4, indent=2), encoding="utf-8")

    summary = {
        "status": "completed", "env": env,
        "steps": STEPS, "image_size": IMAGE_SIZE,
        "effective_batch": EFFECTIVE_BATCH, "replay_every": REPLAY_EVERY,
        "finalist": finalist, "best_non_visual": non_visual,
        "selection": seed_reports,
        "test_nonfr": {a: round(v, 4) for a, v in test_acc.items()
                       if not a.endswith("_fr")},
        "test_fr": {a[:-3]: round(v, 4) for a, v in test_acc.items() if a.endswith("_fr")},
        "delta_ci": delta_ci, "text_sets": text_sets, "gates": gates,
        "G3_ablations": g3, "G4_real": g4,
        "wall_s": round(time.time() - t_start, 1),
    }
    (WORK / "e39_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"E39 done in {summary['wall_s']}s; gates: {json.dumps(gates)}")


if __name__ == "__main__":
    main()
