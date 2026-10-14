"""E39: compared training, synthetic-only (criteria: docs/E39_CRITERIA_*.md at
freeze time). ONE job, both arms, real eval included when its dataset is
attached. NOT LAUNCHED until the criteria are frozen and the user gives GO.

Arms (seed 42 head-to-head; the finalist also runs seeds 43/44):
  A1-V      NeoMME + marker head, init = E37 C3 checkpoint, 768 px
  OCR->MK1  Tesseract(original PNG) -> state, MK-SH-1 fine-tuned (E35 recipe)

Selection is dev-only: dev-half every 500 steps, best weights kept by
macro-dev; test is opened once, with the BEST (not last) weights.

Expected attachments:
  dataset gjusev/mkshv-e37-ckpt : c3_final.safetensors (A1-V init; an absent
                                  file triggers the pre-declared C3-rerun
                                  fallback), mksh1_v12.safetensors (OCR init)
  dataset gjusev/mokka-decisions-bundle : E35 text sets (G2) + fallback mix
  secrets: HF_TOKEN (read-only, per-notebook) - checked FIRST, before any
  download, so a misconfigured notebook burns no quota. The token is never
  printed, logged or written to files.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

WORK = Path("/kaggle/working")
E38_REPOS = {"E38a": "Gjusev/MK-SH-V-E38a", "E38b": "Gjusev/MK-SH-V-E38b"}
STEPS = 2000
EFFECTIVE_BATCH = 32
MICROBATCH = 8
ACCUMULATION = EFFECTIVE_BATCH // MICROBATCH
IMAGE_SIZE = 768
MAX_STATE_TOKENS = 700
MAX_OPTION_TOKENS = 96
DEV_EVERY = 500
DEV_HALF_SEED = 39
TEXT_MARGIN = 0.02
DELTA_LO_MARGIN = 0.02
SWAP_DROP_BAR = 0.15
REAL_EVAL_KILL = 0.95
CKPT_SHA_PREFIXES = {  # full shas pinned at criteria freeze
    "c3_final.safetensors": "bac7c72242d236f8",   # E37 C3 (A1-V init)
    "mksh1_v12.safetensors": "ea66cdde",          # MK-SH-1 v1.2 (OCR init)
}
# real-eval repo (probe outcome: CORD->field_threshold ~188, GUI-Odyssey->
# task_done ~500 pending re-probe r2; DocLayNet out). Provisional name; the
# kernel only uses it if it exists and is readable.
REAL_EVAL_REPO = "Gjusev/MK-SH-V-E38c"


def log(msg: str) -> None:
    print(f"[e39 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_secret_and_repos() -> None:
    """Cheap auth gate FIRST: abort before any download/training if unusable."""
    from kaggle_secrets import UserSecretsClient  # Kaggle-only

    token = UserSecretsClient().get_secret("HF_TOKEN")  # never printed/logged
    if not token:
        raise SystemExit("E39 ABORT: HF_TOKEN secret missing on this notebook "
                         "(Add-ons -> Secrets). No quota spent.")
    os.environ["HF_TOKEN"] = token
    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    for repo in E38_REPOS.values():
        files = api.list_repo_files(repo, repo_type="dataset")
        if not any(f.endswith("rows.jsonl") for f in files):
            raise SystemExit(f"E39 ABORT: {repo} readable but has no rows.jsonl")
    log(f"HF auth OK (user {who.get('name', '?')}); both E38 repos readable")


def pip_install() -> None:
    pkgs = [
        "git+https://github.com/Gjusev/MK-SH-1@9e1a226e",
        "transformers==5.19.0",
        "pillow>=10.1",
        "huggingface_hub>=0.34",
        "pytesseract>=0.3",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
    subprocess.run(["apt-get", "-qq", "install", "-y",
                    "tesseract-ocr", "tesseract-ocr-deu", "tesseract-ocr-spa",
                    "tesseract-ocr-fra"], check=False)
    out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
    log("tesseract: " + ((out.stdout or out.stderr).splitlines() or ["?"])[0])


_TESS_LANG = {"en": "eng", "de": "deu", "es": "spa", "fr": "fra"}


def _ocr_one(job):
    import pytesseract
    from PIL import Image

    path, lang = job
    try:
        text = pytesseract.image_to_string(Image.open(path), lang=_TESS_LANG.get(lang, "eng"))
        return path, text, None
    except Exception as exc:  # measured failure, never dropped silently
        return path, "", f"{type(exc).__name__}: {exc}"


OCR_TEXT: dict[str, str] = {}


def start_ocr_thread(jobs: list) -> threading.Thread:
    """OCR on CPUs concurrently with GPU training; fills OCR_TEXT when done."""
    from concurrent.futures import ProcessPoolExecutor

    def run():
        t0 = time.time()
        ocr, errors = {}, {}
        with ProcessPoolExecutor(max_workers=3) as pool:
            for path, text, err in pool.map(_ocr_one, jobs, chunksize=16):
                if err:
                    errors[path] = err
                ocr[path] = text
        OCR_TEXT.update(ocr)
        (WORK / "ocr_manifest.json").write_text(json.dumps({
            "n": len(ocr), "errors": errors, "wall_s": round(time.time() - t0, 1),
            "per_image_mean_s": round((time.time() - t0) / max(len(ocr), 1), 3),
        }, indent=2), encoding="utf-8")
        log(f"OCR thread done: {len(ocr)} images, {len(errors)} errors")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def ocr_real_rows_sync(rows: list, images_root: Path) -> None:
    """OCR the (small) real-eval image set synchronously before its eval."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(str(images_root / r["image"]), r.get("language", "en")) for r in rows]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for path, text, err in pool.map(_ocr_one, jobs, chunksize=8):
            OCR_TEXT[path] = text
    log(f"real-eval OCR done: {len(jobs)} images")


def download_data() -> dict:
    from huggingface_hub import snapshot_download

    roots = {}
    for key, repo in E38_REPOS.items():
        t0 = time.time()
        roots[key] = Path(snapshot_download(
            repo_id=repo, repo_type="dataset",
            allow_patterns=["manifest.json", "*/rows.jsonl", "*/train/*.png",
                            "*/dev/*.png", "*/test/*.png"],
        ))
        log(f"downloaded {repo} ({time.time() - t0:.0f}s)")
    return roots


def verify_rows_sha(roots: dict) -> None:
    """The per-family rows.jsonl must match the repo manifest's frozen sha."""
    for key, root in roots.items():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for family, spec in manifest["families"].items():
            got = hashlib.sha256((root / spec["rows"]).read_bytes()).hexdigest()
            if got != spec["rows_sha256"]:
                raise SystemExit(f"E39 ABORT: rows sha mismatch for {key}/{family}: {got}")
    log("rows.jsonl sha256 verified against both repo manifests")


def find_bundle() -> Path:
    for cand in Path("/kaggle/input").rglob("BUNDLE_MANIFEST.json"):
        return cand.parent
    raise SystemExit("E39 ABORT: gjusev/mokka-decisions-bundle dataset not attached "
                     "(needed for the E35 text sets and the C3 fallback)")


def image_path_of(row: dict, roots: dict) -> Path:
    repo = "E38a" if row["id"].startswith("e38a") else "E38b"
    return roots[repo] / row["image"]


def open_image(row: dict, roots: dict):
    from PIL import Image

    return Image.open(image_path_of(row, roots)).convert("RGB")


def expected_checkpoint(name: str) -> Path | None:
    for cand in sorted(Path("/kaggle/input").rglob(name)):
        got = hashlib.sha256(cand.read_bytes()).hexdigest()
        if got.startswith(CKPT_SHA_PREFIXES[name]):
            return cand
        log(f"sha mismatch for {name} at {cand}: {got[:16]}...")
    return None


def save_full_checkpoint(path: Path, model, opt, sched, scaler, step, cursor, rng_state) -> None:
    """Rolling full-state checkpoint (overwritten in place: ~3 GB per run)."""
    import torch

    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
        "step": step, "data_cursor": cursor, "rng_python": rng_state,
        "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
    }, path)


def save_best_weights(path: Path, model) -> None:
    import torch

    torch.save({"model": model.state_dict()}, path)


def load_best_weights(path: Path, model) -> None:
    import torch

    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])


def lr_lambda_of(total_steps: int):
    warmup = max(int(total_steps * 0.05), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        pr = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.05, 0.95 * (1 - pr) + 0.05)

    return lr_lambda


# ---------------------------------------------------------------------------
# arm 1: A1-V
# ---------------------------------------------------------------------------
def build_a1v(device, from_fallback: bool = False):
    from mksh_v.arch_a1v import NeoMMEMarkerScorer

    model = NeoMMEMarkerScorer().to(device)
    model.ensure_markers(32)  # the C3 checkpoint's tables are 32-marker wide
    ckpt = expected_checkpoint("c3_final.safetensors")
    if ckpt and not from_fallback:
        from safetensors.torch import load_file

        state = load_file(str(ckpt))
        model.load_state_dict(state, strict=True)  # shapes must match exactly
        log(f"A1-V init: E37 C3 checkpoint (sha verified) {ckpt}")
    else:
        log("A1-V init: c3_final.safetensors absent -> C3-rerun fallback (pre-declared)")
        c3_rerun(model, device)
    return model


def c3_rerun(model, device) -> None:
    """C3 recipe rerun (E35 mix, seed 42): 3 phases x 5 epochs, fresh AdamW
    per phase, rng 42^0xE35^phase — mirroring kaggle/e37_c3.py exactly."""
    import torch

    from mksh_v.arch_a1v import forward_a1v
    from mksh_v.train_compare import case_of_row

    bundle = find_bundle()
    rows_mix = []
    for rel in ("e35-extremes/train_rows.jsonl", "e28-ext/train_rows.jsonl"):
        rows_mix += [json.loads(l) for l in (bundle / rel).read_text(encoding="utf-8")
                     .splitlines() if l.strip()]
    man = json.loads((bundle / "e29-mix/manifest.json").read_text(encoding="utf-8"))
    want = set(man["original_train_ids"])
    rows_mix += [json.loads(l) for l in (bundle / "data/processed/v1/instances/train.jsonl")
                 .read_text(encoding="utf-8").splitlines()
                 if l.strip() and json.loads(l)["id"] in want]
    cases = [case_of_row(r) for r in rows_mix]
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
            for _ in range(ACCUMULATION):
                if cursor + MICROBATCH > len(order):
                    rng.shuffle(order)
                    cursor = 0
                idx = order[cursor:cursor + MICROBATCH]
                cursor += MICROBATCH
                batch = model.build_batch([encoded[i] for i in idx], [None] * len(idx), device)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = forward_a1v(model, batch).float()
                    loss = torch.nn.functional.cross_entropy(
                        logits, batch["targets"], ignore_index=-100) / ACCUMULATION
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


def train_a1v(model, train_rows, roots, dev_sel, device, *, seed, tag) -> dict:
    import torch

    from mksh_v.arch_a1v import evaluate_a1v, forward_a1v
    from mksh_v.train_compare import case_of_row, macro_family_accuracy

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    model.train()
    cases = [case_of_row(r) for r in train_rows]
    encoded = model.encode_cases(cases, max_state_tokens=MAX_STATE_TOKENS,
                                 max_option_tokens=MAX_OPTION_TOKENS)
    dev_cases = [case_of_row(r) for r in dev_sel]
    opt = torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_of(STEPS))
    rng = random.Random(seed)
    order, cursor = list(range(len(cases))), 0
    rng.shuffle(order)
    best = {"dev_half_macro": -1.0, "dev_half_acc": -1.0, "step": 0}
    best_path = WORK / f"best_a1v_{tag}.pt"
    full_path = WORK / f"ckpt_a1v_{tag}_latest.pt"  # rolling, overwritten
    curves = []
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUMULATION):
            if cursor + MICROBATCH > len(order):
                rng.shuffle(order)
                cursor = 0
            idx = order[cursor:cursor + MICROBATCH]
            cursor += MICROBATCH
            batch = model.build_batch(
                [encoded[i] for i in idx],
                [open_image(train_rows[i], roots) for i in idx],  # lazy: RAM stays flat
                device, max_side=IMAGE_SIZE)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = forward_a1v(model, batch).float()
                loss = torch.nn.functional.cross_entropy(
                    logits, batch["targets"], ignore_index=-100) / ACCUMULATION
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % DEV_EVERY == 0 or step == STEPS:
            model.eval()
            ev = evaluate_a1v(model, dev_cases,
                              [open_image(r, roots) for r in dev_sel], device=device,
                              max_side=IMAGE_SIZE, batch_decisions=16,
                              max_state_tokens=MAX_STATE_TOKENS,
                              max_option_tokens=MAX_OPTION_TOKENS, collect_rows=True)
            recs = [dict(r, family=rr["family"]) for r, rr in zip(ev["rows"], dev_sel)]
            macro = macro_family_accuracy(recs)
            curves.append({"step": step, "dev_half_macro": round(macro, 4),
                           "dev_half_acc": round(ev["accuracy"], 4)})
            log(f"A1-V[{tag}] step {step}/{STEPS} dev-half macro {macro:.4f}")
            if (macro, ev["accuracy"]) >= (best["dev_half_macro"], best["dev_half_acc"]):
                best = {"dev_half_macro": macro, "dev_half_acc": ev["accuracy"], "step": step}
                save_best_weights(best_path, model)
            model.train()
        if step % 500 == 0:
            save_full_checkpoint(full_path, model, opt, sched, scaler, step, cursor,
                                 rng.getstate())
    (WORK / f"curves_a1v_{tag}.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in curves), encoding="utf-8")
    load_best_weights(best_path, model)  # selection decides, not the last step
    model.to(device)
    return {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()},
            "best_path": str(best_path)}


# ---------------------------------------------------------------------------
# arm 2: OCR -> MK-SH-1
# ---------------------------------------------------------------------------
def ocr_case_of(row: dict, roots: dict | None):
    """E38/real row -> case with OCR text as state; rows without an image
    (E35 text sets in G2) keep their own state verbatim. Empty OCR text is
    kept (an empty extraction is a measured fact, not a fallback)."""
    from mksh_v.train_compare import case_of_row

    c = case_of_row(row)
    if roots is None or not row.get("image"):
        return c
    key = str(image_path_of(row, roots))
    if key in OCR_TEXT:
        return dataclasses.replace(c, state=OCR_TEXT[key])
    return c


def train_ocr(model, tok, train_rows, roots, dev_sel, device, *, seed, tag) -> dict:
    import torch

    from mokka_decisions.arch_julia import collate_julia, evaluate_julia_arch
    from mksh_v.train_compare import macro_family_accuracy, qtype_of_row

    torch.manual_seed(seed)
    random.seed(seed)
    model.train()
    keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    train_cases = [ocr_case_of(r, roots) for r in train_rows]
    qt = {c.id: qtype_of_row(r) for c, r in zip(train_cases, train_rows)}
    opt = torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_of(STEPS))
    rng = random.Random(seed)
    order, cursor = list(range(len(train_cases))), 0
    rng.shuffle(order)
    best = {"dev_half_macro": -1.0, "dev_half_acc": -1.0, "step": 0}
    best_path = WORK / f"best_ocr_{tag}.pt"
    full_path = WORK / f"ckpt_ocr_{tag}_latest.pt"
    curves = []
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUMULATION):
            if cursor + MICROBATCH > len(order):
                rng.shuffle(order)
                cursor = 0
            idx = order[cursor:cursor + MICROBATCH]
            cursor += MICROBATCH
            chunk_cases = [train_cases[i] for i in idx]
            batch = collate_julia(tok, chunk_cases, {c.id: qt[c.id] for c in chunk_cases},
                                  1024, 256, device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**{k: batch[k] for k in keys})
                loss = torch.nn.functional.cross_entropy(
                    logits.float(), batch["targets"], ignore_index=-100) / ACCUMULATION
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % DEV_EVERY == 0 or step == STEPS:
            model.eval()
            dev_cases = [ocr_case_of(r, roots) for r in dev_sel]
            dev_qt = {c.id: qtype_of_row(r) for c, r in zip(dev_cases, dev_sel)}
            ev = evaluate_julia_arch(model, tok, dev_cases, dev_qt, device=device,
                                     batch_decisions=32, collect_rows=True)
            recs = [dict(r, family=rr["family"]) for r, rr in zip(ev["rows"], dev_sel)]
            macro = macro_family_accuracy(recs)
            curves.append({"step": step, "dev_half_macro": round(macro, 4),
                           "dev_half_acc": round(ev["accuracy"], 4)})
            log(f"OCR[{tag}] step {step}/{STEPS} dev-half macro {macro:.4f}")
            if (macro, ev["accuracy"]) >= (best["dev_half_macro"], best["dev_half_acc"]):
                best = {"dev_half_macro": macro, "dev_half_acc": ev["accuracy"], "step": step}
                save_best_weights(best_path, model)
            model.train()
        if step % 500 == 0:
            save_full_checkpoint(full_path, model, opt, sched, scaler, step, cursor,
                                 rng.getstate())
    (WORK / f"curves_ocr_{tag}.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in curves), encoding="utf-8")
    load_best_weights(best_path, model)
    return {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()},
            "best_path": str(best_path)}


# ---------------------------------------------------------------------------
# chunked evaluation (RAM-safe: images opened per chunk)
# ---------------------------------------------------------------------------
def eval_a1v_chunked(model, rows, roots, device, *, chunk=400, transform=None,
                     image_index_map=None):
    from mksh_v.arch_a1v import evaluate_a1v
    from mksh_v.train_compare import case_of_row

    model.eval()
    out_rows, correct, n = [], 0, 0
    for s in range(0, len(rows), chunk):
        part = rows[s:s + chunk]
        if image_index_map is not None:  # ablation: image of a DIFFERENT case
            imgs = [open_image(rows[image_index_map[s + j]], roots) for j in range(len(part))]
        else:
            imgs = [open_image(r, roots) for r in part]
        if transform is not None:
            imgs = [transform(im) for im in imgs]
        ev = evaluate_a1v(model, [case_of_row(r) for r in part], imgs, device=device,
                          max_side=IMAGE_SIZE, batch_decisions=16,
                          max_state_tokens=MAX_STATE_TOKENS,
                          max_option_tokens=MAX_OPTION_TOKENS, collect_rows=True)
        out_rows += [dict(r, family=rr.get("family", "text"),
                          language=rr.get("language", "en"),
                          group_id=rr.get("group_id", rr["id"]))
                     for r, rr in zip(ev["rows"], part)]
        correct += sum(r["correct"] for r in ev["rows"])
        n += ev["n"]
    return out_rows, (correct / n if n else None)


def eval_ocr_chunked(model, tok, rows, roots, device, *, chunk=800):
    from mokka_decisions.arch_julia import evaluate_julia_arch
    from mksh_v.train_compare import qtype_of_row

    model.eval()
    out_rows, correct, n = [], 0, 0
    for s in range(0, len(rows), chunk):
        part = rows[s:s + chunk]
        cases = [ocr_case_of(r, roots) for r in part]
        qts = {c.id: qtype_of_row(r) for c, r in zip(cases, part)}
        ev = evaluate_julia_arch(model, tok, cases, qts, device=device,
                                 batch_decisions=32, collect_rows=True)
        out_rows += [dict(r, family=rr.get("family", "text"),
                          language=rr.get("language", "en"),
                          group_id=rr.get("group_id", rr["id"]))
                     for r, rr in zip(ev["rows"], part)]
        correct += sum(r["correct"] for r in ev["rows"])
        n += ev["n"]
    return out_rows, (correct / n if n else None)


def main() -> None:
    t_start = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    check_secret_and_repos()  # FIRST: no quota burnt on a bad secret
    pip_install()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("E39 ABORT: no CUDA device assigned")
    if "T4" not in torch.cuda.get_device_name(0):
        raise SystemExit(f"E39 ABORT: expected a T4, got {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    import transformers

    env = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "transformers": transformers.__version__,
           "gpu": torch.cuda.get_device_name(0),
           "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)}
    (WORK / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    log(f"env: {env}")
    device = "cuda"
    find_bundle()  # fail fast on a missing attachment, before any training

    roots = download_data()
    verify_rows_sha(roots)
    from mksh_v.train_compare import (
        E38A_FAMILIES, E38B_FAMILIES, fixed_dev_half, load_e38_rows, rows_by_split,
    )

    rows = (load_e38_rows(roots["E38a"], families=E38A_FAMILIES)
            + load_e38_rows(roots["E38b"], families=E38B_FAMILIES))
    splits = {sp: rows_by_split(rows, sp, drop_fr=False) for sp in ("train", "dev", "test")}
    log(f"rows: train {len(splits['train'])} dev {len(splits['dev'])} "
        f"test {len(splits['test'])} (FR kept inside test, gated separately)")
    dev_half = fixed_dev_half(splits["dev"], seed=DEV_HALF_SEED, fraction=0.5)
    dev_sel = [r for r in splits["dev"] if r["id"] in dev_half]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    mk1_path = expected_checkpoint("mksh1_v12.safetensors")
    if mk1_path is None:
        raise SystemExit("E39 ABORT: mksh1_v12.safetensors (MK-SH-1 v1.2, sha ea66cdde...) "
                         "not attached via the gjusev/mkshv-e37-ckpt dataset")

    # OCR runs on CPUs while the GPU trains arm 1 (train+dev+test images only;
    # cal splits are not downloaded)
    ocr_thread = start_ocr_thread(
        [(str(image_path_of(r, roots)), r["language"])
         for sp in ("train", "dev", "test") for r in splits[sp]])

    # -- arm 1: A1-V seed 42 ---------------------------------------------------
    a1v = build_a1v(device)
    best_a1v = train_a1v(a1v, splits["train"], roots, dev_sel, device, seed=42, tag="s42")
    ocr_thread.join()
    if not OCR_TEXT:
        raise SystemExit("E39 ABORT: OCR produced nothing (check ocr_manifest.json)")

    # -- arm 2: OCR -> MK-SH-1 seed 42 ------------------------------------------
    from mokka_decisions.arch_julia import JuliaArchScorer
    from safetensors.torch import load_file

    mk1 = JuliaArchScorer("jhu-clsp/mmBERT-small").to(device)
    mk1.load_state_dict(load_file(str(mk1_path)))
    log(f"OCR arm init: MK-SH-1 v1.2 {mk1_path}")
    best_ocr = train_ocr(mk1, tok, splits["train"], roots, dev_sel, device, seed=42, tag="s42")

    finalist = "a1v" if best_a1v["dev_half_macro"] >= best_ocr["dev_half_macro"] else "ocr"
    log(f"finalist by dev-half macro: {finalist} "
        f"({best_a1v['dev_half_macro']:.4f} vs {best_ocr['dev_half_macro']:.4f})")

    test_nonfr = [r for r in splits["test"] if r["language"] != "fr"]
    test_fr = [r for r in splits["test"] if r["language"] == "fr"]

    # -- finalist seeds 43/44 (whichever arm won; each eval'd on test) ----------
    seed_reports = []
    for seed in (43, 44):
        if finalist == "a1v":
            del a1v
            torch.cuda.empty_cache()
            a1v = build_a1v(device)
            best = train_a1v(a1v, splits["train"], roots, dev_sel, device,
                             seed=seed, tag=f"s{seed}")
            _, acc_seed = eval_a1v_chunked(a1v, test_nonfr, roots, device)
        else:
            mk1 = JuliaArchScorer("jhu-clsp/mmBERT-small").to(device)
            mk1.load_state_dict(load_file(str(mk1_path)))
            best = train_ocr(mk1, tok, splits["train"], roots, dev_sel, device,
                             seed=seed, tag=f"s{seed}")
            _, acc_seed = eval_ocr_chunked(mk1, tok, test_nonfr, roots, device)
        seed_reports.append({"seed": seed, "test_nonfr_acc": round(acc_seed, 4), **best})
    # gates always use the seed-42 best weights
    load_best_weights(best_a1v["best_path"], a1v)
    a1v.to(device)
    load_best_weights(best_ocr["best_path"], mk1)
    mk1.to(device)

    # -- test: opened ONCE; non-FR gates, FR reported ---------------------------
    rows_a1v, acc_a1v = eval_a1v_chunked(a1v, test_nonfr, roots, device)
    rows_ocr, acc_ocr = eval_ocr_chunked(mk1, tok, test_nonfr, roots, device)
    for name, rr in (("a1v", rows_a1v), ("ocr", rows_ocr)):
        (WORK / f"eval_test_nonfr__{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rr), encoding="utf-8")
    fr_a1v_rows, acc_a1v_fr = eval_a1v_chunked(a1v, test_fr, roots, device)
    fr_ocr_rows, acc_ocr_fr = eval_ocr_chunked(mk1, tok, test_fr, roots, device)
    for name, rr in (("a1v", fr_a1v_rows), ("ocr", fr_ocr_rows)):
        (WORK / f"eval_test_fr__{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rr), encoding="utf-8")
    log(f"test non-FR: A1-V {acc_a1v:.4f} OCR {acc_ocr:.4f} | "
        f"FR: {acc_a1v_fr:.4f}/{acc_ocr_fr:.4f}")

    # -- G1: paired-by-episode bootstrap ----------------------------------------
    from mksh_v.train_compare import gate_report, paired_delta_ci

    by_id_ocr = {r["id"]: r for r in rows_ocr}
    delta_ci = paired_delta_ci(
        [r["correct"] for r in rows_a1v],
        [int(by_id_ocr[r["id"]]["correct"]) for r in rows_a1v],
        [r["group_id"] for r in rows_a1v], n_boot=1000, seed=42)

    # -- G2: E35 text sets, CPU float32, both arms -------------------------------
    bundle = find_bundle()
    man29 = json.loads((bundle / "e29-mix/manifest.json").read_text(encoding="utf-8"))
    want = set(man29["original_dev_ids"])
    task_rows = [json.loads(l) for l in (bundle / "data/processed/v1/eval/dev_eval.jsonl")
                 .read_text(encoding="utf-8").splitlines()
                 if l.strip() and json.loads(l)["id"] in want]
    text_sets = {}
    a1v.to("cpu")
    a1v.eval()
    mk1.to("cpu")
    mk1.eval()
    from mksh_v.arch_a1v import evaluate_a1v
    from mksh_v.train_compare import case_of_row

    for name, rr in (("testC", [json.loads(l) for l in (bundle / "e26-order/test_c_rows.jsonl")
                                 .read_text(encoding="utf-8").splitlines() if l.strip()]),
                     ("task", task_rows)):
        cases = [case_of_row(r) for r in rr]
        ev = evaluate_a1v(a1v, cases, None, device="cpu", batch_decisions=16,
                          max_state_tokens=MAX_STATE_TOKENS,
                          max_option_tokens=MAX_OPTION_TOKENS, collect_rows=True)
        txt_rows, acc_txt = eval_ocr_chunked(mk1, tok, rr, None, "cpu", chunk=400)
        with open(WORK / f"eval_text_{name}__both.jsonl", "w", encoding="utf-8") as fh:
            for r in ev["rows"]:
                fh.write(json.dumps({"arm": "a1v", **r}) + "\n")
            for r in txt_rows:
                fh.write(json.dumps({"arm": "ocr", **r}) + "\n")
        text_sets[name] = {"a1v": round(ev["accuracy"], 4), "ocr": round(acc_txt, 4)}
    gates = gate_report(delta_ci=delta_ci, text_sets=text_sets,
                        text_margin=TEXT_MARGIN, delta_lo_margin=DELTA_LO_MARGIN)

    # -- G3: ablation tripwire on the VISUAL arm (always A1-V, s42 best) ---------
    from mksh_v.train_compare import blank_like, degrade, swap_map_within_blocks

    a1v.to(device)
    swap_map = swap_map_within_blocks(
        [f'{r["family"]}|{r["language"]}' for r in test_nonfr])
    swap_rows, acc_swap = eval_a1v_chunked(a1v, test_nonfr, roots, device,
                                           image_index_map=swap_map)
    _, acc_blank = eval_a1v_chunked(a1v, test_nonfr, roots, device, transform=blank_like)
    _, acc_degr = eval_a1v_chunked(a1v, test_nonfr, roots, device, transform=degrade)
    g3 = {"bar": f"swap drop >= {SWAP_DROP_BAR:.2f}",
          "swap_drop": round(acc_a1v - acc_swap, 4), "swap_acc": round(acc_swap, 4),
          "blank_acc": round(acc_blank, 4), "degraded_acc": round(acc_degr, 4)}
    g3["passed"] = (acc_a1v - acc_swap) >= SWAP_DROP_BAR
    (WORK / "eval_ablations__a1v.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in swap_rows), encoding="utf-8")
    (WORK / "eval_ablations__a1v.json").write_text(json.dumps(g3, indent=2), encoding="utf-8")

    # -- G4: real eval (repo may not exist yet; skipped cleanly) -----------------
    g4 = {"status": "not_run",
          "reason": f"{REAL_EVAL_REPO} not available (E38c pending)"}
    real_rows_file = None
    try:
        from huggingface_hub import list_repo_files, snapshot_download

        files = list_repo_files(REAL_EVAL_REPO, repo_type="dataset")
        real_rows_file = next((f for f in files if f.endswith("rows.jsonl")), None)
    except Exception as exc:
        log(f"real eval skipped: {REAL_EVAL_REPO} not readable ({type(exc).__name__})")
    if real_rows_file:
        from huggingface_hub import snapshot_download

        real_root_path = Path(snapshot_download(
            repo_id=REAL_EVAL_REPO, repo_type="dataset",
            allow_patterns=["*.jsonl", "*.png", "*.jpg", "*.jpeg"]))
        real_rows = [json.loads(l) for l in (real_root_path / real_rows_file)
                     .read_text(encoding="utf-8").splitlines() if l.strip()]
        # real rows are not e38a/e38b-prefixed: point both keys at the real root
        real_root = {"E38a": real_root_path, "E38b": real_root_path}
        ocr_real_rows_sync(real_rows, real_root_path)
        real_rows_a1v, acc_real_a1v = eval_a1v_chunked(a1v, real_rows, real_root, device)
        real_rows_ocr, acc_real_ocr = eval_ocr_chunked(mk1.to(device), tok, real_rows,
                                                       real_root, device)
        for name, rr in (("a1v", real_rows_a1v), ("ocr", real_rows_ocr)):
            (WORK / f"eval_real__{name}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rr), encoding="utf-8")
        g4 = {"status": "completed", "repo": REAL_EVAL_REPO,
              "a1v": round(acc_real_a1v, 4), "ocr": round(acc_real_ocr, 4),
              "kill_bar": REAL_EVAL_KILL,
              "non_discriminating": bool(acc_real_a1v >= REAL_EVAL_KILL
                                         and acc_real_ocr >= REAL_EVAL_KILL)}
    (WORK / "e39_real_eval.json").write_text(json.dumps(g4, indent=2), encoding="utf-8")

    summary = {
        "status": "completed",
        "env": env,
        "steps": STEPS, "image_size": IMAGE_SIZE, "effective_batch": EFFECTIVE_BATCH,
        "finalist": finalist,
        "selection": {"a1v_s42": best_a1v, "ocr_s42": best_ocr,
                      "seed_reports": seed_reports},
        "test_nonfr": {"a1v": round(acc_a1v, 4), "ocr": round(acc_ocr, 4)},
        "test_fr": {"a1v": round(acc_a1v_fr, 4), "ocr": round(acc_ocr_fr, 4)},
        "delta_ci": delta_ci,
        "text_sets": text_sets,
        "gates": gates,
        "G3_ablations": g3,
        "G4_real": g4,
        "wall_s": round(time.time() - t_start, 1),
    }
    (WORK / "e39_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"E39 done in {summary['wall_s']}s; gates: {json.dumps(gates)}")


if __name__ == "__main__":
    main()
