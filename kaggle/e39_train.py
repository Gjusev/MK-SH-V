"""E39: compared training, synthetic-only (criteria: docs/E39_CRITERIA_*.md at
freeze). ONE GPU orchestrator + self-spawning per-GPU workers. NOT LAUNCHED
until the criteria are frozen and the user gives GO.

Priority order (frozen; the kernel executes in this order and cuts the tail
if the session wall budget runs out — (a)-(c) protect the scientific
question):
  (a) A1-V s42 at 1024 px + eval
  (b) OCR->MK-SH-1 s42 (the OCR+rules comparator is CPU, from ocrprep)
  (c) statistical comparison G1-G4 + per-family analysis
  (d) finalist seeds 43/44
  (e) resolution sweep 768/1024/1536 (inference) + optional 1536 training
  (f) jev-vlm-4b zero-shot reference + SmolVLM2 comparator

Tables carry a `regimen` column: task-trained (a1v, ocr, smol, dev-fitted
rules) vs zero-shot (jev-vlm-4b). No sentence compares A1-V with a zero-shot
model without that label. G4 interpretation rule (frozen): if G1 passes on
synthetic but the real-eval Delta(A1-V - OCR) CI includes 0 or is negative,
the result is reported as "synthetic-only advantage, no transfer evidence"
and no general visual-advantage claim is made.

Every trained arm mixes E35 text replay (1 of every 4 microbatches).
Selection is dev-only (frozen half, best checkpoint restored); test opens
once. G2 anchors to frozen MK-SH-1 v1.1 (testC 0.959 / task 0.780 after
-2 pp). Quota bills the SESSION wall at double rate.

Expected attachments: gjusev/mkshv-e37-ckpt (c3_final.safetensors,
mksh1_v12.safetensors), gjusev/mokka-decisions-bundle, gjusev/mkshv-e39-ocr
(ocr.jsonl + rules_predictions.jsonl). Secret HF_TOKEN checked FIRST, per
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
REAL_EVAL_REPO = "Gjusev/MK-SH-V-E38c"  # private; FIXED revision, never main
REAL_EVAL_REVISION = "98dde3d2b3dd0fd319ff44d059656a2d2864c66e"
REAL_EVAL_ROWS_SHA = {  # verified in-kernel; mismatch -> real eval not_run
    "cord/rows.jsonl": "5d0dbc9e7ba3b3301bbcc1d52ecc2fb352b281dcb72c76bbe21ad34e23ab3f09",
    "gui_odyssey/rows.jsonl": "dcdbe6170752738cbb4a72a7f4cafbb56f4b30177cdb79add4c0e2c5a3c2a8da",
}
# SmolVLM2 comparator (verified on the Hub 2026-10-14): pinned id+revision.
SMOL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SMOL_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
JEV_ID = "sidman007/jev-vlm-4b"  # ADAPTER repo with custom_code
JEV_REVISION = "8c56e46a2b47dee398bc71c39c355b9716dec21e"
STEPS = 2000
SMOKE_STEPS = 20
EFFECTIVE_BATCH = 32
ARM_MICROBATCH = {"a1v": 4, "ocr": 8, "smol": 4}
IMAGE_SIZE = 1024  # validated end-to-end by C4; 1536 has 100-step C1 evidence only
RES_SWEEP = (768, 1024, 1536)
TRAIN_1536_EST_S = 11_580  # C1: 5.79 s/step x 2000 (optional phase (e) train)
MAX_WALL_S = 41_400  # 11.5 h of the 12 h session cap
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
REGIMEN = {"a1v": "task-trained", "ocr": "task-trained", "smol": "task-trained",
           "rules": "task-trained (dev-fitted rules)", "jev": "zero-shot"}


def log(msg: str) -> None:
    print(f"[e39 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Amendment A1 (pre-measurement): Kaggle's secrets service answered 400 on
# every HF_TOKEN read, so each private Hub repo also ships as a private Kaggle
# dataset with byte-identical files. Attached copies are preferred; the frozen
# rows-sha checks are unchanged; the Hub path stays as fallback.
KAGGLE_DATA = {"Gjusev/MK-SH-V-E38a": "mkshv-e38a", "Gjusev/MK-SH-V-E38b": "mkshv-e38b",
               "Gjusev/MK-SH-V-E38c": "mkshv-e38c"}


def kaggle_copy(repo: str) -> Path | None:
    """Root of the attached Kaggle copy of ``repo`` (dir holding manifest.json).
    Follows symlinks: mounts may be links, and Path.rglob does not follow them."""
    inp = Path("/kaggle/input")
    slug = KAGGLE_DATA[repo]
    for base in [inp / slug, *inp.glob(f"*/{slug}"), *inp.glob(f"*/*/{slug}")]:
        if base.is_dir():
            for dirpath, _dirs, files in os.walk(base, followlinks=True):
                if "manifest.json" in files:
                    return Path(dirpath)
    return None


_TESS_RES = {"en": "eng", "de": "deu", "es": "spa", "fr": "fra"}


def _fmt_amount(x: float, lang: str) -> str:
    """Same thousands/decimal format the E38a generator renders (EN 1,234.56)."""
    s = f"{x:,.2f}"
    return s if lang == "en" else s.replace(",", "_").replace(".", ",").replace("_", ".")


# Process-pool workers must live at module level (A1: they were nested in
# main(), which ProcessPoolExecutor cannot pickle).
def _legibility_job(job):
    import pytesseract
    from PIL import Image

    path, lang, total, res = job
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, res / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.BILINEAR)
    text = pytesseract.image_to_string(img, lang=_TESS_RES.get(lang, "eng"))
    return (path, res), _fmt_amount(total, lang) in text


def _real_ocr_job(p_lang):
    import pytesseract
    from PIL import Image

    p, lang = p_lang
    return pytesseract.image_to_string(Image.open(p), lang=_TESS_RES.get(lang, "eng"))


class _RealEvalSkip(Exception):
    """G4 cannot run on the frozen real-eval data (reason in the message)."""


def ocr_state(state: str, text: str) -> str:
    """A1: the OCR arm sees the case text PLUS the OCR of the image (the same
    information A1-V gets), never the OCR alone: screen states carry the goal
    or ticket that the image must be checked against."""
    return f"{state}\n[OCR]\n{text}"


# ---------------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------------
def check_secret_and_repos() -> None:
    if all(kaggle_copy(r) for r in E38_REPOS.values()):
        log("transport A1: E38a/E38b from attached Kaggle datasets; HF secret not needed")
        return
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
                    "huggingface_hub>=0.34", "pytesseract>=0.3",
                    "num2words"], check=True)  # A1 item 15: SmolVLM processor needs it
    subprocess.run(["apt-get", "-qq", "install", "-y", "tesseract-ocr",
                    "tesseract-ocr-deu", "tesseract-ocr-spa", "tesseract-ocr-fra"],
                   check=False)


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


# A1 item 14: the v5 run trained both s42 arms to completion and then hit the
# wall in phase (c). Its s42 runs ship as the private dataset below and are
# resumed (sha-verified) instead of retrained: one shot per seed.
RESUME_DATA = "mkshv-e39-v5-runs"
RESUME_SHA = {  # sha256 of the v5 best.pt files (pinned at A1 item 14)
    "a1v_s42": "274dfef220028969a296fd87d371b3b5ca3f19e286745dec7590b72b36d65201",
    "ocr_s42": "10ee0ba648da082b7530989109d0dd62e2d144fc40f478c481538038b7b8118d",
}


def resume_s42_runs() -> bool:
    """Copy the attached v5 s42 runs into WORK/runs; False if not attached.
    An attached copy that fails its sha check aborts (never silently retrains)."""
    import shutil

    inp = Path("/kaggle/input")
    base = next((b for b in [inp / RESUME_DATA, *inp.glob(f"*/{RESUME_DATA}"),
                             *inp.glob(f"*/*/{RESUME_DATA}")] if b.is_dir()), None)
    if base is None:
        return False
    found = {}
    for dirpath, _dirs, files in os.walk(base, followlinks=True):
        tag = Path(dirpath).name
        if tag in RESUME_SHA and "best.pt" in files and "summary.json" in files:
            found[tag] = Path(dirpath)
    if set(found) != set(RESUME_SHA):
        raise SystemExit(f"E39 ABORT: {RESUME_DATA} attached but runs {sorted(found)} "
                         f"!= {sorted(RESUME_SHA)}")
    for tag, src in found.items():
        got = hashlib.sha256((src / "best.pt").read_bytes()).hexdigest()
        if got != RESUME_SHA[tag]:
            raise SystemExit(f"E39 ABORT: {tag}/best.pt sha {got} != pinned {RESUME_SHA[tag]}")
        dst = WORK / "runs" / tag
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
        log(f"resumed {tag} from {src} (sha {got[:16]})")
    return True


# A1 item 15: v6 aborted in the SmolVLM2 smoke (its processor needs
# `num2words`, missing from pip_install) AFTER the finalist seeds were already
# trained. The trained seeds ship as this dataset and are resumed (one shot
# per seed); the summary-writing may never again be killed by a reported,
# non-gate phase.
RESUME_DATA_V6 = "mkshv-e39-v6-runs"
RESUME_SHA_V6 = {
    "a1v_s43": "077d678737ec910cee74e313bf0272d5ea97d39310be312c6ec4725e3971ac9e",
    "a1v_s44": "0f4ebeed1412a81581c949acec2ab90e320ab64e5b873af7f5d11d9e90373d2e",
    "a1v_s45_sz1536": "aa7146365c8f31876f9ad2a1a50ce1276486993bbc1294a551cf2098464ad2c2",
}


def resume_runs(dataset: str, pins: dict) -> list:
    """Copy sha-verified runs from an attached dataset into WORK/runs.
    Returns the resumed tags ([] when the dataset is not attached); an
    attached run that fails its sha check aborts (never silently retrains)."""
    import shutil

    inp = Path("/kaggle/input")
    base = next((b for b in [inp / dataset, *inp.glob(f"*/{dataset}"),
                             *inp.glob(f"*/*/{dataset}")] if b.is_dir()), None)
    if base is None:
        return []
    found = {}
    for dirpath, _dirs, files in os.walk(base, followlinks=True):
        tag = Path(dirpath).name
        if tag in pins and "best.pt" in files and "summary.json" in files:
            found[tag] = Path(dirpath)
    for tag, src in found.items():
        got = hashlib.sha256((src / "best.pt").read_bytes()).hexdigest()
        if got != pins[tag]:
            raise SystemExit(f"E39 ABORT: {tag}/best.pt sha {got} != pinned {pins[tag]}")
        dst = WORK / "runs" / tag
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
        log(f"resumed {tag} from {dataset} (sha {got[:16]})")
    not_here = sorted(set(pins) - set(found))
    if not_here:
        log(f"resume {dataset}: not attached: {not_here}")
    return sorted(found)


def load_legibility() -> dict[tuple, bool]:
    """{(row id, res): legible} from legibility.jsonl next to the OCR cache."""
    for cand in Path("/kaggle/input").rglob("legibility.jsonl"):
        out = {}
        for line in cand.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["id"], int(r["res"]))] = bool(r["legible"])
        log(f"legibility loaded: {len(out)} (row, res) entries from {cand}")
        return out
    log("legibility.jsonl not attached -> phase-(e) error analysis not_run")
    return {}


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
        local = kaggle_copy(repo)
        if local is not None:
            roots[key] = local
            log(f"{repo}: attached Kaggle copy at {local}")
            continue
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
    """C3 recipe rerun (E35 mix, seed 42): 3 phases x 5 epochs, fresh AdamW
    per phase, rng 42^0xE35^phase — mirroring kaggle/e37_c3.py exactly."""
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

    def __init__(self, device, model_id: str = SMOL_ID, revision: str = SMOL_REVISION):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision).to(device)
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
    return dataclasses.replace(c, state=ocr_state(c.state, text)) if text is not None else c


def run_worker(args) -> None:
    """--arm --seed --gpu --steps [--size]; writes runs/<tag>/{best.pt,...}."""
    device = "cuda"
    size = args.size or IMAGE_SIZE
    tag = f"{args.arm}_s{args.seed}" + (f"_sz{size}" if size != IMAGE_SIZE else "")
    out_dir = WORK / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    roots = download_data()
    verify_rows_sha(roots)
    splits = load_e38_rows(roots)
    bundle = find_bundle()
    ocr = load_ocr_cache() if args.arm == "ocr" else {}
    from mksh_v.train_compare import (case_of_row, fixed_dev_half,
                                      macro_family_accuracy, qtype_of_row)

    dev_half = fixed_dev_half(splits["dev"], seed=DEV_HALF_SEED, fraction=0.5)
    dev_sel = [r for r in splits["dev"] if r["id"] in dev_half]
    replay_rows = load_replay_mix(bundle)
    replay_cases = [case_of_row(r) for r in replay_rows]
    replay_qt = {c.id: qtype_of_row(r) for c, r in zip(replay_cases, replay_rows)}
    train_rows = splits["train"]

    micro = ARM_MICROBATCH[args.arm]
    accum = EFFECTIVE_BATCH // micro

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
        enc = rep_enc = None
        opt_groups = model.parameter_groups(2e-5, 1e-4)
    else:  # smol
        model = SmolScorer(device)
        train_cases = [case_of_row(r) for r in train_rows]
        enc = rep_enc = tok = None
        opt_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                       "lr": 2e-5}]

    def eval_dev():
        from mksh_v.arch_a1v import evaluate_a1v
        from mokka_decisions.arch_julia import evaluate_julia_arch

        if isinstance(model, SmolScorer):
            model.model.eval()
        else:
            model.eval()
        rows = []
        with torch.no_grad():
            if args.arm == "a1v":
                ev = evaluate_a1v(model, [case_of_row(r) for r in dev_sel],
                                  [open_image(r, roots) for r in dev_sel],
                                  device=device, max_side=size, batch_decisions=16,
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
                for s in range(0, len(dev_sel), 16):
                    part = dev_sel[s:s + 16]
                    cases = [case_of_row(r) for r in part]
                    imgs = [open_image(r, roots) for r in part]
                    logits = model.batch_logits(cases, imgs)
                    probs = logits.softmax(-1)
                    for j, r in enumerate(part):
                        k = len(cases[j].options)
                        gold = cases[j].option_ids.index(cases[j].target_option)
                        rows.append({"correct": int(int(probs[j, :k].argmax().item()) == gold)})
        recs = [dict(r, family=rr["family"]) for r, rr in zip(rows, dev_sel)]
        return macro_family_accuracy(recs), sum(r["correct"] for r in rows) / len(rows)

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
    from mksh_v.arch_a1v import forward_a1v

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
                        device, max_side=size)
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
                    qts = {c.id: qtype_of_row(train_rows[i]) for c, i in zip(chunk, idx)}
                batch = collate_julia(tok, chunk, qts, 1024, 256, device)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**{k: batch[k] for k in keys})
                    loss = torch.nn.functional.cross_entropy(
                        logits.float(), batch["targets"], ignore_index=-100) / accum
            else:  # smol
                if use_replay:
                    idx = next_replay(micro)
                    chunk = [replay_cases[i] for i in idx]
                    imgs = None
                else:
                    idx = next_synthetic(micro)
                    chunk = [train_cases[i] for i in idx]
                    imgs = [open_image(train_rows[i], roots) for i in idx]
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
            log(f"{args.arm}[{tag}] step {step}/{args.steps} dev-half macro {macro:.4f}")
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
         "arm": args.arm, "seed": args.seed, "steps": args.steps, "size": size},
        indent=2), encoding="utf-8")
    log(f"worker done: {tag} best {best}")


def spawn(arm: str, seed: int, gpu: int, steps: int, size: int | None = None) -> tuple:
    """Worker as subprocess with output to a FILE (a PIPE nobody reads while
    another proc runs would fill its buffer and deadlock the worker)."""
    tag = f"{arm}_s{seed}" + (f"_sz{size}" if size and size != IMAGE_SIZE else "")
    log_dir = WORK / "runs" / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"worker_gpu{gpu}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--arm", arm, "--seed", str(seed), "--gpu", str(gpu),
           "--steps", str(steps)]
    if size and size != IMAGE_SIZE:
        cmd += ["--size", str(size)]
    fh = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
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
    parser.add_argument("--size", type=int, default=None)
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

    def budget_ok(need_s: float = 0.0) -> bool:
        return (time.time() - t_start) + need_s < MAX_WALL_S

    skipped: list[dict] = []

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

    if resume_s42_runs():
        log("A1 item 14: s42 runs of both arms resumed from v5 (sha verified); "
            "no smoke, no retraining (one shot per seed)")
    else:
        # smoke-first (cheap): the two gated arms; smol smokes in phase (f)
        run_and_log(spawn("a1v", 42, 0, SMOKE_STEPS), "smoke-a1v")
        run_and_log(spawn("ocr", 42, 1, SMOKE_STEPS), "smoke-ocr")

        # ---- (a)+(b): A1-V s42 on gpu0, OCR->MK-SH-1 s42 on gpu1 ------------
        p_a1v = spawn("a1v", 42, 0, STEPS)
        run_and_log(spawn("ocr", 42, 1, STEPS), "ocr-s42")
        run_and_log(p_a1v, "a1v-s42")

    def summary_of(tag: str) -> dict:
        return json.loads((WORK / "runs" / tag / "summary.json").read_text(encoding="utf-8"))

    s42 = {arm: summary_of(f"{arm}_s42") for arm in ("a1v", "ocr")}
    finalist = "a1v" if s42["a1v"]["dev_half_macro"] >= s42["ocr"]["dev_half_macro"] else "ocr"
    log(f"finalist={finalist} (dev macro a1v {s42['a1v']['dev_half_macro']} vs "
        f"ocr {s42['ocr']['dev_half_macro']})")

    # ---- (c): statistical comparison G1-G4 + per-family -----------------------
    from mksh_v.arch_a1v import NeoMMEMarkerScorer, evaluate_a1v
    from mksh_v.train_compare import (case_of_row, gate_report,
                                      paired_delta_ci, qtype_of_row)

    ocr = load_ocr_cache()
    device = "cuda:0"
    a1v = NeoMMEMarkerScorer().to(device)
    a1v.ensure_markers(32)  # match the workers/C3 tables (16 would break the strict best.pt load)
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
    smol = None  # created in phase (f) only

    test_nonfr = [r for r in splits["test"] if r["language"] != "fr"]
    test_fr = [r for r in splits["test"] if r["language"] == "fr"]

    # A1 item 14: the legibility probe (input of the descriptive phase-(e)
    # error analysis) ran 1,800 tesseract calls on Kaggle CPU (~1 min each,
    # measured). It is precomputed locally with the same `_legibility_job` and
    # shipped as legibility.jsonl next to the OCR cache; keyed by (row id, res).
    ft_test_rows = [r for r in test_nonfr if r["family"] == "field_threshold"
                    and r.get("total") is not None]
    LEGIBILITY = load_legibility()

    def eval_rows(arm, rows, *, transform=None, image_index_map=None, max_side=None):
        """Per-case rows + accuracy of one arm on rows (chunked, RAM-safe)."""
        out_rows, correct, n = [], 0, 0
        chunk = 400 if arm != "ocr" else 800
        max_side = max_side or IMAGE_SIZE
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
                                  device=device, max_side=max_side, batch_decisions=16,
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
            else:  # smol
                rows_out = []
                hits = cnt = 0
                with torch.no_grad():
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
                            ok = int(int(probs[j, :k].argmax().item()) == gold)
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

    test_rows, test_acc = {}, {}
    for arm in ("a1v", "ocr"):
        test_rows[arm], test_acc[arm] = eval_rows(arm, test_nonfr)
        (WORK / f"eval_test_nonfr__{arm}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in test_rows[arm]), encoding="utf-8")
    for arm in ("a1v", "ocr"):
        fr_rows, fr_acc = eval_rows(arm, test_fr)
        (WORK / f"eval_test_fr__{arm}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in fr_rows), encoding="utf-8")
        test_acc[f"{arm}_fr"] = fr_acc

    # G1 (roadmap): A1-V vs OCR->MK-SH-1, paired by episode
    by_id_ocr = {r["id"]: r for r in test_rows["ocr"]}
    delta_ci = paired_delta_ci(
        [r["correct"] for r in test_rows["a1v"]],
        [int(by_id_ocr[r["id"]]["correct"]) for r in test_rows["a1v"]],
        [r["group_id"] for r in test_rows["a1v"]], n_boot=1000, seed=42)
    g1_by_family = {}
    for family in sorted({r["family"] for r in test_rows["a1v"]}):
        sub = [r for r in test_rows["a1v"] if r["family"] == family]
        g1_by_family[family] = paired_delta_ci(
            [r["correct"] for r in sub],
            [int(by_id_ocr[r["id"]]["correct"]) for r in sub],
            [r["group_id"] for r in sub], n_boot=1000, seed=42)

    # rules comparator (frozen rules from ocrprep): per-family claim rule
    rules_rows = {}
    for cand in Path("/kaggle/input").rglob("rules_predictions.jsonl"):
        for line in cand.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["split"] == "test":
                    rules_rows[r["id"]] = r
    rules_vs_a1v = {}
    for family in sorted({r["family"] for r in test_rows["a1v"]}):
        sub = [r for r in test_rows["a1v"]
               if r["family"] == family and r["id"] in rules_rows]
        if not sub:
            continue
        ci = paired_delta_ci(
            [int(rules_rows[r["id"]]["pred"] == rules_rows[r["id"]]["target"]) for r in sub],
            [r["correct"] for r in sub],
            [r["group_id"] for r in sub], n_boot=1000, seed=42)
        # claim rule: rules match-or-beat A1-V when the CI lower bound of
        # (rules - a1v) is >= 0 -> no visual-advantage claim in that family
        rules_vs_a1v[family] = {**ci, "claim_blocked": ci["ci95"][0] >= 0.0}

    # G2: E35 text sets, CPU float32, gated arms vs frozen bars
    bundle = find_bundle()
    man29 = json.loads((bundle / "e29-mix/manifest.json").read_text(encoding="utf-8"))
    want = set(man29["original_dev_ids"])
    task_rows = [json.loads(l) for l in (bundle / "data/processed/v1/eval/dev_eval.jsonl")
                 .read_text(encoding="utf-8").splitlines()
                 if l.strip() and json.loads(l)["id"] in want]
    a1v.to("cpu"); a1v.eval()
    mk1.to("cpu"); mk1.eval()
    text_sets = {}
    for name, rr in (("testC", [json.loads(l) for l in (bundle / "e26-order/test_c_rows.jsonl")
                                 .read_text(encoding="utf-8").splitlines() if l.strip()]),
                     ("task", task_rows)):
        cases = [case_of_row(r) for r in rr]
        ev = evaluate_a1v(a1v, cases, None, device="cpu", batch_decisions=16,
                          max_state_tokens=MAX_STATE_TOKENS,
                          max_option_tokens=MAX_OPTION_TOKENS)
        qts = {c.id: qtype_of_row(r) for c, r in zip(cases, rr)}
        ev2 = evaluate_julia_arch(mk1, tok, cases, qts, device="cpu", batch_decisions=32)
        text_sets[name] = {"a1v": round(ev["accuracy"], 4),
                           "ocr": round(ev2["accuracy"], 4)}
    gates = gate_report(delta_ci=delta_ci, text_sets=text_sets, text_bars=TEXT_BARS,
                        delta_lo_margin=DELTA_LO_MARGIN)

    # G3: ablation tripwire on the visual arm (a1v s42 best)
    from mksh_v.train_compare import blank_like, degrade, swap_map_within_blocks

    a1v.to(device)
    swap_map = swap_map_within_blocks(
        [f'{r["family"]}|{r["language"]}' for r in test_nonfr])
    _, acc_swap = eval_rows("a1v", test_nonfr, image_index_map=swap_map)
    _, acc_blank = eval_rows("a1v", test_nonfr, transform=blank_like)
    _, acc_degr = eval_rows("a1v", test_nonfr, transform=degrade)
    g3 = {"bar": f"swap drop >= {SWAP_DROP_BAR:.2f}",
          "swap_drop": round(test_acc["a1v"] - acc_swap, 4),
          "swap_acc": round(acc_swap, 4), "blank_acc": round(acc_blank, 4),
          "degraded_acc": round(acc_degr, 4)}
    g3["passed"] = (test_acc["a1v"] - acc_swap) >= SWAP_DROP_BAR
    (WORK / "eval_ablations__a1v.json").write_text(json.dumps(g3, indent=2), encoding="utf-8")

    # G4: real eval at the FIXED revision with rows-sha verification; any
    # mismatch -> not_run (never trained-on, never swapped). CORD and
    # GUI-Odyssey go in the MAIN table; frozen interpretation rule computed.
    g4 = {"status": "not_run", "reason": f"{REAL_EVAL_REPO} not available"}
    real_ctx = None
    rows_files = []
    real_local = kaggle_copy(REAL_EVAL_REPO)
    if real_local is not None:
        rows_files = [rel for rel in REAL_EVAL_ROWS_SHA if (real_local / rel).exists()]
        log(f"transport A1: {REAL_EVAL_REPO} from attached Kaggle copy at {real_local}")
    else:
        try:
            from huggingface_hub import list_repo_files

            files = list_repo_files(REAL_EVAL_REPO, repo_type="dataset",
                                    revision=REAL_EVAL_REVISION)
            rows_files = [f for f in files if f in REAL_EVAL_ROWS_SHA]
        except Exception as exc:
            log(f"real eval skipped: {REAL_EVAL_REPO}@{REAL_EVAL_REVISION[:8]} "
                f"not readable ({type(exc).__name__})")
    # A1: G4 can never take the run down; any failure is recorded instead.
    try:
        if rows_files:
            if real_local is not None:
                real_root_path = real_local
            else:
                from huggingface_hub import snapshot_download

                real_root_path = Path(snapshot_download(
                    repo_id=REAL_EVAL_REPO, repo_type="dataset",
                    revision=REAL_EVAL_REVISION,
                    allow_patterns=["*.jsonl", "*.png", "*.jpg", "*.jpeg"]))
            sha_ok = True
            for rel, want_sha in REAL_EVAL_ROWS_SHA.items():
                f = real_root_path / rel
                if not f.exists():
                    sha_ok = False
                    g4 = {"status": "not_run",
                          "reason": f"missing {rel} at revision {REAL_EVAL_REVISION[:8]}"}
                    break
                got = hashlib.sha256(f.read_bytes()).hexdigest()
                if got != want_sha:
                    sha_ok = False
                    g4 = {"status": "not_run",
                          "reason": f"rows sha mismatch for {rel}: {got}"}
                    break
            if not sha_ok:
                raise _RealEvalSkip(g4["reason"])
            log("E38c rows sha256 verified (cord + gui_odyssey)")
            if sha_ok:
                real_rows = []
                for rel in REAL_EVAL_ROWS_SHA:
                    fam_dir = rel.rsplit("/", 1)[0]
                    for l in (real_root_path / rel).read_text(encoding="utf-8").splitlines():
                        if l.strip():
                            r = json.loads(l)
                            # A1: E38c rows store images relative to their family
                            # dir (cord/, gui_odyssey/); every reader joins on the
                            # repo root, so the path is made root-relative here.
                            r["image"] = f"{fam_dir}/{r['image']}"
                            real_rows.append(r)
                missing = [r["id"] for r in real_rows if not (real_root_path / r["image"]).exists()]
                if missing:
                    raise _RealEvalSkip(f"{len(missing)} E38c images missing, e.g. {missing[:3]}")
                real_root = {"E38a": real_root_path, "E38b": real_root_path}
            # A1 item 14: real-image OCR comes from the frozen OCR cache (the
            # same 862 E38c images, keyed "<repo>|<root-relative path>"); it
            # was recomputed here on Kaggle CPU and stalled v5 at the wall.
            real_ocr = {r["id"]: ocr.get(f"{REAL_EVAL_REPO}|{r['image']}") for r in real_rows}
            no_ocr = [i for i, t in real_ocr.items() if t is None]
            if no_ocr:
                raise _RealEvalSkip(f"{len(no_ocr)} E38c rows without cached OCR, e.g. {no_ocr[:3]}")
            log(f"E38c OCR from cache: {len(real_ocr)} rows")
            real_ctx = {"rows": real_rows, "root": real_root,
                        "ocr": real_ocr, "root_path": real_root_path}

            def eval_real(arm, rows, *, max_side=None):
                max_side = max_side or IMAGE_SIZE
                out_rows, correct, n = [], 0, 0
                for s in range(0, len(rows), 200):
                    part = rows[s:s + 200]
                    if arm == "a1v":
                        imgs = [open_image(r, real_root) for r in part]
                        ev = evaluate_a1v(a1v, [case_of_row(r) for r in part], imgs,
                                          device=device, max_side=max_side,
                                          batch_decisions=16,
                                          max_state_tokens=MAX_STATE_TOKENS,
                                          max_option_tokens=MAX_OPTION_TOKENS,
                                          collect_rows=True)
                        out_rows += ev["rows"]
                        correct += sum(r["correct"] for r in ev["rows"])
                        n += ev["n"]
                    elif arm == "ocr":
                        cases = []
                        for r in part:
                            c = case_of_row(r)
                            text = real_ocr.get(r["id"])
                            if text is not None:
                                c = dataclasses.replace(c, state=ocr_state(c.state, text))
                            cases.append(c)
                        qts = {c.id: qtype_of_row(r) for c, r in zip(cases, part)}
                        ev = evaluate_julia_arch(mk1.to(device), tok, cases, qts,
                                                 device=device, batch_decisions=32,
                                                 collect_rows=True)
                        out_rows += ev["rows"]
                        correct += sum(r["correct"] for r in ev["rows"])
                        n += ev["n"]
                    else:  # smol on real images
                        with torch.no_grad():
                            for s2 in range(0, len(part), 16):
                                sub = part[s2:s2 + 16]
                                cases = [case_of_row(r) for r in sub]
                                imgs = [open_image(r, real_root) for r in sub]
                                logits = smol.batch_logits(cases, imgs)
                                for j, r in enumerate(sub):
                                    k = len(cases[j].options)
                                    gold = cases[j].option_ids.index(cases[j].target_option)
                                    ok = int(int(logits[j, :k].argmax().item()) == gold)
                                    out_rows.append({"id": r["id"], "group_id": r.get("group_id", r["id"]),
                                                     "correct": ok})
                                    correct += ok
                                    n += 1
                return out_rows, (correct / n if n else None)

            accs_real, real_rows_by_arm = {}, {}
            for arm in ("a1v", "ocr"):
                real_rows_arm, acc = eval_real(arm, real_rows)
                accs_real[arm] = round(acc, 4)
                real_rows_by_arm[arm] = real_rows_arm
                (WORK / f"eval_real__{arm}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in real_rows_arm), encoding="utf-8")

            def real_delta(rows_a1v, rows_ocr):
                by_id = {r["id"]: r for r in rows_ocr}
                return paired_delta_ci(
                    [r["correct"] for r in rows_a1v],
                    [int(by_id[r["id"]]["correct"]) for r in rows_a1v],
                    [r.get("group_id", r["id"]) for r in rows_a1v],
                    n_boot=1000, seed=42)

            # frozen interpretation rule: real Delta(a1v - ocr) CI...
            real_delta_ci = real_delta(real_rows_by_arm["a1v"], real_rows_by_arm["ocr"])
            # ...plus the PRE-REGISTERED sensitivity: (i) all episodes vs (ii)
            # task_done excluding purchase-goal episodes; if the readings differ
            # in sign or in whether the CI excludes 0, (ii) governs
            real_by_id_a1v = {r["id"]: r for r in real_rows_by_arm["a1v"]}
            purchase_ids = {r["id"] for r in real_rows
                            if "purchase" in r.get("state", "").lower()}
            strict_a1v = [r for r in real_rows_by_arm["a1v"] if r["id"] not in purchase_ids]
            strict_ocr = [r for r in real_rows_by_arm["ocr"] if r["id"] not in purchase_ids]
            real_delta_ci_strict = real_delta(strict_a1v, strict_ocr)

            def excludes_zero(ci):
                return ci["ci95"][0] > 0 or ci["ci95"][1] < 0

            def sign_of(ci):
                return 1 if ci["delta"] > 0 else (-1 if ci["delta"] < 0 else 0)

            differ = (sign_of(real_delta_ci) != sign_of(real_delta_ci_strict)
                      or excludes_zero(real_delta_ci) != excludes_zero(real_delta_ci_strict))
            governing = "strict_no_purchase" if differ else "full"
            # criteria: OCR+rules is "reported on test, real and per family";
            # A1 adds the real half (CORD field_threshold), same paired CI and
            # claim rule as on synthetic test
            rules_vs_a1v_real = {}
            for family in sorted({r.get("family") for r in real_rows}):
                fam_ids = {r["id"] for r in real_rows if r.get("family") == family}
                sub = [r for r in real_rows_by_arm["a1v"]
                       if r["id"] in fam_ids and r["id"] in rules_rows]
                if sub:
                    ci = paired_delta_ci(
                        [int(rules_rows[r["id"]]["pred"] == rules_rows[r["id"]]["target"])
                         for r in sub],
                        [r["correct"] for r in sub],
                        [r.get("group_id", r["id"]) for r in sub], n_boot=1000, seed=42)
                    rules_vs_a1v_real[family] = {**ci, "n": len(sub),
                                                 "claim_blocked": ci["ci95"][0] >= 0.0}
            g4 = {"status": "completed", "repo": REAL_EVAL_REPO,
                  "rules_vs_a1v_real": rules_vs_a1v_real,
                  "revision": REAL_EVAL_REVISION,
                  "n_purchase_episodes_excluded": len(purchase_ids), **accs_real,
                  "delta_ci_a1v_vs_ocr": real_delta_ci,
                  "delta_ci_a1v_vs_ocr_strict": real_delta_ci_strict,
                  "sensitivity_readings_differ": differ,
                  "governing_reading": governing,
                  "governing_delta_ci": real_delta_ci_strict if differ else real_delta_ci,
                  "kill_bar": REAL_EVAL_KILL,
                  "non_discriminating": all(v >= REAL_EVAL_KILL for v in accs_real.values())}
    except _RealEvalSkip as exc:
        g4, real_ctx = {"status": "not_run", "reason": str(exc)}, None
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed silently
        log(f"G4 failed: {type(exc).__name__}: {exc}")
        g4, real_ctx = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, None
    (WORK / "e39_real_eval.json").write_text(json.dumps(g4, indent=2), encoding="utf-8")

    # ---- (d): finalist seeds 43/44 --------------------------------------------
    seed_reports = [dict(s42[a], arm=a, regimen=REGIMEN[a]) for a in ("a1v", "ocr")]
    resumed = resume_runs(RESUME_DATA_V6, RESUME_SHA_V6)  # A1 item 15
    if budget_ok():
        # A1 item 15: a reported (non-gate) phase may never abort the run.
        try:
            if {f"{finalist}_s43", f"{finalist}_s44"} <= set(resumed):
                log("seeds 43/44 resumed from v6 (no retraining)")
            else:
                # A1 item 14: one seed per GPU in parallel (the quota arithmetic
                # assumed this; the code ran them one after the other)
                p43 = spawn(finalist, 43, 0, STEPS)
                run_and_log(spawn(finalist, 44, 1, STEPS), f"{finalist}-s44")
                run_and_log(p43, f"{finalist}-s43")
            for seed in (43, 44):
                if finalist == "a1v":
                    load_best_weights(WORK / f"runs/a1v_s{seed}/best.pt", a1v)
                    a1v.to(device)
                else:
                    load_best_weights(WORK / f"runs/ocr_s{seed}/best.pt", mk1)
                    mk1.to(device)
                _, acc_seed = eval_rows(finalist, test_nonfr)
                seed_reports.append({"arm": finalist, "regimen": REGIMEN[finalist],
                                     "seed": seed, "test_nonfr_acc": round(acc_seed, 4),
                                     **summary_of(f"{finalist}_s{seed}")})
            load_best_weights(WORK / "runs/a1v_s42/best.pt", a1v)
            a1v.to(device)
            load_best_weights(WORK / "runs/ocr_s42/best.pt", mk1)
            mk1.to(device)
        except SystemExit as exc:
            skipped.append({"phase": "(d) seeds", "reason": str(exc)[:400]})
        except Exception as exc:  # noqa: BLE001 - recorded, never silent
            skipped.append({"phase": "(d) seeds",
                            "reason": f"{type(exc).__name__}: {exc}"[:400]})
    else:
        skipped.append({"phase": "(d) seeds", "reason": "wall budget"})

    # ---- (e): resolution sweep (inference) + field_threshold error analysis --
    sweep, error_analysis = {}, {}
    if budget_ok():
        for res in RES_SWEEP:
            _, acc_dev_res = eval_rows("a1v", splits["dev"], max_side=res)
            sweep[f"dev_{res}"] = round(acc_dev_res, 4)
        ft_errors = [r for r in test_rows["a1v"]
                     if r["family"] == "field_threshold" and not r["correct"]]
        ft_total_by_id = {r["id"]: r for r in ft_test_rows}
        for res in RES_SWEEP:
            if not LEGIBILITY:
                error_analysis = {"status": "not_run", "reason": "legibility.jsonl not attached"}
                break
            illegible = 0
            for r in ft_errors:
                base = ft_total_by_id.get(r["id"])
                if base is None:
                    continue
                if not LEGIBILITY.get((base["id"], res), True):
                    illegible += 1
            error_analysis[res] = {
                "a1v_errors": len(ft_errors),
                "errors_with_illegible_total": illegible,
                "errors_with_legible_total": len(ft_errors) - illegible,
                "legibility_rate_all_ft": round(
                    sum(LEGIBILITY.values()) / len(LEGIBILITY) if LEGIBILITY else 0.0, 4)}
        (WORK / "field_threshold_error_analysis.json").write_text(
            json.dumps(error_analysis, indent=2), encoding="utf-8")
        if real_ctx:
            for res in RES_SWEEP:
                _, acc = eval_real("a1v", real_ctx["rows"], max_side=res)
                sweep[f"real_{res}"] = round(acc, 4)
        log(f"resolution sweep: {sweep}")
    else:
        skipped.append({"phase": "(e) sweep", "reason": "wall budget"})

    train_1536 = {"status": "not_run", "reason": "wall budget"}
    if budget_ok(TRAIN_1536_EST_S + 1200):
        try:  # A1 item 15: reported phase, never fatal
            if "a1v_s45_sz1536" in resumed:
                log("s45@1536 resumed from v6 (no retraining)")
            else:
                run_and_log(spawn("a1v", 45, 1, STEPS, size=1536), "a1v-s45-sz1536")
            s1536 = summary_of("a1v_s45_sz1536")
            load_best_weights(WORK / "runs/a1v_s45_sz1536/best.pt", a1v)
            a1v.to(device)
            _, acc_1536 = eval_rows("a1v", test_nonfr, max_side=1536)
            train_1536 = {"status": "completed", "regimen": "task-trained (descriptive)",
                          "test_nonfr_acc": round(acc_1536, 4), **s1536}
            load_best_weights(WORK / "runs/a1v_s42/best.pt", a1v)
            a1v.to(device)
        except SystemExit as exc:
            train_1536 = {"status": "failed", "reason": str(exc)[:400]}
        except Exception as exc:  # noqa: BLE001
            train_1536 = {"status": "failed",
                          "reason": f"{type(exc).__name__}: {exc}"[:400]}
    (WORK / "train_1536.json").write_text(json.dumps(train_1536, indent=2), encoding="utf-8")

    # ---- (f): jev-vlm-4b zero-shot reference + SmolVLM2 comparator ------------
    smol_section = {"status": "not_run", "reason": "wall budget"}
    delta_ci_smol = None
    if budget_ok(9_000):
        try:  # A1 item 15: reported comparator, never fatal
            run_and_log(spawn("smol", 42, 1, SMOKE_STEPS), "smoke-smol")
            run_and_log(spawn("smol", 42, 1, STEPS), "smol-s42")
            smol = SmolScorer(device)
            load_best_weights(WORK / "runs/smol_s42/best.pt", smol.model)
            smol_rows, smol_acc = eval_rows("smol", test_nonfr)
            (WORK / "eval_test_nonfr__smol.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in smol_rows), encoding="utf-8")
            fr_rows_s, fr_acc_s = eval_rows("smol", test_fr)
            (WORK / "eval_test_fr__smol.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in fr_rows_s), encoding="utf-8")
            # text sets for smol (reported, not gating)
            smol_text = {}
            smol.model.to("cpu"); smol.model.eval()
            for name, rr in (("testC", [json.loads(l) for l in (bundle / "e26-order/test_c_rows.jsonl")
                                         .read_text(encoding="utf-8").splitlines() if l.strip()]),
                             ("task", task_rows)):
                cases = [case_of_row(r) for r in rr]
                hits = cnt = 0
                with torch.no_grad():
                    for s2 in range(0, len(cases), 16):
                        logits = smol.batch_logits(cases[s2:s2 + 16], [None] * 16)
                        for j, c in enumerate(cases[s2:s2 + 16]):
                            hits += int(int(logits[j, :len(c.options)].argmax().item())
                                        == c.option_ids.index(c.target_option))
                            cnt += 1
                smol_text[name] = round(hits / cnt, 4)
            smol.model.to(device)
            smol_real = None
            if real_ctx:
                _, smol_real = eval_real("smol", real_ctx["rows"])
            by_id_smol = {r["id"]: r for r in smol_rows}
            delta_ci_smol = paired_delta_ci(
                [r["correct"] for r in test_rows["a1v"]],
                [int(by_id_smol[r["id"]]["correct"]) for r in test_rows["a1v"]],
                [r["group_id"] for r in test_rows["a1v"]], n_boot=1000, seed=42)
            smol_section = {"status": "completed", "regimen": REGIMEN["smol"],
                            "dev_half_macro": summary_of("smol_s42")["dev_half_macro"],
                            "test_nonfr_acc": round(smol_acc, 4),
                            "test_fr_acc": round(fr_acc_s, 4),
                            "text_sets": smol_text,
                            "real_acc": round(smol_real, 4) if smol_real is not None else None}
        except SystemExit as exc:
            smol_section = {"status": "not_run", "reason": str(exc)[:400]}
        except Exception as exc:  # noqa: BLE001 - recorded, never silent
            smol_section = {"status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}"[:400]}
    else:
        skipped.append({"phase": "(f) smol", "reason": "wall budget"})

    jev_section = {"status": "not_run", "reason": "wall budget"}
    if budget_ok(3_000):
        from mksh_v.train_compare import accuracy_ci_by_group, ece_15, sha_sample

        jev_sample = sha_sample(test_nonfr, per_family=200)
        (WORK / "jev_sample_ids.json").write_text(json.dumps(
            {"rule": "sha256(id)-ordered first 200 per family, non-FR test",
             "ids": [r["id"] for r in jev_sample]}), encoding="utf-8")
        try:
            from transformers import AutoModel, AutoProcessor

            processor = AutoProcessor.from_pretrained(
                JEV_ID, revision=JEV_REVISION, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                JEV_ID, revision=JEV_REVISION, trust_remote_code=True,
                dtype=torch.float16).to("cuda:1")
            model.eval()

            def ask(case, image):
                opts = "\n".join(f"{case.options[i].id}: {case.options[i].description}"
                                 for i in range(len(case.options)))
                prompt = (f"state: {case.state}\nquestion: {case.question}\n"
                          f"options:\n{opts}\n"
                          "Answer with the option id and its probability (System One).")
                try:
                    feats = processor(text=[prompt], images=[image],
                                      return_tensors="pt").to("cuda:1")
                    with torch.no_grad():
                        out = model.generate(**feats, max_new_tokens=48)
                    text_out = processor.batch_decode(out, skip_special_tokens=True)[0]
                    low = text_out.lower()
                    pred = next((o.id for o in case.options if o.id.lower() in low), None)
                    import re as _re

                    m = _re.search(r"(0\.\d+)", low)
                    return pred, (float(m.group(1)) if m else None), text_out[:200]
                except Exception as exc:
                    return None, None, f"{type(exc).__name__}: {exc}"[:200]

            from PIL import Image

            jev_rows = []
            for r in jev_sample:
                pred, conf, raw = ask(case_of_row(r), open_image(r, roots))
                jev_rows.append({"id": r["id"], "family": r["family"],
                                 "group_id": r["group_id"], "pred": pred, "conf": conf,
                                 "target": r["target_option"],
                                 "correct": int(pred == r["target_option"]), "raw": raw})
            real_jev = []
            if real_ctx:
                for r in real_ctx["rows"]:
                    pred, conf, raw = ask(
                        case_of_row(r),
                        Image.open(real_ctx["root_path"] / r["image"]).convert("RGB"))
                    real_jev.append({"id": r["id"], "family": r.get("family", "real"),
                                     "group_id": r.get("group_id", r["id"]),
                                     "pred": pred, "conf": conf,
                                     "target": r["target_option"],
                                     "correct": int(pred == r["target_option"])})
            # per-family accuracy + CI, ECE, and paired delta vs a1v on the sample
            by_family = {}
            for r in jev_rows:
                by_family.setdefault(r["family"], []).append(r)
            per_family = {}
            for fam, rs in sorted(by_family.items()):
                per_family[fam] = accuracy_ci_by_group(
                    [r["correct"] for r in rs], [r["group_id"] for r in rs])
            confs = [r["conf"] for r in jev_rows if r["conf"] is not None]
            corrects = [r["correct"] for r in jev_rows if r["conf"] is not None]
            a1v_on_sample = {r["id"]: r for r in test_rows["a1v"]}
            delta_jev = paired_delta_ci(
                [r["correct"] for r in jev_rows],
                [int(a1v_on_sample[r["id"]]["correct"]) for r in jev_rows],
                [r["group_id"] for r in jev_rows], n_boot=1000, seed=42)
            jev_section = {"status": "completed", "regimen": REGIMEN["jev"],
                           "model": JEV_ID, "n_sample": len(jev_rows),
                           "sample_acc_ci": accuracy_ci_by_group(
                               [r["correct"] for r in jev_rows],
                               [r["group_id"] for r in jev_rows]),
                           "per_family": per_family,
                           "ece": round(ece_15(confs, corrects), 4) if confs else None,
                           "delta_jev_minus_a1v_ci": delta_jev,
                           "real_acc": (round(sum(r["correct"] for r in real_jev)
                                              / len(real_jev), 4) if real_jev else None),
                           "framing": "existing ~15x larger model WITHOUT adaptation; "
                                      "reference, not an equal-footing comparison"}
            (WORK / "eval_jev.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in jev_rows), encoding="utf-8")
            if real_jev:
                (WORK / "eval_jev_real.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in real_jev), encoding="utf-8")
        except Exception as exc:
            jev_section = {"status": "not_run",
                           "reason": f"load failed ({type(exc).__name__}): {exc}"[:500]}
    else:
        skipped.append({"phase": "(f) jev", "reason": "wall budget"})

    # transfer-evidence flag (frozen G4 interpretation rule; the GOVERNING
    # sensitivity reading decides when the two differ)
    transfer_evidence = None
    if g4.get("status") == "completed":
        ci = g4.get("governing_delta_ci", {}).get("ci95")
        if gates["G1_visual_advantage"]["passed"] and ci and ci[1] <= 0:
            transfer_evidence = "synthetic-only advantage, no transfer evidence"
        elif gates["G1_visual_advantage"]["passed"]:
            transfer_evidence = "transfer supported"

    summary = {
        "status": "completed", "env": env,
        "priority_order": "(a) a1v+eval (b) ocr+rules (c) G1-G4 (d) seeds "
                          "(e) sweep+1536 (f) jev+smol",
        "steps": STEPS, "image_size": IMAGE_SIZE,
        "effective_batch": EFFECTIVE_BATCH, "replay_every": REPLAY_EVERY,
        "finalist": finalist,
        "selection": seed_reports,
        "test_nonfr": {a: round(v, 4) for a, v in test_acc.items()
                       if not a.endswith("_fr")},
        "test_fr": {a[:-3]: round(v, 4) for a, v in test_acc.items() if a.endswith("_fr")},
        "delta_ci": delta_ci, "g1_by_family": g1_by_family,
        "rules_vs_a1v": rules_vs_a1v,
        "text_sets": text_sets, "gates": gates,
        "G3_ablations": g3, "G4_real": g4,
        "transfer_evidence": transfer_evidence,
        "resolution_sweep": sweep,
        "field_threshold_error_analysis": error_analysis,
        "train_1536": train_1536,
        "comparator_smol": {**smol_section,
                            "delta_ci": delta_ci_smol,
                            "claim_rule": "if ci95 upper bound < 0 the report and "
                                          "model card declare it; no 'best small "
                                          "visual model' claim"},
        "comparator_jev": jev_section,
        "regimen_labels": REGIMEN,
        "skipped_phases": skipped,
        "wall_s": round(time.time() - t_start, 1),
    }
    (WORK / "e39_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"E39 done in {summary['wall_s']}s; gates: {json.dumps(gates)}; "
        f"skipped: {skipped}")


if __name__ == "__main__":
    main()
