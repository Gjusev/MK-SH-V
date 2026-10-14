"""E39 OCR precompute (CPU-ONLY Kaggle kernel; no GPU quota).

Tesseracts the ORIGINAL PNGs of E38a+E38b (train/dev/test; plus E38c if the
repo exists) and writes `ocr.jsonl` — one {"key", "text"} per image, key =
"<repo_id>|<repo-relative image path>" so any consumer can map its local
snapshot paths back to keys — plus `ocr_manifest.json` with engine versions,
wall time and the sha256 of ocr.jsonl.

Transport chain (needs GO, executed by the coordinator): download this
kernel's output via CLI, upload as the private Kaggle dataset
`gjusev/mkshv-e39-ocr`, attach it to `gjusev/mkshv-e39-train`.

The HF secret is checked FIRST (cheap, before any work). The token is never
printed, logged or written to files.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

WORK = Path("/kaggle/working")
REPOS = ("Gjusev/MK-SH-V-E38a", "Gjusev/MK-SH-V-E38b", "Gjusev/MK-SH-V-E38c")


def log(msg: str) -> None:
    print(f"[ocrprep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_secret() -> None:
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("HF_TOKEN")  # never printed/logged
    if not token:
        raise SystemExit("OCRPREP ABORT: HF_TOKEN secret missing on this notebook "
                         "(Add-ons -> Secrets).")
    os.environ["HF_TOKEN"] = token
    from huggingface_hub import HfApi

    HfApi().whoami()
    log("HF auth OK")


def pip_install() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "huggingface_hub>=0.34", "pillow>=10.1", "pytesseract>=0.3"],
                   check=True)
    subprocess.run(["apt-get", "-qq", "install", "-y",
                    "tesseract-ocr", "tesseract-ocr-deu", "tesseract-ocr-spa",
                    "tesseract-ocr-fra"], check=False)
    out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
    log("tesseract: " + ((out.stdout or out.stderr).splitlines() or ["?"])[0])


_TESS_LANG = {"en": "eng", "de": "deu", "es": "spa", "fr": "fra"}


def _ocr_one(job):
    import pytesseract
    from PIL import Image

    path, lang, key = job
    try:
        text = pytesseract.image_to_string(Image.open(path), lang=_TESS_LANG.get(lang, "eng"))
        return key, text, None
    except Exception as exc:  # measured failure, never dropped silently
        return key, "", f"{type(exc).__name__}: {exc}"


def main() -> None:
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    check_secret()
    pip_install()

    from huggingface_hub import list_repo_files, snapshot_download

    jobs = []
    repo_langs = {}
    for repo in REPOS:
        try:
            files = list_repo_files(repo, repo_type="dataset")
        except Exception as exc:
            log(f"{repo}: not readable ({type(exc).__name__}) -> skipped")
            continue
        rows_files = [f for f in files if f.endswith("rows.jsonl")]
        if not rows_files:
            log(f"{repo}: no rows.jsonl -> skipped")
            continue
        allow = ["*/rows.jsonl"] + [p for p in ("*/train/*.png", "*/dev/*.png",
                                                "*/test/*.png", "*.png", "*.jpg")]
        root = Path(snapshot_download(repo_id=repo, repo_type="dataset",
                                      allow_patterns=allow))
        for rf in rows_files:
            fam_dir = rf.rsplit("/", 1)[0]  # e.g. "doc_type" for doc_type/rows.jsonl
            for line in (root / rf).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") not in ("train", "dev", "test") or not row.get("image"):
                    continue
                img_path = root / fam_dir / row["image"]
                if not img_path.exists():
                    img_path = root / row["image"]  # some repos root images directly
                if not img_path.exists():
                    log(f"missing image for {row['id']}: {img_path}")
                    continue
                jobs.append((str(img_path), row.get("language", "en"),
                             f"{repo}|{img_path.relative_to(root).as_posix()}"))
        log(f"{repo}: queued {sum(1 for j in jobs if j[2].startswith(repo))} images")

    log(f"OCR over {len(jobs)} images with 4 CPU workers")
    results, errors = {}, {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for key, text, err in pool.map(_ocr_one, jobs, chunksize=16):
            if err:
                errors[key] = err
            results[key] = text

    ocr_path = WORK / "ocr.jsonl"
    with open(ocr_path, "w", encoding="utf-8") as fh:
        for key in sorted(results):
            fh.write(json.dumps({"key": key, "text": results[key]}, ensure_ascii=False) + "\n")
    out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
    manifest = {
        "n_images": len(results),
        "n_errors": len(errors),
        "errors": errors,
        "wall_s": round(time.time() - t0, 1),
        "tesseract": (out.stdout or out.stderr).splitlines()[0],
        "ocr_jsonl_sha256": hashlib.sha256(ocr_path.read_bytes()).hexdigest(),
    }
    (WORK / "ocr_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done: {len(results)} texts, {len(errors)} errors, sha {manifest['ocr_jsonl_sha256'][:16]}")


if __name__ == "__main__":
    main()
