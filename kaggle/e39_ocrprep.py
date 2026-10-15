"""E39 OCR precompute + deterministic-rules comparator (CPU-ONLY Kaggle
kernel; no GPU quota).

Per image of E38a+E38b (+E38c if the repo exists), train/dev/test splits:
Tesseract on the ORIGINAL PNG -> `ocr.jsonl` rows {"key", "text", "conf"}
(key = "<repo_id>|<repo-relative path>"; conf = mean word confidence from
image_to_data). Then the FROZEN rules of `mksh_v.rules_baseline` run over
every row of the three rule families (doc_type, field_threshold,
form_complete) -> `rules_predictions.jsonl` ({"id","family","split","pred",
"target"}). The rules were tuned ONLY on the E38a dev split offline; this
kernel re-reports their dev accuracy as a frozen-code verification.

Transport chain (needs GO; CLI with ABSOLUTE -p paths on Windows):
download this kernel's output, upload as the private Kaggle dataset
`gjusev/mkshv-e39-ocr`, attach it to `gjusev/mkshv-e39-train`.

The HF secret is checked FIRST. The token is never printed/logged/written.
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
# A1 item 10: the single 29,562-image run hit Kaggle's 12 h CPU session wall
# and wrote nothing (results were only written at the end). The OCR now runs
# as one kernel per dataset (REPOS/OUTPUT_PREFIX are set per generated
# variant) and flushes results incrementally, so a wall hit keeps the work.
REPOS = ("Gjusev/MK-SH-V-E38a", "Gjusev/MK-SH-V-E38b", "Gjusev/MK-SH-V-E38c")
OUTPUT_PREFIX = ""  # "e38a_" / "e38b_" / "e38c_" in the split variants


def log(msg: str) -> None:
    print(f"[ocrprep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Amendment A1 (pre-measurement): Kaggle's secrets service answered 400 on
# every HF_TOKEN read, so each private Hub repo also ships as a private Kaggle
# dataset with byte-identical files; attached copies are preferred.
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


def check_secret() -> None:
    if all(kaggle_copy(r) for r in REPOS):
        log("transport A1: all repos from attached Kaggle datasets; HF secret not needed")
        return
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
        img = Image.open(path).convert("RGB")
        tl = _TESS_LANG.get(lang, "eng")
        text = pytesseract.image_to_string(img, lang=tl)
        data = pytesseract.image_to_data(img, lang=tl, output_type=pytesseract.Output.DICT)
        confs = [c for c, w in zip(data["conf"], data["text"])
                 if w.strip() and c is not None and c >= 0]
        conf = round(sum(confs) / len(confs), 2) if confs else 0.0
        return key, text, conf, None
    except Exception as exc:  # measured failure, never dropped silently
        return key, "", 0.0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    check_secret()
    pip_install()

    from huggingface_hub import list_repo_files, snapshot_download

    jobs = []
    job_keys: set[str] = set()
    rows_by_repo: dict[str, list[dict]] = {}
    for repo in REPOS:
        local = kaggle_copy(repo)
        if local is not None:
            root = local
            rows_files = sorted(p.relative_to(root).as_posix() for p in root.rglob("rows.jsonl"))
            log(f"{repo}: attached Kaggle copy at {root} ({len(rows_files)} rows files)")
        else:
            try:
                files = list_repo_files(repo, repo_type="dataset")
            except Exception as exc:
                log(f"{repo}: not readable ({type(exc).__name__}) -> skipped")
                continue
            rows_files = [f for f in files if f.endswith("rows.jsonl")]
            if not rows_files:
                log(f"{repo}: no rows.jsonl -> skipped")
                continue
            root = Path(snapshot_download(
                repo_id=repo, repo_type="dataset",
                allow_patterns=["*/rows.jsonl", "*/train/*.png", "*/dev/*.png",
                                "*/test/*.png", "*.png", "*.jpg"]))
        repo_rows = []
        for rf in rows_files:
            fam_dir = rf.rsplit("/", 1)[0]
            for line in (root / rf).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") not in ("train", "dev", "test") or not row.get("image"):
                    continue
                img_path = root / fam_dir / row["image"]
                if not img_path.exists():
                    img_path = root / row["image"]
                if not img_path.exists():
                    log(f"missing image for {row['id']}: {img_path}")
                    continue
                # A1: ONE key per image, shared by OCR and rules (E38c rows
                # store family-relative paths, so row["image"] is not the key)
                key = f"{repo}|{img_path.relative_to(root).as_posix()}"
                row["_ocr_key"] = key
                repo_rows.append(row)
                if key not in job_keys:  # A1: each image OCR'd once
                    job_keys.add(key)
                    # A1: real images (E38c) are English/Indonesian whatever the
                    # question language; synthetic ones are rendered in it
                    lang = "en" if repo.endswith("E38c") else row.get("language", "en")
                    jobs.append((str(img_path), lang, key))
        rows_by_repo[repo] = repo_rows
        log(f"{repo}: {len(repo_rows)} rows, {len(jobs)} total OCR jobs")

    log(f"OCR over {len(jobs)} images with 4 CPU workers")
    ocr, errors = {}, {}
    ocr_path = WORK / f"{OUTPUT_PREFIX}ocr.jsonl"
    ocr_fh = open(ocr_path, "w", encoding="utf-8")  # A1 item 10: incremental
    with ProcessPoolExecutor(max_workers=4) as pool:
        done = 0
        for key, text, conf, err in pool.map(_ocr_one, jobs, chunksize=4):
            if err:
                errors[key] = err
            ocr[key] = (text, conf)
            ocr_fh.write(json.dumps({"key": key, "text": text, "conf": conf},
                                    ensure_ascii=False) + "\n")
            done += 1
            if done % 200 == 0:  # survive a session wall with partial output
                ocr_fh.flush()
                log(f"OCR progress: {done}/{len(jobs)}")
    ocr_fh.close()
    log(f"OCR written: {len(ocr)} entries -> {ocr_path.name}")

    # -- frozen deterministic rules over the rule families ---------------------
    from mksh_v.rules_baseline import RULE_FAMILIES, predict_rule

    preds = []
    dev_stats: dict[str, dict] = {}
    for repo, repo_rows in rows_by_repo.items():
        for row in repo_rows:
            if row.get("family") not in RULE_FAMILIES:
                continue
            key = row["_ocr_key"]
            if key not in ocr:
                continue
            text, conf = ocr[key]
            pred = predict_rule(row["family"], row.get("question", ""),
                                text, conf, row.get("language", "en"))
            preds.append({"id": row["id"], "family": row["family"],
                          "split": row["split"], "pred": pred,
                          "target": row["target_option"]})
            if row["split"] == "dev":
                st = dev_stats.setdefault(row["family"], {"n": 0, "ok": 0})
                st["n"] += 1
                st["ok"] += int(pred == row["target_option"])
    rules_path = WORK / f"{OUTPUT_PREFIX}rules_predictions.jsonl"
    with open(rules_path, "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
    dev_report = {f: {"n": s["n"], "acc": round(s["ok"] / s["n"], 4) if s["n"] else None}
                  for f, s in dev_stats.items()}
    manifest = {
        "n_images": len(ocr), "n_errors": len(errors), "errors": errors,
        "wall_s": round(time.time() - t0, 1),
        "tesseract": (out.stdout or out.stderr).splitlines()[0],
        "ocr_jsonl_sha256": hashlib.sha256(ocr_path.read_bytes()).hexdigest(),
        "rules_predictions_sha256": hashlib.sha256(rules_path.read_bytes()).hexdigest(),
        "rules_dev_verification": dev_report,
    }
    (WORK / "ocr_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done: {len(ocr)} texts, {len(errors)} errors; rules dev: {dev_report}; "
        f"sha {manifest['ocr_jsonl_sha256'][:16]}")


if __name__ == "__main__":
    main()
