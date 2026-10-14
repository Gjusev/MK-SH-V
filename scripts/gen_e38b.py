"""Render the E38b dataset (criteria: docs/E38B_CRITERIA_2026-10-14.md).

    python scripts/gen_e38b.py --dev            # seed-999 preview -> artifacts/e38b-dev
    python scripts/gen_e38b.py                  # full dataset -> data/e38b
    python scripts/gen_e38b.py --determinism    # G3: re-render 50 sampled cases per family
    python scripts/gen_e38b.py --contact        # contact sheets -> artifacts/e38b-contact
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_e38a import Renderer, git_commit, row_of, sha256  # noqa: E402
from mksh_v.gen_e38a import COUNTS, LANGS, TRAIN_LANGS  # noqa: E402
from mksh_v.gen_e38b import FAMILIES, generate_split  # noqa: E402
from mksh_v.gen_e38a import make_theme  # noqa: E402


def all_cases(seed: int):
    for fam in FAMILIES:
        for split in COUNTS:
            for lang in (LANGS if split == "test" else TRAIN_LANGS):
                yield fam, generate_split(fam, split, lang, seed=seed)


def generate(out: Path, seed: int, limit: int | None = None) -> None:
    r = Renderer()
    t0, n_total = time.time(), 0
    manifest = {"criteria": "docs/E38B_CRITERIA_2026-10-14.md", "seed": seed, "generator_commit": git_commit(),
                "versions": r.versions, "fonts_sha256": r.font_shas, "families": {}}
    rows_by_fam: dict[str, list] = {f: [] for f in FAMILIES}
    try:
        for fam, cases in all_cases(seed):
            for gc in cases[:limit] if limit else cases:
                rel = f"{fam}/{gc.case.split}/{gc.case.id}"
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                png = r.render(gc.body_html, make_theme(gc.theme_seed).font)
                (out / f"{rel}.png").write_bytes(png)
                (out / f"{rel}.html").write_text(gc.body_html, encoding="utf-8")
                rows_by_fam[fam].append(row_of(gc, f"{rel}.png", f"{rel}.html", sha256(png)))
                n_total += 1
                if n_total % 500 == 0:
                    print(f"{n_total} cases, {time.time() - t0:.0f}s", flush=True)
    finally:
        r.close()
    for fam, rows in rows_by_fam.items():
        p = out / fam / "rows.jsonl"
        p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
        manifest["families"][fam] = {"rows": f"{fam}/rows.jsonl", "rows_sha256": sha256(p.read_bytes()),
                                     "n": len(rows)}
    manifest["render_seconds"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done: {n_total} cases in {manifest['render_seconds']}s -> {out}")


def determinism(out: Path) -> dict:
    """G3: fresh browser, 50 sampled cases per family (sample seed 7), byte-identical PNGs."""
    r = Renderer()
    result = {}
    try:
        for fam in FAMILIES:
            rows = [json.loads(l) for l in (out / fam / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
            sample = random.Random(7).sample(rows, 50)
            same = sum(sha256(r.render((out / x["html"]).read_text(encoding="utf-8"),
                                       make_theme(x["theme_seed"]).font)) == x["image_sha256"] for x in sample)
            result[fam] = {"sampled": 50, "identical": same, "passed": same == 50}
    finally:
        r.close()
    result["passed"] = all(v["passed"] for v in result.values())
    (out / "g3_determinism.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return result


def contact(out: Path, dest: Path, per_sheet: int = 12) -> None:
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    for fam in FAMILIES:
        rows = [json.loads(l) for l in (out / fam / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        for lang in LANGS:
            pick = [x for x in rows if x["language"] == lang and x["split"] in ("dev", "test")][:per_sheet]
            if not pick:
                continue
            thumbs = []
            for x in pick:
                im = Image.open(out / x["image"]).convert("RGB")
                im.thumbnail((400, 520))
                thumbs.append(im)
            sheet = Image.new("RGB", (4 * 410, 3 * 530), "white")
            for i, im in enumerate(thumbs):
                sheet.paste(im, ((i % 4) * 410, (i // 4) * 530))
            sheet.save(dest / f"{fam}_{lang}.png")
            (dest / f"{fam}_{lang}.txt").write_text(
                "\n".join(f"{i}: {x['target_option']}  {x['question']}" for i, x in enumerate(pick)),
                encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="seed-999 preview, 8 cases per split file")
    ap.add_argument("--determinism", action="store_true")
    ap.add_argument("--contact", action="store_true")
    a = ap.parse_args()
    if a.dev:
        out = ROOT / "artifacts/e38b-dev"
        generate(out, seed=999, limit=8)
        contact(out, out / "contact")
    elif a.determinism:
        return 0 if determinism(ROOT / "data/e38b")["passed"] else 1
    elif a.contact:
        contact(ROOT / "data/e38b", ROOT / "artifacts/e38b-contact")
    else:
        generate(ROOT / "data/e38b", seed=38)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
