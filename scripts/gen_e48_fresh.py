"""E48 fresh numeric set (criteria: docs/E48_CRITERIA_2026-10-25.md).

    python scripts/gen_e48_fresh.py   # -> data/e48/fresh_ft/{rows.jsonl, field_threshold/..., manifest.json}

The frozen E38a generator, unchanged, on the field_threshold test templates
with five new seeds (EN/ES/DE, 200 cases per seed and language). Ids are
renamed e38a- -> e48s{S}-; any image byte-identical to an E38a image aborts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_e38a import Renderer, git_commit, row_of, sha256  # noqa: E402
from mksh_v.gen_e38a import FAMILIES, TRAIN_LANGS, generate_split, make_theme  # noqa: E402

SEEDS = (4801, 4802, 4803, 4804, 4805)
FAMILY = "field_threshold"
OUT = ROOT / "data/e48/fresh_ft"


def rename(row: dict, seed: int) -> dict:
    pre = f"e48s{seed}-"
    out = dict(row)
    for k in ("id", "group_id", "cf_base_id"):
        if isinstance(out.get(k), str):
            assert out[k].startswith("e38a-"), (k, out[k])
            out[k] = pre + out[k].removeprefix("e38a-")
    return out


def main() -> int:
    old = set()
    for fam in FAMILIES:
        for line in (ROOT / f"data/e38a/{fam}/rows.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                old.add(json.loads(line)["image_sha256"])
    r = Renderer()
    rows = []
    try:
        for seed in SEEDS:
            for lang in TRAIN_LANGS:
                for gc in generate_split(FAMILY, "test", lang, seed=seed):
                    png = r.render(gc.body_html, make_theme(gc.theme_seed).font)
                    sha = sha256(png)
                    if sha in old:
                        raise SystemExit(f"E48 ABORT: image identical to an E38a image ({gc.case.id}, seed {seed})")
                    new_id = f"e48s{seed}-" + gc.case.id.removeprefix("e38a-")
                    rel = f"{FAMILY}/s{seed}/{new_id}"
                    (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
                    (OUT / f"{rel}.png").write_bytes(png)
                    rows.append(rename(row_of(gc, f"{rel}.png", "", sha), seed))
                print(f"seed {seed} {lang}: {len(rows)} rows", flush=True)
        versions = r.versions
    finally:
        r.close()
    ids = [x["id"] for x in rows]
    assert len(ids) == len(set(ids)) == 3000, len(ids)
    assert len({x["image_sha256"] for x in rows}) == len(rows), "duplicate images inside the fresh set"
    (OUT / "rows.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                                    encoding="utf-8")
    labels = {k: sum(x["target_option"] == k for x in rows) for k in ("true", "false")}
    manifest = {"criteria": "docs/E48_CRITERIA_2026-10-25.md", "family": FAMILY, "seeds": list(SEEDS),
                "languages": list(TRAIN_LANGS), "n": len(rows), "labels": labels,
                "generator_commit": git_commit(), "versions": versions,
                "rows_sha256": sha256((OUT / "rows.jsonl").read_bytes())}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
