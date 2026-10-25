"""E43 inference: per-case raw probabilities for a rows.jsonl, with the E42
pipeline unchanged (evaluate_a1v, fp32, batch 16, longest side 1,024 px,
700/96 tokens), on CPU. Checkpoints are sha256-verified. Output is the E42
per-case schema, written incrementally so an interrupted run resumes.

    python scripts/e43_infer.py --seed 42 --rows data/e43/gui_cal/rows.jsonl \
        --root data/e43 --out runs/e43/percase_s42__gui_cal.jsonl [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SHA = {  # E41 selected checkpoints (docs/E42_CRITERIA_2026-10-22.md)
    42: "e4b4f6f5a56fedae18886e378cd9fb8f84afb1a77536b168a2fdfb990d9641a5",
    43: "30f2eb429c5b25dca00e1ef53fd509232997c34799a5a22fa1c77200619c3819",
}
IMAGE_SIZE, MAX_STATE_TOKENS, MAX_OPTION_TOKENS, BATCH = 1024, 700, 96, 16


def load_model(seed: int, ckpt: Path | None = None, sha: str | None = None):
    import torch
    from mksh_v.arch_a1v import NeoMMEMarkerScorer

    ckpt = ckpt or ROOT / f"runs/e41/runs/a1v_real_s{seed}/best.pt"
    want = sha or SHA[seed]
    got = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"E43 ABORT: checkpoint sha {got} != {want}")
    model = NeoMMEMarkerScorer()
    model.ensure_markers(32)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model"])
    model.eval()
    return model, got


def main() -> int:
    import torch
    from PIL import Image
    from mksh_v.arch_a1v import evaluate_a1v
    from mksh_v.train_compare import case_of_row

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="image root for row['image']")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt", type=Path, help="other checkpoint (default: E41 of --seed); needs --sha")
    ap.add_argument("--sha", help="pinned sha256 of --ckpt (e.g. from the kernel summary)")
    ap.add_argument("--upscale", type=int, default=0,
                    help="E46 input rule: upscale to and cap at this longest side (0 = off, 1,024 cap)")
    a = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    rows = [json.loads(l) for l in a.rows.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    done = set()
    if a.out.exists():
        done = {json.loads(l)["id"] for l in a.out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"seed {a.seed}: {len(rows)} rows, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return 0
    if a.ckpt and not a.sha:
        raise SystemExit("--ckpt needs --sha")
    model, sha = load_model(a.seed, a.ckpt, a.sha)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with a.out.open("a", encoding="utf-8") as fh:
        for s in range(0, len(todo), 64):
            part = todo[s : s + 64]
            cases = [case_of_row(r) for r in part]
            imgs = [Image.open(a.root / r["image"]).convert("RGB") for r in part]
            with torch.inference_mode():
                ev = evaluate_a1v(model, cases, imgs, device="cpu", max_side=a.upscale or IMAGE_SIZE,
                                  upscale_to=a.upscale or None, batch_decisions=BATCH, max_state_tokens=MAX_STATE_TOKENS,
                                  max_option_tokens=MAX_OPTION_TOKENS, collect_rows=True)
            for r, er in zip(part, ev["rows"]):
                if er["id"] != r["id"]:
                    raise SystemExit(f"E43 ABORT: row order {er['id']} != {r['id']}")
                fh.write(json.dumps({"id": r["id"], "family": r.get("family", ""),
                                     "group_id": r.get("group_id", r["id"]),
                                     "probs": [round(p, 6) for p in er["probabilities"]],
                                     "gold": er["gold_idx"], "pred": er["pred_idx"]}) + "\n")
            fh.flush()
            n = s + len(part)
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el / n:.2f} s/case  eta {el / n * (len(todo) - n) / 60:.0f} min",
                  flush=True)
    print(f"done seed {a.seed} sha {sha[:12]} in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
