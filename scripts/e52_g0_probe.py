"""E52 G0 pre-freeze probe: does a bottom-band crop beat the full image for
reading the printed total on CORD train receipts?

    .venv/Scripts/python.exe -I scripts/e52_g0_probe.py

Runs the released-lineage s43 checkpoint (sha-pinned) on the 716 CORD-train
digit-position rows of data/e47/bind/rows.jsonl, twice per row: full image
(U1024 rule) and bottom-40% band crop (then U1024). Gate (frozen in the E52
criteria the probe precedes): band accuracy >= full accuracy + 10 pp on >= 200
rows, else the crop arm is dropped. Writes runs/e52/g0_probe.json.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CKPT = ROOT / "checkpoints/e51/best_s43.pt"
SHA = "afcbeaa45a4c033b5e7d3f033b85fdda2642c0fef78397fcecc5bef53cd83c29"
IMG_ROOT = ROOT / "data/e41"
BAND = 0.40  # bottom fraction kept


def main() -> int:
    import torch
    from PIL import Image
    from e43_infer import load_model
    from mksh_v.arch_a1v import evaluate_a1v
    from mksh_v.train_compare import case_of_row

    got = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    assert got == SHA, got
    rows = [json.loads(l) for l in (ROOT / "data/e47/bind/rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("digit_q") and str(r.get("image", "")).startswith("cord/")]
    assert len(rows) >= 200, len(rows)

    model, _ = load_model(43, CKPT, SHA)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = {"n": len(rows), "variants": {}}
    t0 = time.time()
    for variant in ("full", "band"):
        correct = 0
        per_digit_q = {}
        for s in range(0, len(rows), 64):
            part = rows[s : s + 64]
            cases = [case_of_row(r) for r in part]
            if variant == "full":
                imgs = [Image.open(IMG_ROOT / r["image"]).convert("RGB") for r in part]
            else:
                ims = []
                for r in part:
                    im = Image.open(IMG_ROOT / r["image"]).convert("RGB")
                    w, h = im.size
                    ims.append(im.crop((0, int(h * (1 - BAND)), w, h)))
                imgs = ims
            with torch.inference_mode():
                ev = evaluate_a1v(model, cases, imgs, device="cpu", max_side=1024,
                                  upscale_to=1024, batch_decisions=16,
                                  max_state_tokens=700, max_option_tokens=96,
                                  collect_rows=True)
            for r, er in zip(part, ev["rows"]):
                ok = int(er["pred_idx"] == er["gold_idx"])
                correct += ok
                d = per_digit_q.setdefault(r["digit_q"], [0, 0])
                d[0] += ok
                d[1] += 1
            if s % 256 == 0:
                print(f"{variant} {s + len(part)}/{len(rows)} acc {correct / (s + len(part)):.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        out["variants"][variant] = {
            "accuracy": round(correct / len(rows), 4),
            "by_digit_q": {k: [v[0], v[1], round(v[0] / v[1], 3)] for k, v in sorted(per_digit_q.items())},
        }
    f, b = out["variants"]["full"]["accuracy"], out["variants"]["band"]["accuracy"]
    out["delta_pp"] = round((b - f) * 100, 2)
    out["crop_arm_included"] = bool(b - f >= 0.10 and len(rows) >= 200)
    d = ROOT / "runs/e52"
    d.mkdir(exist_ok=True)
    (d / "g0_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("full", f, "band", b, "delta_pp", out["delta_pp"], "crop_arm_included", out["crop_arm_included"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
