"""E52 local text-only evals (task + testC) with the e52 checkpoints, using the
kernel's own evaluate_set path (imgs None for text rows).

    .venv/Scripts/python.exe -I scripts/e52_text_eval.py <seed> <ckpt> <sha>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
sys.path.insert(0, str(ROOT / "src"))

MOKKA = Path(r"C:\Code Main\mokka-decisions")
SHAS = {42: "3d2ffdbcfe6eb9e851716079796cf9a2a04e1e0e90f30b9ffacecf66093c9aa3",
        43: "dfc6d60e15247c9e4b5902f6e84cbca9c70eba8a5489a06a580b6b43330de37d"}


def load_text_sets() -> dict:
    man = json.loads((MOKKA / "e29-mix/manifest.json").read_text(encoding="utf-8"))
    want = set(man["original_dev_ids"])
    task = [json.loads(l) for l in (MOKKA / "data/processed/v1/eval/dev_eval.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l)["id"] in want]
    testc = [json.loads(l) for l in (MOKKA / "data/e26-order/test_c_rows.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"task": task, "testC": testc}


def main() -> int:
    import hashlib
    from e42_calib import evaluate_set
    from e43_infer import load_model

    seed, ckpt = int(sys.argv[1]), Path(sys.argv[2])
    sha = sys.argv[3] if len(sys.argv) > 3 else SHAS[seed]
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == sha
    model, _ = load_model(seed, ckpt, sha)
    for name, rows in load_text_sets().items():
        out = evaluate_set(model, rows, None, "cpu")
        p = ROOT / f"runs/e52/percase_e52s{seed}__{name}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
        acc = sum(1 for r in out if r["pred"] == r["gold"]) / len(out)
        print(f"s{seed} {name}: n={len(out)} acc={acc:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
