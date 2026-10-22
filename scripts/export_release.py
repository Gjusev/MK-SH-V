"""Export an E41 best.pt to release/model.safetensors for the HF repo.

    python scripts/export_release.py runs/e41/runs/a1v_real_s42/best.pt release/model.safetensors
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    state = torch.load(src, map_location="cpu", weights_only=True)["model"]
    state = {k: v.contiguous() for k, v in state.items()}
    save_file(state, str(dst), metadata={"format": "pt"})
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    print(f"{dst} written: {len(state)} tensors, {dst.stat().st_size / 2**20:.0f} MiB, sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
