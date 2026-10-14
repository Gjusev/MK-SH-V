"""E37 preflight: pin the environment BEFORE freezing criteria.

Reports versions, GPU, and NeoMME API behavior (processor text/image smoke,
config dump). Deliberately does NOT train, resize embeddings or touch marker
tokens: C1-C4 are measured only after the criteria file is frozen and
committed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")


def log(msg: str) -> None:
    print(f"[preflight {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pip_install() -> None:
    # torch/torchvision come with the Kaggle GPU image; do not touch them.
    pkgs = [
        "git+https://github.com/Gjusev/MK-SH-1@9e1a226e",
        "transformers==5.19.0",
        "pillow>=10.1",
    ]
    wheel = sorted(Path("/kaggle/input").rglob("mksh_v-*.whl"))
    if wheel:
        pkgs.append(str(wheel[0]))
    log(f"pip install {pkgs}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def main() -> None:
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    pip_install()

    import torch
    import transformers
    from PIL import Image

    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "pillow": Image.__version__ if hasattr(Image, "__version__") else None,
        "cuda_available": torch.cuda.is_available(),
    }
    if not env["cuda_available"]:
        raise SystemExit("PREFLIGHT ABORT: torch.cuda.is_available() is False - no GPU assigned")
    env["gpu_count"] = torch.cuda.device_count()
    env["gpu_name"] = torch.cuda.get_device_name(0)
    env["gpu_total_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    log(f"GPU: {env['gpu_name']} x{env['gpu_count']} {env['gpu_total_mem_gb']}GB, "
        f"torch {env['torch']} cu{env['torch_cuda']}, transformers {env['transformers']}")

    from transformers import AutoModel, AutoProcessor

    model_name = "Hcompany/NeoMME-260M"
    processor = AutoProcessor.from_pretrained(model_name)
    tok = processor.tokenizer
    cfg_dump = {}
    model = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
    model.float()
    cfg = model.config
    cfg_dump = {
        "model_type": cfg.model_type,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "embedding_rank": cfg.embedding_rank,
        "embedding_dim": getattr(cfg, "embedding_dim", None),
        "num_hidden_layers": cfg.num_hidden_layers,
        "layer_types_full_at_idx": [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"],
        "max_position_embeddings": cfg.max_position_embeddings,
        "patch_size": cfg.patch_size,
    }
    env["config"] = cfg_dump
    word = model.get_input_embeddings().weight
    backbone = getattr(model, model.base_model_prefix, model)
    env["tables"] = {
        "word_embeddings": list(word.shape),
        "value_embeddings": list(backbone.value_embeddings.weight.shape),
    }

    # processor smoke A: text-only
    feats_text = processor(text="motor temperature reading is 18.3", return_tensors="pt")
    assert "input_ids" in feats_text and "pixel_values" not in feats_text
    env["smoke_text"] = {"input_ids_len": int(feats_text["input_ids"].shape[1])}

    # processor smoke B: image-only -> doc + img grid + 2-axis positions
    img = Image.new("RGB", (128, 96), (255, 255, 255))
    feats_img = processor(images=[img], max_side=64, padding=True, return_tensors="pt")
    ids = feats_img["input_ids"][0].tolist()
    env["smoke_image"] = {
        "input_ids_len": len(ids),
        "starts_doc": ids[0] == tok.document_token_id,
        "img_tokens": int(sum(1 for t in ids if t == tok.image_token_id)),
        "pixel_values_shape": list(feats_img["pixel_values"].shape),
        "position_ids_shape": list(feats_img["position_ids"].shape),
    }

    # API smoke: one tiny no-grad CPU forward (not a measurement, a load check)
    with torch.no_grad():
        out = model(
            input_ids=feats_text["input_ids"],
            attention_mask=feats_text["attention_mask"],
        )
    env["smoke_forward"] = {"last_hidden_shape": list(out.last_hidden_state.shape)}

    env["preflight_wall_s"] = round(time.time() - t0, 1)
    env["status"] = "completed"
    (WORK / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    log(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
