"""Push the MK-SH-V release to the Hugging Face Hub.

    python scripts/push_hf.py model   # Gjusev/MK-SH-V: card, policy, weights
    python scripts/push_hf.py space   # Gjusev/MK-SH-V-demo: the Gradio demo

Both repos are created public if missing (the release decision, 2026-10-22).
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Gjusev/MK-SH-V"
SPACE_ID = "Gjusev/MK-SH-V-demo"


def push_model(api: HfApi) -> None:
    create_repo(MODEL_ID, repo_type="model", exist_ok=True)
    upload_file(path_or_fileobj=str(ROOT / "MODEL_CARD.md"), path_in_repo="README.md",
                repo_id=MODEL_ID, repo_type="model")
    upload_file(path_or_fileobj=str(ROOT / "release/inference-policy.json"),
                path_in_repo="inference-policy.json", repo_id=MODEL_ID, repo_type="model")
    upload_file(path_or_fileobj=str(ROOT / "release/model-s42.safetensors"),
                path_in_repo="model-s42.safetensors", repo_id=MODEL_ID, repo_type="model")
    upload_file(path_or_fileobj=str(ROOT / "release/inference-policy-ensemble.json"),
                path_in_repo="inference-policy-ensemble.json", repo_id=MODEL_ID, repo_type="model")
    for asset in ("assets/mkshv-film.mp4", "assets/mkshv-film.jpg", "assets/logo-mark.png",
                  "assets/chart-accuracy.png", "assets/chart-gate.png"):  # card embeds
        upload_file(path_or_fileobj=str(ROOT / asset), path_in_repo=asset,
                    repo_id=MODEL_ID, repo_type="model")
    print("card + policy up; uploading weights (~1 GB)...")
    upload_file(path_or_fileobj=str(ROOT / "release/model.safetensors"),
                path_in_repo="model.safetensors", repo_id=MODEL_ID, repo_type="model")
    print(api.model_info(MODEL_ID).sha, MODEL_ID)


def push_space(api: HfApi) -> None:
    create_repo(SPACE_ID, repo_type="space", space_sdk="gradio", exist_ok=True)
    demo = ROOT / "demo"
    files = ["app.py", "requirements.txt", "README.md", "mksh_v/__init__.py",
             "mksh_v/arch_a1v.py", "mksh_v/sequence.py", "examples/NOTICE.md",
             "demo_cases.json", "cases.json", "static/logo-mark.png",
             "game/__init__.py", "game/days.json"]
    files += [f"examples/{p.name}" for p in sorted((demo / "examples").glob("*.png"))]
    files += [f"game/{p.relative_to(demo / 'game').as_posix()}"
              for p in sorted((demo / "game" / "assets").rglob("*")) if p.is_file()]
    for rel in files:
        if not (demo / rel).exists():
            continue
        upload_file(path_or_fileobj=str(demo / rel), path_in_repo=rel,
                    repo_id=SPACE_ID, repo_type="space")
    print(api.space_info(SPACE_ID).sha, SPACE_ID)


def main() -> int:
    api = HfApi()
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("model", "both"):
        push_model(api)
    if what in ("space", "both"):
        push_space(api)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
