# Third-party notices

This project builds on open models and datasets from third parties. Their
licenses and attributions apply to their artifacts independently of this
repository's code license. The complete licence review lives in
`docs/data-licenses.md`; this file is the release-facing summary.

## Models

| Artifact | License | Use here |
|---|---|---|
| [NeoMME-260M](https://huggingface.co/Hcompany/NeoMME-260M) (Hcompany) | Apache-2.0 | pretrained single-tower backbone; fine-tuned by this project with a new marker head |
| [MK-SH-1](https://huggingface.co/Gjusev/MK-SH-1) v1.1 (this project's sibling) | Apache-2.0 | E35 text replay mix during training; OCR→MK-SH-1 comparator; text evaluation sets testC and task |
| [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | Apache-2.0 | comparator (task-trained rerun attempt, E39/E39-r2; no valid result) |
| [jev-vlm-4b](https://huggingface.co/sidman007/jev-vlm-4b) | per its model card | zero-shot comparator, evaluation only, not modified |

The trained weights produced by this project fine-tune NeoMME-260M; the
model card states exactly what was inherited and what was trained.

## Datasets

| Dataset | License | Use |
|---|---|---|
| [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (Park et al. 2019, NAVER Clova) | CC BY 4.0 | evaluation: `validation`+`test` receipts mapped to field_threshold (E38c/E39+); training: `train` split only, from E41 (`5ed7a4f`) |
| [GUI-Odyssey](https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey) (Lu et al. 2024, OpenGVLab) | CC BY 4.0 | evaluation: `task_split` test episodes mapped to task_done (E38c/E39+); training: `task_split` train only, from E41 (`5ed7a4f`) |
| Own HTML+Playwright synthetic generator | ours, Apache-2.0 | training/eval/calibration families E38a/E38b |
| MK-SH-1 E35 text mix (BANKING77, MASSIVE, CLINC150 derivations) | see the MK-SH-1 notices | text replay, 1 of 4 microbatches |

Derivations: images shipped byte-identical to the sources (no re-encoding);
labels re-derived from each source's own ground truth (CORD parsed totals;
GUI-Odyssey step structure) with rules committed before generation; question
and option templates are ours; per-file SHA-256 manifests ship with every
derived dataset. Pinning: CORD v2 rev `7f0115a4`, GUI-Odyssey rev
`71e0e7e2` for training; the frozen evaluation revision and row hashes are
pinned in `docs/E39_CRITERIA_2026-10-14.md`.

## Software

Python packages used at build/train/eval time (pinned versions in the run
logs): torch, transformers, datasets, safetensors, pillow, pytesseract
(Tesseract OCR, Apache-2.0, with tessdata/tessdata_best Apache-2.0
language data), scipy, pytest, huggingface_hub, kaggle CLI.
