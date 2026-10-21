# Data and model licenses

Status of every external source considered for MK-SH-V. Policy: **train only on
sources with a verified permissive license or on data we generate ourselves.**
Non-commercial or unclear sources are used for evaluation only, and that use is
declared. A source moves from `unverified` to `verified` only after reading the
primary license text (repository LICENSE file or official dataset card), with
the URL and date recorded here.

Last reviewed: 2026-10-14.

## Models

| Source | License | Status | Role |
|---|---|---|---|
| Hcompany/NeoMME-260M | Apache-2.0 | verified (model card) | backbone |
| jhu-clsp/mmBERT-small | MIT | verified earlier in MK-SH-1 | fallback text encoder |
| Gjusev/MK-SH-1 | Apache-2.0 | own | comparator, fallback init |
| SupersonicLabs/Julia-1 | Apache-2.0 | model card | comparator |
| [HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) rev `7b375e1b` | Apache-2.0 | verified (Hub API, 2026-10-14) | E39 comparator (task-trained) |
| [sidman007/jev-vlm-4b](https://huggingface.co/sidman007/jev-vlm-4b) rev `8c56e46a` | Apache-2.0 | verified (Hub API, 2026-10-14) | E39 external comparator (zero-shot, adapter + custom code) |
| nnlgsakib/laya-vision | Apache-2.0 | verified (Hub API, 2026-10-14) | not run: no inference code published |
| [Tesseract](https://github.com/tesseract-ocr/tesseract) + tessdata / tessdata_best | Apache-2.0 | verified (GitHub LICENSE, 2026-10-14) | OCR for E39 comparators (OCR→MK-SH-1, OCR+rules) |

## Datasets

| Source | License | Status | Allowed use |
|---|---|---|---|
| Own synthetic generator (HTML templates rendered with Playwright) | ours, Apache-2.0 | own | train / dev / calib / test |
| [CORD](https://github.com/clovaai/cord) (Indonesian receipts) | CC BY 4.0 | re-verified 2026-10-14 (repo `LICENSE-CC-BY`; v2 data on HF `naver-clova-ix/cord-v2`, 800/100/100) | evaluation: `validation`+`test` (E38c, E39+); training: `train` split only, from E41 (frozen `5ed7a4f`); attribution required |
| [DocLayNet](https://github.com/DS4SD/DocLayNet/blob/main/LICENSE) | CDLA-Permissive-1.0 | re-verified 2026-10-14 (LICENSE text: permissive, commercial OK, attribution + license notice on redistribution) | evaluation (E39); training not before a later frozen experiment; preserve attribution if republished |
| [GUI-Odyssey](https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey) | CC BY 4.0 | re-verified 2026-10-14 (dataset card, `cc-by-4.0`) | evaluation: `task_split` test (E38c, E39+); training: `task_split` train only, from E41 (frozen `5ed7a4f`); attribution required |
| [ScreenQA](https://github.com/google-research-datasets/screen_qa) annotations | CC BY 4.0 | verified for annotations | blocked until Rico image license is verified |
| Rico screenshots | unknown | unverified (official page 404) | none yet |
| [AndroidControl](https://github.com/google-research/google-research/blob/master/android_control/README.md) | Apache-2.0 covers the repo; data license not stated | unverified | none yet |
| [FUNSD](https://guillaumejaume.github.io/FUNSD/work/) | "solely for non-commercial, research and educational purposes" | verified restrictive | eval only |
| RVL-CDIP / IIT-CDIP | no explicit license; source documents from UCSF Industry Documents Library | restrictive / unclear | eval only; known train/test overlap |
| DocVQA | portal terms, images from the same tobacco archive | unclear | eval only |
| [CoSyn-400K](https://huggingface.co/datasets/allenai/CoSyn-400K) | ODC-BY-1.0, outputs subject to Anthropic and OpenAI terms | risk | not used |
| [Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix) | MIT annotations; PDFs from Common Crawl (PDFA) | underlying copyright unclear | not used for training |
| WebUI | reported license issue (TinyClick paper) | restrictive | not used |

Attribution for CC BY sources goes into the model card and `THIRD_PARTY_NOTICES.md`
at release time.
