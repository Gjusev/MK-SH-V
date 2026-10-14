# E39 criteria (DRAFT — NOT FROZEN; frozen by the coordinator when the E38c probe lands)

E39 answers one question: **on our synthetic document/screen families, does the
visual A1-V beat an OCR→text pipeline built on MK-SH-1, without losing text
capability — and does any of it transfer to real documents/screens?**

Decision adopted upstream (review council, 2026-10-14, `docs/decisions.md`):
E39 trains **synthetic-only** (E38a+E38b). Real data (CORD, DocLayNet,
GUI-Odyssey) is **evaluation only**. One single job runs the arms and also
evaluates them on real data.

## Pins (fill at freeze time)

- mksh-v code commit: `TBD` (the commit that carries `kaggle/e39_train.py`)
- Kaggle image / torch / CUDA / transformers / Python: `TBD` (env dump in the
  job output, E37 procedure)
- Tesseract + language packs (eng/deu/spa/fra) versions: `TBD` (from the
  OCR-prep kernel; license re-checked at freeze in data-licenses.md)
- Data: Hub privado `Gjusev/MK-SH-V-E38a` + `Gjusev/MK-SH-V-E38b`
  (per-family rows.jsonl sha256 verified in-kernel against each manifest);
  read with the Kaggle secret `HF_TOKEN` (fine-grained, read-only, per
  notebook)
- OCR cache + frozen rules: private Kaggle dataset `gjusev/mkshv-e39-ocr`
  (`ocr.jsonl` with per-image text+confidence, `rules_predictions.jsonl`;
  shas in its manifest). Transport: CPU kernel `gjusev/mkshv-e39-ocrprep` →
  CLI output download → CLI dataset upload (GO-gated, ABSOLUTE `-p` paths)
- A1-V init: E37 C3 checkpoint `c3_final.safetensors` sha256
  `bac7c72242d236f8717338d5bfe9751a24df13c41147de6a5b83a248dd220f6c`
  (1 056 039 004 B) via the private Kaggle dataset `gjusev/mkshv-e37-ckpt`
  (uploaded and ready; carries a SHA256SUMS file). Pre-declared fallback: C3
  recipe rerun in-job (3 phases × fresh AdamW, rng 42^0xE35^phase), sha
  recorded, divergence declared.
- OCR-arm init: `mksh1_v12.safetensors` = MK-SH-1 v1.2, sha256
  `ea66cdde2271b98efda9dbb5d4437b59aa422cb34daca85d4203508aa56264d5`
  (576 788 228 B), same dataset.
- SmolVLM2 arm: **the pinned "SmolVLM2-500M" does not exist** — the series is
  256M / 2.2B only (verified on the Hub 2026-10-14). Verified candidates,
  both Apache-2.0, ungated: `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`
  (sha `067788b187b95ebe7b2e040b3e4299e342e5b8fd`; working default — quota
  as computed) and `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
  (sha `482adb537c021c86670beed01cd58990d01e72e4`; needs new arithmetic).
  **The freezer picks one and re-checks data-licenses at freeze.**
- jev-vlm-4b comparator: `sidman007/jev-vlm-4b`, revision
  `8c56e46a2b47dee398bc71c39c355b9716dec21e`, Apache-2.0 (verified on the
  Hub 2026-10-14). **It is an adapter repo with custom_code**
  (`adapter/adapter_config.json`, `modeling_jevvlm.py`): if it does not load
  in the pinned environment → `not_run` with the exact error, no patches.
- Real-eval set: `Gjusev/MK-SH-V-E38c`, PRIVATE, FIXED revision
  `98dde3d2b3dd0fd319ff44d059656a2d2864c66e` (never `main`; manifest.json
  sha256 `c4aa04f823ff9a78d8b7916350c715735c1f1d900d424dc4f5a19c5e4ac95ed2`).
  The kernel verifies in-job the sha256 of both rows files and cuts the real
  eval to `not_run` on any mismatch:
  `cord/rows.jsonl` `5d0dbc9e7ba3b3301bbcc1d52ecc2fb352b281dcb72c76bbe21ad34e23ab3f09`
  (576 rows = 192 receipts × 3 languages, field_threshold) and
  `gui_odyssey/rows.jsonl` `dcdbe6170752738cbb4a72a7f4cafbb56f4b30177cdb79add4c0e2c5a3c2a8da`
  (2 010 rows = 335 episodes × 2 polarities × 3 languages, task_done,
  label_origin `annotation+visual-verification`). All rows are split=test.
  The goal text rides in `state` in English across the three languages
  (declared).
- Laya Vision (`nnlgsakib/laya-vision`): **`not_run`, declared** — no
  published inference code; reimplementing it would not be a fair comparison.

## Arms, comparators and régimen

Every results table carries a `régimen` column: **task-trained** vs
**zero-shot**. No sentence of the report compares A1-V with a zero-shot model
without that label.

- **A1-V** (task-trained; gated): NeoMME-260M + marker head, init = E37 C3
  checkpoint, image at **1024 px** (validated end-to-end by C4; mb 4×8,
  5,56 s/step, 11,7 GB peak — 1536 only has 100-step C1 evidence with
  14 608 MB reserved vs a 14,56 GB T4, an OOM risk).
- **OCR→MK-SH-1** (task-trained; gated): Tesseract on the ORIGINAL PNG
  (precomputed by the CPU kernel) → OCR text as `state`; MK-SH-1 fine-tuned
  with the E35 recipe.
- **OCR + deterministic rules** (task-trained, dev-fitted; comparator):
  frozen rules over the OCR text for field_threshold (localized-total regex
  + comparison with the question's threshold), doc_type (class keywords) and
  form_complete (required-`*` fields empty or low-confidence OCR →
  illegible). Tuned ONLY on the E38a dev split; frozen with this file.
  Reported on test, real and per family.
- **SmolVLM2-500M** (task-trained; reported visual comparator, no gate):
  first-token option-letter logits, same recipe.
- **jev-vlm-4b** (zero-shot; reference): evaluated without training on the
  full real eval and on a sha256-ordered sample of 200 cases per family of
  the non-FR test (~1 400). Framing: **an existing ~15× larger model without
  adaptation — a reference, not an equal-footing comparison.**

## Priority order (frozen; the kernel cuts the tail if quota runs out)

(a) A1-V s42 at 1024 px + eval · (b) OCR→MK-SH-1 and OCR+rules · (c)
statistical comparison G1–G4 + per-family analysis · (d) finalist seeds
43/44 · (e) 768/1536 inference sweep and, if quota remains, a 1536-px
training run · (f) jev-vlm-4b zero-shot and SmolVLM2. Steps (a)–(c) protect
the scientific question. Wall budget guard: 11,5 h of the 12 h session cap;
phases beyond it are recorded `not_run` with reason.

## Text-replay (anti-forgetting, fixed for every trained arm)

All trained arms mix the **E35 text mix** (extremes 248 + grid 944 + task
944) as replay: **1 of every 4 microbatches**, total optimizer steps
unchanged. G2 is anchored to frozen references precisely so that replay and
comparators cannot co-degrade through a relative gate.

## Mixture and budget

- Train = E38a+E38b `train` verbatim (21 000; 7 families × 3 000; en/es/de)
  + E35 replay. No real data in training.
- Effective batch 32 (a1v mb 4×8; ocr mb 8×4; smol mb 4×8). **2 000 steps**,
  AdamW 2e-5/1e-4, wd 0,01, warmup 5 %, clip 1,0, fp16, seed 42.
- **Both GPUs**: orchestrator + per-GPU workers; quota bills the session
  wall at double rate either way.
- Selection: dev ONLY (frozen 1 050-id half, crc32-seeded, ids written to
  evidence), every 500 steps, metric = macro-mean of the 7 family
  accuracies; BEST checkpoint kept and restored before any test/gate
  evaluation. Test (5 600) opens exactly once, with the best weights.
- Finalist (better macro-dev among a1v/ocr) runs seeds 43/44, whichever arm
  it is; each evaluated once on non-FR test. Gates compare seed-42.
- Smoke-first: 20 steps per arm before its full budget.

## Gates (numeric margins fixed now)

All accuracies on the **non-FR test slice** (4 200; FR's 1 400 rows reported
separately, never gate):

- **G1 visual advantage (roadmap gate)**: Δ = acc(A1-V s42) −
  acc(OCR→MK-SH-1 s42); PASS iff the 95 % paired-bootstrap CI of Δ
  (whole `group_id` episodes, 1 000 resamples, seed 42) has its **lower
  bound ≥ +2,0 pp**. Also reported: Δ with CI **per family** (no gate).
- **G2 no text forgetting** (anchored to frozen MK-SH-1 v1.1, E35 s42 —
  testC 0,979 / task 0,800): each gated arm must reach testC ≥ **0,959**
  and task ≥ **0,780** (−2 pp), CPU float32. testC/task reported for every
  arm that runs, with its régime label.
- **G3 ablation tripwire** (visual arm, A1-V s42-best): swapped images (roll
  by 1 within family×language) must DROP ≥ 15 pp; absent and degraded are
  descriptive.
- **G4 real eval + frozen interpretation rule**: reads `Gjusev/MK-SH-V-E38c`
  at the pinned revision with rows-sha verification (mismatch → `not_run`).
  Label quality (E38c record): negatives 1/95 invalid (CP95 upper 5,0 %);
  positives 335/982 verifiable (65,9 % discarded); coordinator agreement
  43/50 = 86 % (≥ 85 % → not a noisy eval); kappa not computable (the
  subsample is all operator-"yes"); 6 of 7 disagreements are purchase goals
  whose final capture only shows "Added to cart". CORD and GUI-Odyssey are
  reported in the **main table**, not an annex. **Kill criterion**: if every
  evaluated arm scores ≥ 95 %, the real eval does not discriminate.
  **Interpretation rule (frozen)**: if G1 passes on synthetic but the
  governing real Δ(A1-V − OCR) CI includes 0 or is negative, the result is
  reported as **"ventaja solo sintética, sin evidencia de transferencia"**
  and neither the model card nor any announcement claims a general visual
  advantage. Release gates do not change.
- **Pre-registered sensitivity (task_done real)**: reported (i) with all 335
  episodes and (ii) excluding the 38 purchase-goal episodes (297). If the
  readings differ in sign or in whether the Δ CI excludes 0, **(ii) — the
  strict one — governs** the interpretation rule; the kernel computes both
  and flags the governing reading.
- **Claim rules (comparators, frozen now)**: (i) SmolVLM2 — if the CI of
  Δ(A1-V − SmolVLM2) has upper bound < 0, it is declared in the report and
  model card; no "best small visual model" claim; not a stop. (ii) rules —
  in each family where the CI of Δ(rules − A1-V) has lower bound ≥ 0, no
  visual-advantage claim is made for that family.

### Outcome rule

- **GO-E40** iff G1 and G2 hold (G3 is a tripwire: failing it blocks claims
  and forces a generator fix). G4 informs the narrative per its frozen
  interpretation rule.
- If G1 fails: negative result published, no visual release (roadmap).
- One job (or resumed sessions if the 12 h wall approaches), one shot per
  seed, no retries that touch measurements.

## Per-family power (fixed table; episodes = distinct group_id in non-FR test)

Minimal true Δ so that the one-sided 95 % lower bound of the paired CI
clears +2,0 pp (normal approximation, independent errors; episodes per
family ≈ 420–509, aggregate ≈ 2 900):

| accuracy del perdedor | por familia (n≈420) | por familia (n≈500) | agregado (n≈2 100) |
|---|---|---|---|
| 0,90 | 5,0 pp | 4,8 pp | 3,4 pp |
| 0,95 | 3,9 pp | 3,8 pp | 2,9 pp |
| 0,97 | 3,3 pp | 3,2 pp | 2,6 pp |
| 0,99 | 2,0 pp | 2,0 pp | 2,0 pp |

Reading: at parities of 0,97+ the aggregate gate detects ≈ +2,5 pp true
deltas; per-family statements need ≈ +3,3 pp or more and are reported, not
gated.

## Resolution analysis (descriptive)

- Inference sweep of the finalist A1-V at 768/1024/1536 on dev and on the
  real set; recorded per resolution.
- field_threshold error analysis separating **"cifra decisiva ilegible a esa
  resolución"** (Tesseract cannot recover the ground-truth total from the
  downscaled image) from **"error de decisión"**, per resolution.
- Optional 1536-px training run at the end of phase (e) if the wall budget
  allows (estimate 11 580 s from C1); descriptive, seed 45.

## Calibration note (E40, recorded here so it is not lost)

E40 must report risk–coverage curves **per family and per set** (synthetic
test, FR, real) in addition to the global curve.

## Evidence

`runs/e39-*/`: `env.json`, `dev_half_ids.json`, `ocr_manifest.json`,
per-run `curves.jsonl` + best weights, per-case rows for test (non-FR + FR),
ablations, E35 text sets, real eval (per arm), rules predictions, jev sample
ids + rows + per-family CIs, `field_threshold_error_analysis.json`,
`train_1536.json`, `e39_summary.json` with gates, régimen labels,
`skipped_phases`, and rolling full-state checkpoints for resume.

Kernels (user marks the `HF_TOKEN` secret on each notebook):
- `gjusev/mkshv-e39-ocrprep` (CPU-only, no GPU quota)
- `gjusev/mkshv-e39-train` (GPU orchestrator + per-GPU workers)

## Quota arithmetic (measured 2026-10-14 ~14:55 local, MCP read-only)

- GPU counter: 23 033,553 s used of 108 000 s → **84 966 s remaining**
  (refresh 2026-10-17T00:00Z); after refresh: **108 000 s**. T4×2 bills the
  SESSION wall at double rate regardless of GPUs used.

Per-run GPU time: A1-V 2 000 × 5,56 s = 11 120 s; OCR ≈ 1 700 s (estimate);
SmolVLM2 ≈ 8 000 s (estimate); jev zero-shot ≈ 2 000 s (sample + real,
estimates); evals ≈ 5 000 s; sweep + error analysis ≈ 2 000 s; overhead
≈ 2 000 s; optional 1536 train ≈ 11 580 s.

Phase wall: (a)+(b) ≈ 11 120 s ∥ · (c) ≈ 1,7 h · (d) ≈ 3,1 h · (e) ≈ 0,9 h
(+3,2 h si entra el 1536) · (f) ≈ 2,8 h.

| escenario | wall | counter | holgura s/ 84 966 | holgura s/ 108 000 |
|---|---|---|---|---|
| (a)–(f) sin 1536-train | ≈ 9,5 h | ≈ 68 400 s | 19,5 % ✓ | 36,7 % ✓ |
| (a)–(f) + 1536-train | ≈ 12,7 h ✗ (>12 h) | — | — | — |

With the 12 h session cap the optional 1536 training does NOT fit in one
session after everything else; it only runs if earlier phases finish under
their estimates (wall guard decides at run time) or in a resumed session.
Nothing else falls: 1024 px, OCR+rules, jev-vlm-4b and SmolVLM2 all fit on
both quota bases with ≥ 15 % headroom.
