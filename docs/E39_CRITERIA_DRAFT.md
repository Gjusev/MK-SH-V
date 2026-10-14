# E39 criteria (DRAFT — NOT FROZEN; frozen by the coordinator when the E38c probe lands)

E39 answers one question: **on our synthetic document/screen families, does the
visual A1-V beat an OCR→text pipeline built on MK-SH-1, without losing text
capability — and does any of it transfer to real documents/screens?**

Decision adopted upstream (review council, 2026-10-14, `docs/decisions.md`):
E39 trains **synthetic-only** (E38a+E38b). Real data (CORD, DocLayNet,
GUI-Odyssey) is **evaluation only**; its mapping to the contract is being
probed by a separate session and enters this file only through the G4 section
below. One single job runs all arms and also evaluates them on real data.

## Pins (fill at freeze time)

- mksh-v code commit: `TBD` (the commit that carries `kaggle/e39_train.py`)
- Kaggle image / torch / CUDA / transformers / Python: `TBD` (same pinning
  procedure as E37: env dump recorded in the job output)
- Tesseract + language packs (eng/deu/spa/fra) versions: `TBD` (recorded from
  the OCR-prep kernel; license re-checked at freeze in data-licenses.md)
- Data: Hub privado `Gjusev/MK-SH-V-E38a` + `Gjusev/MK-SH-V-E38b` (per-family
  rows.jsonl sha256 verified in-kernel against each repo manifest); read with
  the Kaggle secret `HF_TOKEN` (fine-grained, read-only, per-notebook)
- OCR cache: private Kaggle dataset `gjusev/mkshv-e39-ocr` (transport chain:
  CPU-only kernel `gjusev/mkshv-e39-ocrprep` → CLI output download → CLI
  dataset upload at GO; sha256 of `ocr.jsonl` in its manifest)
- A1-V init: E37 C3 checkpoint `c3_final.safetensors`
  sha256 `bac7c72242d236f8…` (full sha at freeze) via the private Kaggle
  dataset `gjusev/mkshv-e37-ckpt`. **Pre-declared fallback** if transport
  fails: re-run the C3 recipe in-job (E35 mix, seed 42, 3 phases × fresh
  AdamW, rng 42^0xE35^phase) and record the sha obtained; divergence declared.
- OCR-arm init: `mksh1_v12.safetensors` = MK-SH-1 v1.2 weights
  (sha `ea66cdde…`), via the same `gjusev/mkshv-e37-ckpt` dataset
- SmolVLM2 arm init: `HuggingFaceTB/SmolVLM2-500M…` (exact repo id pinned at
  freeze; Apache-2.0 re-verified then)
- Real-eval set: `Gjusev/MK-SH-V-E38c` (provisional name, may not exist yet —
  skipped cleanly as `not_run` while absent)

## Arms (3)

- **A1-V**: NeoMME-260M + marker head (`mksh_v.arch_a1v`), init = E37 C3
  checkpoint. Input: image at **1024 px** (the C1/C4 point: 5,56 s/step,
  microbatch 4 × 8, 11,7 GB peak) + text tail (state, question, options).
  768 px is a finalist-only descriptive ablation, run only if quota remains.
- **OCR→MK-SH-1**: Tesseract on the ORIGINAL PNG (precomputed by the CPU
  kernel over E38a+E38b train/dev/test) → OCR text becomes `state`;
  MK-SH-1 fine-tuned with the E35 recipe (`julia_sequence` 1024/256).
- **SmolVLM2-500M**: comparator per roadmap; scores options by first-token
  logits (declared design: image + `Question … Options: A)… B)… Answer:`,
  softmax over the option-letter logits). Fine-tuned with the same recipe.
  Included because the two-GPU arithmetic leaves ≥ 15 % headroom on both
  quota bases; dropped to conditional if the measured headroom after the
  compared phase falls below that.

## Text-replay (anti-forgetting, fixed for every arm)

Both/all arms mix the **E35 text mix** (the C3 mix: extremes 248 + grid 944 +
task 944, from `gjusev/mokka-decisions-bundle`) as replay: **1 of every 4
microbatches** is a replay microbatch (text-only), total optimizer steps
unchanged. G2 is anchored to frozen references precisely so that replay and
comparators cannot co-degrade through a relative gate.

## Mixture and budget

- Train = E38a+E38b `train` split verbatim (21 000 cases; 7 families × 3 000;
  en/es/de) + E35 replay fraction above. No real data in training.
- Effective batch 32: A1-V microbatch 4 × 8 (1024-px memory); OCR and
  SmolVLM2 microbatch 8 × 4. **2 000 optimizer steps** (≈3,05 epochs),
  AdamW 2e-5/1e-4, wd 0,01, warmup 5 %, clip 1,0, fp16 autocast, seed 42.
- **Both GPUs of the T4×2 session**: the orchestrator spawns one worker per
  GPU (phase 1: A1-V s42 on gpu0 ∥ OCR s42 then SmolVLM2 s42 on gpu1; phase
  2: finalist seeds 43/44 in parallel, one per GPU). Quota bills the SESSION
  wall at double rate regardless of how many GPUs are used.
- Selection: dev ONLY. Dev half of 1 050 fixed ids (crc32-seeded, frozen list
  written to evidence) evaluated every 500 steps; metric = macro-mean of the
  7 family accuracies; the BEST checkpoint by that metric is kept (weights
  saved at every dev improvement and restored before any test/gate
  evaluation — the last step never decides). The test split (5 600) is opened
  exactly once, at the end, with the best weights.
- Finalist (better macro-dev) runs seeds 43 and 44 additionally, whichever
  arm it is; each seed is also evaluated once on the non-FR test and
  reported. Gates compare seed-42 head-to-head.
- Smoke-first: every arm runs a 20-step smoke before its full budget; a smoke
  failure aborts the job before the budget is spent.

## Gates (numeric margins fixed now)

All accuracies on the **non-FR test slice** (4 200 cases; the 1 400 FR rows
are reported separately and never gate — FR is eval-only by design):

- **G1 visual advantage**: Δ = acc(A1-V s42) − acc(best non-visual arm s42)
  on non-FR test. PASS iff the 95 % paired-bootstrap CI of Δ (resampling
  whole `group_id` episodes, 1 000 resamples, seed 42) has its **lower bound
  ≥ +2,0 pp**.
- **G2 no text forgetting** (anchored to the FROZEN MK-SH-1 v1.1 references,
  E35 s42: testC 0,979, task 0,800 — not to any retrained arm): on the frozen
  E35 text sets (CPU float32), **every arm** must reach testC ≥ **0,959** and
  task ≥ **0,780** (−2 pp). testC/task of both arms are reported.
- **G3 ablation tripwire** (always on the VISUAL arm, A1-V s42-best, non-FR
  test): with images swapped (roll by 1 within each family×language block),
  accuracy must DROP ≥ 15 pp vs clean. Smaller drops mean text leakage and
  block any claim. Absent (white page, same canvas) and degraded (half-res
  round-trip) ablations are descriptive; 768-px re-eval of the finalist is
  descriptive and only if quota remains.
- **G4 real eval**: reads the private Hub repo `Gjusev/MK-SH-V-E38c`
  (provisional name; `not_run` cleanly while it does not exist). Probe
  outcome (2026-10-14): CORD → field_threshold (~188 real receipts, our
  threshold rule, balanced) and GUI-Odyssey → task_done (~500 test episodes,
  final=true vs early-capture=false pairs; re-probe r2 pending). DocLayNet
  out. doc_type, screen_state, source_conflict, urgency and form_complete
  have NO real eval — synthetic only. **Kill criterion (fixed now)**: if ALL
  arms score ≥ 95 % on the real-eval set, the real eval does not
  discriminate — recorded as `real-eval non-discriminating`, no transfer
  claim either way.

### Outcome rule

- **GO-E40** iff G1 and G2 hold (G3 is a tripwire: failing it blocks claims
  and forces a generator fix before any release). G4 informs the release
  narrative; its role is pinned by the freezer when the probe lands.
- If G1 fails: the visual advantage is not established at this budget —
  negative result published, no visual release (roadmap).
- One job (possibly split into sessions with resume if the 12 h session wall
  is approached), one shot per seed, no retries that touch measurements.

## Evidence

`runs/e39-*/`: `env.json`, `dev_half_ids.json`, `ocr_manifest.json` (engine
versions, per-image wall time, sha of ocr.jsonl), per-arm-run `curves.jsonl`
(dev-half every 500 steps) and best-weight files, per-case rows for test
(non-FR + FR), ablations, E35 text sets (both arms) and real eval,
`e39_summary.json` with gates, rolling full-state checkpoints for resume
(model, optimizer, scheduler, scaler, RNG, step, data cursor).

Kernels (user marks the `HF_TOKEN` secret on each notebook):
- `gjusev/mkshv-e39-ocrprep` (CPU-only, no GPU quota)
- `gjusev/mkshv-e39-train` (GPU orchestrator + per-GPU workers)

## Quota arithmetic (measured 2026-10-14 ~14:55 local, MCP read-only)

- GPU counter: 23 033,553 s used of 108 000 s → **84 966 s remaining**
  (refresh 2026-10-17T00:00Z); after refresh: **108 000 s**. T4×2 bills the
  SESSION wall at double rate; using both GPUs does not bill extra.

Per-run GPU time at 1024 px: A1-V 2 000 × 5,56 s = 11 120 s; OCR ≈ 2 000 ×
0,85 s (E35 estimate; measured by the job) = 1 700 s; SmolVLM2 ≈ 2 000 × 4 s
(estimate) = 8 000 s. Evals+ablations+E35+real ≈ 5 000 s GPU; loads and
downloads ≈ 1 500 s.

- Phase 1 wall ≈ max(11 120, 1 700 + 8 000) = 11 120 s (3,1 h)
- Phase 2 wall ≈ 11 120 s (3,1 h; finalist A1-V worst case, one seed per GPU)
- Phase 3 (evals, sequential on gpu0) ≈ 1,4 h; overhead ≈ 0,4 h
- **Session wall ≈ 8,0 h < 12 h cap** → counter ≈ 2 × 8,0 h = **57 600 s**

| base | restante tras E39 | holgura |
|---|---|---|
| 84 966 s (hoy) | 27 366 s | **32 %** ✓ ≥ 15 % → SmolVLM2 incluido |
| 108 000 s (post-refresco) | 50 400 s | **47 %** ✓ → SmolVLM2 incluido |
