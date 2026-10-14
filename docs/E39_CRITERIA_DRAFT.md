# E39 criteria (DRAFT — NOT FROZEN; frozen by the coordinator when the E38c probe lands)

E39 answers one question: **on our synthetic document/screen families, does the
visual A1-V beat an OCR→text pipeline built on MK-SH-1, without losing text
capability — and does any of it transfer to real documents/screens?**

Decision adopted upstream (review council, 2026-10-14, `docs/decisions.md`):
E39 trains **synthetic-only** (E38a+E38b). Real data (CORD, DocLayNet,
GUI-Odyssey) is **evaluation only**; its mapping to the contract is being
probed by a separate session and enters this file only through the
`PENDIENTE-SONDA` section below. One single job runs both arms and also
evaluates them on real data.

## Pins (fill at freeze time)

- mksh-v code commit: `TBD` (the commit that carries `kaggle/e39_train.py`)
- Kaggle image / torch / CUDA / transformers / Python: `TBD` (same pinning
  procedure as E37: preflight-style env dump recorded in the job output)
- Tesseract + language packs (eng/deu/spa/fra) versions: `TBD` (recorded from
  the job's install log; license re-checked at freeze in data-licenses.md)
- Data: Hub privado `Gjusev/MK-SH-V-E38a` (manifest sha `TBD`), 
  `Gjusev/MK-SH-V-E38b` (manifest sha `TBD`); read with the Kaggle secret
  `HF_TOKEN` (fine-grained, read-only, per-notebook)
- A1-V init: E37 C3 checkpoint `c3_final.safetensors`
  sha256 `bac7c72242d236f8…` (full sha at freeze), transported as a private
  Kaggle dataset. **Pre-declared fallback** if transport fails: re-run the C3
  recipe in-job (E35 mix, seed 42, pins of `docs/E37_CRITERIA_2026-10-14.md`)
  and record the sha obtained; the divergence is declared in the report.
- OCR arm init: `Gjusev/MK-SH-1` v1.2 weights (sha `ea66cdde…`, verified
  against the frozen literal), via `mokka_decisions.arch_julia.JuliaArchScorer`
- Real-eval set: `PENDIENTE-SONDA` (name + manifest sha)

## Arms (2; the third does not fit — see quota arithmetic)

- **A1-V**: NeoMME-260M + marker head (`mksh_v.arch_a1v`), init = E37 C3
  checkpoint. Input: image (longest side 768 px, the C1-measured point
  3,44 s/step at microbatch 8) + text tail (state, question, options).
- **OCR→MK-SH-1**: Tesseract on the ORIGINAL PNG (each arm keeps its natural
  pipeline; no downscale before OCR) → OCR text becomes `state`; MK-SH-1
  fine-tuned with the E35 recipe (`julia_sequence` budgets 1024/256).
- **SmolVLM2-500M** (third arm): does NOT fit in the remaining quota with
  headroom (arithmetic below); remains conditional on a fresh quota window,
  same protocol, if ever run.

## Mixture and budget

- Train = E38a+E38b `train` split verbatim: 21 000 cases, 7 families × 3 000,
  languages en/es/de balanced, no real data, no oversampling.
- Effective batch 32 = microbatch 8 × accumulation 4 (A1-V) / microbatch 8 ×
  accumulation 4 (OCR arm). **2 000 optimizer steps** (≈3,05 epochs), AdamW
  groups encoder 2e-5 / head 1e-4, wd 0,01, warmup 5 %, clip 1,0, fp16
  autocast + GradScaler, seed 42. OCR runs as a CPU pool concurrent with
  A1-V training; its wall time is measured and recorded.
- Selection: dev ONLY. Dev half of 1 050 fixed ids (seeded, frozen id list
  from the `dev` split) evaluated every 500 steps; selection metric =
  macro-mean of the 7 family accuracies. The BEST checkpoint by that metric
  is kept (weights saved at every dev improvement and restored before any
  test/gate evaluation — the last step never decides). The test split
  (5 600) is opened exactly once, at the end, with the best weights.
- Finalist (better macro-dev) runs seeds 43 and 44 additionally, same
  budget, whichever arm it is; each seed is also evaluated once on the
  non-FR test and reported. Gates compare seed-42 head-to-head.
- Data integrity: per-family `rows.jsonl` sha256 verified in-kernel against
  each Hub repo's frozen manifest before training.

## Gates (numeric margins fixed now)

All accuracies on the **non-FR test slice** (4 200 cases; FR's 1 400 rows are
reported separately and never gate — FR is eval-only by design):

- **G1 visual advantage**: Δ = acc(A1-V s42) − acc(OCR arm s42) on non-FR
  test. PASS iff the 95 % paired-bootstrap CI of Δ (resampling whole
  `group_id` episodes, 1 000 resamples, seed 42) has its **lower bound
  ≥ +2,0 pp**.
- **G2 no text regression**: on the frozen E35 text sets (testC 48, task 300,
  CPU float32, same harnesses as E37 C3): acc(A1-V) ≥ acc(OCR arm) − 2,0 pp
  on EACH set.
- **G3 ablation tripwire** (always on the VISUAL arm, A1-V s42-best, non-FR
  test): with images swapped (roll by 1 within each family×language block),
  accuracy must DROP ≥ 15 pp vs clean. Smaller drops mean text leakage and
  block any claim. Absent (white page, same canvas) and degraded (half-res
  round-trip) ablations are descriptive.
- **G4 real eval**: reads the private Hub repo `Gjusev/MK-SH-V-E38c`
  (provisional name; skipped cleanly as `not_run` while it does not exist).
  Probe outcome (2026-10-14): CORD → field_threshold (~188 real receipts,
  our threshold rule, balanced) and GUI-Odyssey → task_done (~500 test
  episodes, final=true vs early-capture=false pairs; a re-probe r2 is
  pending). DocLayNet is out. doc_type, screen_state, source_conflict,
  urgency and form_complete have NO real eval — synthetic only. **Kill
  criterion (fixed now)**: if BOTH arms score ≥ 95 % on the real-eval set,
  the real eval does not discriminate — outcome recorded as
  `real-eval non-discriminating`, no transfer claim either way.

### Outcome rule

- **GO-E40** iff G1 and G2 hold (G3 is a tripwire: failing it blocks claims
  and forces a generator fix before any release). G4 informs the release
  narrative; its role is pinned by the freezer when the probe lands.
- If G1 fails: the visual advantage is not established at this budget —
  negative result published, no visual release (roadmap).
- One job, one shot per seed, no retries that touch measurements.

## Evidence

`runs/e39-train/` (kernel output): `env.json`, `ocr_manifest.json` (per-image
OCR wall time + engine versions), per-arm `curves.jsonl` (dev-half per 500
steps), `eval_<arm>__<set>.jsonl` per-case rows for test (non-FR + FR),
ablations, E35 sets, real eval; `e39_summary.json` with gates; checkpoints
(full state: model, optimizer, scheduler, scaler, RNG, step, data cursor)
every 500 steps for resume. Kernel slug: `gjusev/mkshv-e39-train` (the user
must mark the `HF_TOKEN` secret on this notebook before launch).

## Quota arithmetic (measured 2026-10-14 ~14:55 local, MCP read-only)

- GPU counter: 23 033,553 s used of 108 000 s → **84 966 s remaining**
  (refresh 2026-10-17T00:00Z). T4×2 bills double: counter ≈ 2 × GPU wall.
- A1-V: 3 seeds × 2 000 steps × 3,44 s = 20 640 s GPU.
- OCR arm: 2 000 × ~0,85 s (estimate from E35 wall; the job measures it) ≈
  1 700 s GPU.
- Dev/test/ablation/E35/real evals + model loads + OCR tail ≈ 7 400 s GPU.
- Total ≈ 29 700 s GPU ≈ **59 400 s counter → cabe con ~25 500 s (7,1 h) de
  holgura**. Adding SmolVLM2 (~2 000 × ~4 s ≈ 8 000 s GPU ≈ 16 000 s counter
  + evals) consumes the headroom to ≈ 0 → **2 brazos**.
