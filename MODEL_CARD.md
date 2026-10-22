# MK-SH-V (A1-V) — small visual decision model with pre-registered abstention

**Model ID:** MK-SH-V · architecture variant "A1-V" · ~260M parameters
**Backbone:** NeoMME-260M (Apache-2.0) with a 32-slot marker head, single tower
**Inputs:** one image (documents or screenshots, longest side 1024 px) + a text
state, a question and answer options
**Outputs:** calibrated probabilities over the options (temperature T and
decision threshold τ fitted on calibration splits), the chosen option, and an
explicit abstention when the top probability is below τ
**Trained by:** Mokka Agentur (Gjusev). Code and criteria: Apache-2.0, this
repository. Checkpoints: seed 42 and seed 43 of E41 (best step 2,000).

## Intended use

Deciding among options about a document page or a screenshot (UI state,
task completion, thresholds over printed totals) with a calibrated
confidence signal, in settings where a 260M single-tower model is wanted
(edge, cost, auditability). Text decision competence is preserved (E35 text
sets) so the model can serve mixed text/visual decision workloads.

Out of scope: general OCR, captioning, open-ended VQA, safety-critical use
(the abstention guarantee below holds on the evaluated distribution only).

## Measured results (all evaluation criteria were frozen and committed
before any measurement; scripts reproduce every number from per-case files)

### Real images — receipts (CORD) and app screenshots (GUI-Odyssey), 2,586
held-out cases mapped to binary decision questions

| Arm | Params | Real overall | CORD field_threshold | GUI task_done | Regime |
|---|---|---|---|---|---|
| **A1-V s42** | **260M** | **0.765** | 0.550 | **0.826** | task-trained (E41) |
| **A1-V s43** | **260M** | **0.742** | 0.602 | 0.783 | task-trained (E41) |
| jev-vlm-4b | ~4B | 0.785 | 0.972 | 0.731 | zero-shot |
| OCR → MK-SH-1 | ~25M | 0.475 | 0.486 | 0.472 | task-trained |
| A1-V, synthetic-only training (E39) | 260M | 0.471 | 0.394 | 0.493 | task-trained |
| Deterministic rules | — | — | 0.594 | — | — |

- Adding ~10k CC-BY real training rows (disjoint splits) moved real accuracy
  from 0.471 to 0.765/0.742 (paired Δ vs E39: +29.4 pp, CI95 [+25.9, +32.9]
  seed 42; +27.2 pp [+23.3, +31.3] seed 43; episode-grouped bootstrap).
- **On GUI-Odyssey the 260M model beats the ~15× larger zero-shot jev-vlm-4b**
  (0.826 vs 0.731, seed 42); overall the two are statistically close
  (Δ CI95 seed 42 [−5.4, +1.2] pp). On CORD receipts jev and the rules
  baseline remain ahead — see Limitations. jev may have seen CORD /
  GUI-Odyssey during pretraining; this cannot be verified from its card.

### Synthetic held-out test (own generator, 7 families, 4,200 cases)

A1-V 0.908/0.909 (−2.5 pp vs the synthetic-only model, concentrated in
field_threshold and form_complete; the other five families are at ceiling).
Text-only competence is retained: testC 1.000/1.000 (MK-SH-1: 0.979), task
0.810/0.830 (MK-SH-1: 0.800).

### Abstention (E42; protocol: temperature T fitted on cal_temperature,
τ chosen on a separate cal_policy split as the smallest grid value whose
Clopper-Pearson 95 % upper bound on selective risk is ≤ 5 %; both fitted
before any held-out evaluation)

> On our held-out synthetic evaluation, both 260M checkpoints achieved over
> 86 % coverage while maintaining a 95 % Clopper-Pearson upper confidence
> bound on selective risk below 5 %.

- Seed 42: coverage 86.1 %, selective risk 2.65 %, CP95 upper 3.23 % (T=1.546, τ=0.55).
- Seed 43: coverage 86.4 %, selective risk 2.84 %, CP95 upper 3.43 % (T=1.293, τ=0.55).

This guarantee is protocol- and distribution-specific (our synthetic
families, threshold selected on calibration data disjoint from the test set).
No claim is made about abstention behaviour on other distributions — see
Limitations.

## Limitations (measured, not speculated)

1. **Abstention is not yet validated on real images.** On GUI-Odyssey the
   model answers 98 % of cases but 17–21 % of accepted answers are wrong
   (confident errors). On CORD it abstains from almost everything (coverage
   2–12 %) and still errs among accepted cases. The confidence signal that
   works on the synthetic distribution does not yet separate right from
   wrong on real data; per-domain thresholds or an OOD signal are open work.
2. **CORD receipts remain weak** vs baselines (0.55–0.60 vs rules 0.594,
   jev 0.972); threshold questions over printed totals in IDR formats are
   the largest open gap.
3. French held just above the risk budget (5.7 % point risk, CP95 upper
   7.1–7.3 %). The 48-case testC set cannot certify a ≤5 % CP upper bound
   even at zero errors (upper 7.4 % at n=48). Text task set risk 16–18 %.
4. Synthetic accuracy dropped 2.5 pp when real data was added (declared
   cost; families at ceiling were unaffected).
5. The E38c real evaluation set informed the error analysis that motivated
   the E41 training data; it was never trained or tuned on, and all
   selection used separate dev splits. This is disclosed for completeness.

## Training data and provenance

| Source | License | Split used | Pinned revision / SHA |
|---|---|---|---|
| Own HTML+Playwright synthetic generator (7 families, EN/ES/DE/FR) | ours, Apache-2.0 | all | repo `src/mksh_v/gen_e38*.py`; E38a/E38b manifests |
| CORD v2 (Park et al. 2019, NAVER Clova) | CC BY 4.0 | eval: validation+test (E38c); train split only for E41 training | rev `7f0115a4`; rows SHAs in `docs/E41_CRITERIA_2026-10-22.md` |
| GUI-Odyssey (Lu et al. 2024, OpenGVLab) | CC BY 4.0 | eval: task_split test (E38c); task_split train only for E41 training | rev `71e0e7e2`; rows SHAs in `docs/E41_CRITERIA_2026-10-22.md` |
| NeoMME-260M backbone | Apache-2.0 | init | E37 C3 checkpoint (SHA verified in-kernel) |
| MK-SH-1 E35 text replay mix | Apache-2.0 (ours) | text replay 1/4 of microbatches | git pin `9e1a226e` |

Transformations of CC-BY sources: receipts/screenshots kept as original
bytes (no re-encoding); labels derived from each source's own ground truth
(CORD parsed totals; GUI-Odyssey step structure) with the derivation rules
committed before generation; question/options/templates are ours. Full
per-file SHA-256 manifests ship with each dataset. See
`docs/data-licenses.md` for the complete table.

## Reproducibility

- Frozen criteria per experiment: `docs/E37..E42_*CRITERIA*.md` (committed
  before measurement; amendments likewise).
- Per-case outputs for every number above: `runs/e39-train-v7`,
  `runs/e39r2`, `runs/e40-calib`, `runs/e41`, `runs/e42-calib` (this
  repository's release bundle); post-hoc gate reports:
  `scripts/report_e41.py`, `scripts/report_e39r2.py`.
- Statistics: episode-grouped paired bootstrap (1,000 resamples, seed 42),
  exact two-sided Clopper-Pearson 95 % intervals with the risk budget on
  the upper end.

## Citation

Model and evaluation suite by Gjusev (Mokka Agentur), 2026. Repository
`mk-sh-v`; companion text model MK-SH-1 (`Gjusev/MK-SH-1`).
