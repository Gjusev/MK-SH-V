# MK-SH-V design

Status: **planned**. Nothing in this document is a measured result.

## Decision contract

Same contract as MK-SH-1, extended with images, and compatible with the
System One request shape that llama.cpp serves at `/v1/systemone`:

```
request:
  state:     text | JSON | chat messages (image parts as data URLs)
  images:    optional list of base64 data URLs
  questions: { name: { type: choice | score | noul,
                       instructions: str,
                       criteria: [ {label, description?} ] } }   # score: 2-10 levels, lowest first
response:
  answers:   { name: { choice | score | p_yes,
                       probabilities, confidence,
                       abstained: bool,
                       reason_code: null | below_validated_threshold | input_error } }
  model_revision, calibration_revision, policy_revision
```

`none` (a semantic answer) and abstention (the policy refuses) stay distinct.
`noul` is scored as two options (yes, no). `score` returns the expected level
for MAE and argmax for accuracy.

## Architecture

**A1 (primary): NeoMME-260M single tower with a marker head.**

```
[image patches] [doc] state </s> question [O0] option_0 [O1] option_1 ...
```

- One encoder pass per question. Image patches and text share the encoder.
- One marker token per option. A shared MLP head maps each marker's hidden
  state to a scalar. Softmax runs over valid options only. This is the
  MK-SH-1 `MarkerScorer` idea (`mokka_decisions.arch_a1`) ported to NeoMME.
- Image budget: longest side capped (start at 1024 px = up to 1024 patches of
  32 px; measure 768 and 1536 in E37). Aspect ratio preserved.
- Several questions about the same image: the first version re-encodes per
  question. Sharing the image encoding across questions is a later ablation.

**A2 (fallback): hybrid.** mmBERT-small initialised from MK-SH-1 reads the
text; image tokens from NeoMME (or SigLIP2-base) are projected into its input
sequence. Used only if E37 shows NeoMME loses clearly on text-only tasks.

Calibration: one temperature per checkpoint, fitted on a dedicated split.
Abstention: tau = smallest grid value with selective risk <= 5 % on the
calibration split, then reported per held-out set with Clopper-Pearson upper
bounds. Per-modality temperatures are an ablation, not the default.

## Task families

All generated from templates where the label is computed from the rendered
content, never read back from it.

| Family | Type | Example question |
|---|---|---|
| doc_type | choice | Which document type is this? invoice, receipt, contract, form, letter, other |
| field_threshold | noul | Is the invoice total at least 250.00? (bridges MK-SH-1 numeric order to pixels) |
| form_complete | choice | complete, missing required field, illegible |
| screen_state | choice | success, error, loading, needs input, blocked |
| task_done | noul | Given this goal, does the screenshot show it completed? |
| source_conflict | choice | Do the ticket text and the screenshot agree? (authority rule stated in the instructions) |
| urgency | score 1-5 | ticket text plus screenshot with or without an error banner |

Counterfactual pairs: a decisive change in the image flips the label; an
irrelevant change (colour, position, font) keeps it. Pairs share a split.

## Splits

Split by template family, app and episode **before** generating variants.
Held-out sets: unseen templates, unseen apps (GUI-Odyssey), unseen language
(FR eval only), real permissive documents (CORD, DocLayNet), and eval-only
restrictive sets (FUNSD, RVL-CDIP, DocVQA-derived). The test set never feeds
thresholds, temperature, hyperparameters or checkpoint choice.

## Metrics

Accuracy, NLL, multiclass Brier, ECE (15 bins) as secondary, ordinal MAE,
coverage and selective risk per set, false acceptance, AURC. Breakdowns by
family, language, K, image size and set. Paired bootstrap by episode. Latency
p50/p95 end to end on CPU and T4, preprocessing and OCR included for the
comparators, cold start reported separately. Parameter counts measured from
the export, total and trainable.

## Required ablations

Text only, image only, image swapped, image missing, image degraded, OCR text
with the same head, and the comparators in `docs/data-licenses.md`.
