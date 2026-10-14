# E37 criteria (DRAFT, not yet frozen)

E37 answers one question: **can NeoMME-260M carry the MK-SH decision head on
the hardware we have, without losing the text capability MK-SH-1 already has?**

Freeze this file (rename to `E37_CRITERIA_<date>.md`, commit) before any
measurement. After freezing, thresholds and outcomes below do not change; a
new experiment number is needed for anything else.

## Pins (fill at freeze time)

- NeoMME-260M revision SHA: `TBD`
- transformers version (>= 5.17.0): `TBD`
- torch version and CUDA on the Kaggle image: `TBD`
- mokka-decisions commit: `9e1a226e`
- MK-SH-1 reference: Hub `Gjusev/MK-SH-1`, weights sha `ea66cdde...` (v1.1 = v1.2 weights)

## C1. Hardware compatibility

On one Kaggle T4, FP16 autocast, SDPA:
- 100 training steps of A1 on synthetic doc_type data at image longest side
  768, 1024 and 1536 px; microbatch found by measurement.
- Record: peak allocated and reserved memory, seconds per step, loss finite at
  every step, quota consumed (counter before and after).
- Pass: at least one image size runs 100 steps with finite loss and fits in
  one T4. Otherwise: try P100 FP32 at 768 px; if that fails too, outcome STOP-A1.

## C2. Marker tokens

NeoMME uses factorized embeddings. Check that adding 32 marker tokens and
resizing embeddings works and the new rows receive gradients. If resizing is
not supported, use existing reserved or rare tokens as markers and document
the choice. Pass: marker hidden states differ across options and the head
loss decreases on a 200-case overfit test.

## C3. Text parity (the gate)

Train NeoMME + marker head **text only** with the MK-SH-1 v1.1 recipe (E35
data mix, steps and seed 42; see `mokka-decisions/docs/experiments.md`).
Evaluate on the frozen MK-SH-1 sets with the E32 local CPU float32 harness:
testC, E-extrap, mono, task.

Reference values (MK-SH-1 v1.1, E35 s42): testC 0.979, E-extrap 0.875,
routing task as in the E35 report.

Decision rule, project thresholds fixed now:
- If NeoMME is within 5 pp of MK-SH-1 on testC **and** on task: **GO-A1**.
- If it is worse by more than 5 pp on either: **GO-A2** (hybrid becomes primary).
- One seed is enough for this gate; it is a routing decision, not a claim.
  The result is reported as exploratory.

## C4. Visual smoke

A1 on 2,000 synthetic doc_type cases (template-split dev of 400):
- Pass: dev accuracy above the majority-class rate, and swapping images
  between cases drops accuracy to near majority. If swapping does not hurt,
  the generator leaks the label through text and must be fixed before E38.

## Outcomes

`GO-A1`, `GO-A2`, or `STOP-A1` (hardware or marker failure), each with the
evidence files listed in `docs/status.md`.
