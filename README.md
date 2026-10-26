<table><tr>
<td><img src="assets/logo-mark.png" width="130" alt="MK-SH-V mark: the family's whale shark over a tide band in the amber of the gate"></td>
<td><h1>MK-SH-V</h1><p><sub>a 260M visual decision model: calibrated probabilities, a choice, and a refusal below the gate</sub></p></td>
</tr></table>

A small open **visual decision model**. Give it a document page or a
screenshot, a question and the options; it returns calibrated probabilities
and a choice, or it refuses when its confidence sits below a gate whose risk
was validated on held-out data. The visual sibling of
[MK-SH-1](https://github.com/Gjusev/MK-SH-1).

<p align="center">
  <a href="https://huggingface.co/Gjusev/MK-SH-V"><b>Model</b></a> ·
  <a href="https://huggingface.co/spaces/Gjusev/MK-SH-V-demo"><b>Live demo</b></a> ·
  <a href="#use-it"><b>Use it</b></a> ·
  <a href="#measured-against-the-field"><b>Results</b></a> ·
  <a href="https://github.com/Gjusev/MK-SH-1"><b>MK-SH-1</b></a>
</p>

| 0.828 | τ = 0.55 | 98.6 % |
|:---|:---|:---|
| accuracy on real screenshots, held out; a ~15× larger model run zero-shot scores 0.731 | the refusal gate, fitted on a calibration split by a rule frozen in advance | answered on the held-out synthetic test, with a 95 % Clopper-Pearson risk bound of 4.1 % |

## What it does

Three real cases, run through the released weights and the shipped policy:

| Input | Question | Calibrated probabilities | Output |
|---|---|---|---|
| <img src="demo/examples/gui-notes-fin.png" width="72" alt="phone screenshot of a chat with meeting links, from GUI-Odyssey"> | Does the screenshot show the goal as completed? | true **0.998**, false 0.002 | **choice: true** |
| <img src="demo/examples/cord-receipt-1.png" width="72" alt="photographed receipt with a total of 302.016 IDR, from CORD"> | Is the total on this document at least 45.662 IDR? | true **0.978**, false 0.022 | **choice: true** |
| <img src="demo/examples/cord-receipt-1.png" width="72" alt="the same receipt"> | Which range does the total fall into? (below 100.000, 100.000 to 200.000, above 200.000 IDR) | low **0.757**, the other two 0.243 together | **choice: low**, wrong |

The receipt's total is 302.016 IDR. The E41 release refused the second
question at 0.507; this one reads the total and answers it. The third row is
where it stops: the right range is "above 200.000", and the model picked
"below 100.000" with a confidence that clears the gate. On real receipts
about a quarter of the answers the gate lets through are wrong (see [the gate,
held out](#the-refusal-gate-held-out)); the gate's risk budget is validated
on the synthetic test only.

How a decision is made: one encoder pass reads the image and the text
together, each option is marked inline and scored by a shared head, and the
softmax runs over the valid options only. The temperature T only rescales
confidence, so it never changes a choice. The gate tau was fitted on a
calibration split (E48, by the E42 protocol) as the smallest threshold whose
Clopper-Pearson 95 % upper bound on selective risk stays within a 5 % budget.
Images whose longest side is under 1,024 px are first enlarged to it
(Lanczos), exactly as in training and calibration (E46). It is **not** a chat
model and generates no text.

```mermaid
flowchart LR
    A["image + state + question + options"] --> B["NeoMME-260M single tower<br/>one pass, pixels and text"]
    B --> C["marker head, one [O_i] per option"]
    C --> D["option scores"]
    D --> E["softmax(logits / T)"]
    E --> F{"max probability >= tau"}
    F -- yes --> G["choice + calibrated probabilities"]
    F -- no --> H["choice = null<br/>reason_code = below_validated_threshold"]
```

## Use it

```bash
pip install "mksh-v @ git+https://github.com/Gjusev/mk-sh-v"
```

```python
import json, torch
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from mokka_decisions.contracts import DecisionCase, Option
from mksh_v.arch_a1v import NeoMMEMarkerScorer, forward_a1v

repo = "Gjusev/MK-SH-V"
policy = json.load(open(hf_hub_download(repo, "inference-policy.json")))
T, tau = policy["temperature"]["value"], policy["abstention"]["threshold"]

model = NeoMMEMarkerScorer()                 # NeoMME-260M backbone, marker head
model.ensure_markers(32)
model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")), strict=True)
model.eval()

case = DecisionCase(
    id="q1", group_id="q1", source="me", language="en", domain="screens",
    state="Goal: print orders_nov.csv for the records",
    question="Does the screenshot show the goal as completed?",
    options=(Option(id="false", description="No"), Option(id="true", description="Yes")))
enc = model.encode_cases([case], max_state_tokens=700, max_option_tokens=96)
batch = model.build_batch(enc, [Image.open("screenshot.png").convert("RGB")], "cpu",
                          max_side=1024, upscale_to=1024)   # the input rule in inference-policy.json
with torch.inference_mode():
    probs = (forward_a1v(model, batch) / T).softmax(-1)[0].tolist()
top = max(range(len(probs)), key=probs.__getitem__)
choice = case.options[top].id if probs[top] >= tau else None   # None = below_validated_threshold
```

About one second per decision on a laptop CPU. Up to 8 options per question
in the demo (the head has 32 markers). Evaluated in English, Spanish, German
and French (French synthetic test 0.930 and 0.919, its gate outside budget);
the real-image questions are in English, Spanish and German.

## Measured against the field

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/chart-accuracy-dark.png">
  <img src="assets/chart-accuracy.png" alt="Accuracy on real held-out images. Screenshots: MK-SH-V 0.828 (shipped seed 43) and 0.800, jev-vlm-4b 0.731, OCR plus MK-SH-1 0.472, synthetic only 0.493. Receipts: MK-SH-V 0.734 and 0.734, jev-vlm-4b 0.972, OCR 0.486, synthetic only 0.394. All real: 0.807 and 0.786 against 0.785, 0.475 and 0.471.">
</picture>

Every arm was run by us on the same 2,586 real held-out cases with pinned
revisions; labels come from each source's ground truth. On real screenshots
the 260M model leads a ~15× larger zero-shot model; on receipt thresholds it
trails it (0.734 against 0.972) and sits above a deterministic rules baseline
(0.594). Overall the two are level (0.807 and 0.786 against 0.785); no paired
test was run for this checkpoint, so read that as a tie. That comparator may
have seen these sources in pretraining; its card cannot tell us.

| Arm | Params | Real overall | Receipts (CORD) | Screenshots (GUI) |
|---|---|---|---|---|
| **MK-SH-V s43 (shipped)** | **260M** | **0.807** | 0.734 | **0.828** |
| **MK-SH-V s42** | **260M** | **0.786** | 0.734 | **0.800** |
| jev-vlm-4b, zero-shot | ~4B | 0.785 | 0.972 | 0.731 |
| OCR → MK-SH-1 | ~25M | 0.475 | 0.486 | 0.472 |
| Same model, synthetic only (E39) | 260M | 0.471 | 0.394 | 0.493 |

How it got here. E41 added real data to the synthetic-only model (+29.4 pp
real accuracy, CI95 [+25.9, +32.9], seed 42, paired by episode). E44 to E47
fine-tuned that checkpoint for printed numbers: E46 enlarges every image to
1,024 px before the encoder, E47 adds digit-reading data. E48 confirmed the
numeric result on 3,000 fresh documents and refitted the gate. Against the
E41 release the synthetic test rises from 0.908 and 0.909 to 0.959 on both
seeds and real receipts from 0.550 and 0.602 to 0.734. Text competence from
MK-SH-1 is kept through replay: testC 1.000, routing task 0.830 and 0.833.
Criteria for every step: `docs/E44_CRITERIA_*.md` to `docs/E48_CRITERIA_*.md`.

## The refusal gate, held out

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/chart-gate-dark.png">
  <img src="assets/chart-gate.png" alt="Selective risk at each seed's fitted tau (0.55 for the shipped seed 43, 0.30 for seed 42) with exact 95 percent Clopper-Pearson intervals per held-out set. Synthetic test: 3.5 and 4.1 percent, upper bounds 4.1 and 4.8 percent, inside the 5 percent budget. French synthetic, fresh numeric documents, real screenshots 16.1 and 20.0 percent, real receipts 25.3 and 26.6 percent and text routing: outside the budget.">
</picture>

> On our held-out synthetic evaluation, both 260M checkpoints answered at
> least 98.6 % of cases while maintaining a 95 % Clopper-Pearson upper
> confidence bound on selective risk below 5 % (4.1 % and 4.8 %).

That sentence is the whole claim, and the chart shows where it stops. With
the shipped seed, the gate keeps 97 % of real screenshots while 16 % of the
kept answers are wrong, 89 % of real receipts with 25 % wrong, and 88 % of
the fresh numeric documents with 23 % wrong. The confidence signal that
separates right from wrong on the synthetic distribution does not yet do so
on real images. That is the campaign's open problem, published with its
per-set tables (`runs/e48/`), not hidden. A screens-specific gate (E43, on
the E41 checkpoint) held the budget on one seed and missed on the other, so
real-image abstention stays open.

## Use cases

Each case names the measured capability it rests on, and where the
measurement stops.

1. **End-of-task screen detection over screenshots.** "Does this screenshot
   show the goal as completed": agent supervision, UI QA, step verification.
   Rests on GUI-Odyssey held-out pairs, 0.828 and 0.800. Bound, from the
   goal-swap diagnostic (each final screenshot paired with an unrelated goal):
   E47 rejects the wrong goal 0.775 and 0.785 of the time, up from 0.12 and
   0.135 in E41 but under the 0.80 bar frozen for a goal claim. Use it as
   end-of-task screen detection, not goal verification. Also binary questions
   in EN, ES and DE from that corpus; not a general GUI understanding model,
   and the gate is not validated on real screens.
2. **Document state decisions.** Document type, form completeness, a ticket's
   urgency, conflicting sources. Rests on the E38 synthetic families
   (doc_type, form_complete, screen_state, source_conflict, task_done and
   urgency at 0.995 to 1.000 held out, seed 42). **Printed numeric thresholds** ("is the total at least
   X") are now a measured capability with a ceiling: 0.718 and 0.725 on the
   synthetic test, 0.737 and 0.760 on 3,000 fresh documents nobody had read
   (E48, criteria frozen before generation), and 0.734 on real receipts
   against a rules baseline at 0.594. Bound: fine gaps under 1 % between total
   and threshold remain the hard part, the fresh set shares the test
   templates, and the gate is not validated for this family.
3. **Calibrated scoring with a safety valve.** Decision queues where refusals
   route to a human. Rests on the E48 held-out result above. Bound: the
   guarantee holds on the evaluated synthetic distribution only.
4. **Mixed text and visual queues.** Rests on the retained text sets (testC
   1.000, task 0.830 and 0.833). Bound: descriptive; the shipped policy was
   fitted on the visual calibration split.

## Rigor

The claims are only as good as the record, and the record includes its
corrections:

- **Criteria frozen before measurement, every time.** E37 to E48 criteria are
  committed before any job runs (`docs/E*_CRITERIA_*.md`); amendments land as
  commits with their evidence, before results are read.
- **A mislabel corrected in public.** E39's kernel printed "transfer
  supported" on a statistical tie; the frozen criteria and the report say
  "advantage only on synthetic, no evidence of transfer", and that is what the
  record keeps. The tie motivated E41.
- **A wrong rule, replaced before it was used again.** E40's threshold rule
  (point risk) landed on the budget edge; the CP upper-bound rule was frozen
  for E42 before any E41 calibration output existed, and the cross-check
  against the old rule ships in every E42 summary.
- **The evaluation that guided the fix is disclosed.** The E38c real set was
  inspected for error analysis before E41 designed its training data; it was
  never trained or tuned on, and all selection used separate dev splits.
- **The shipped seed, and why.** Seed 42's fitted tau is 0.30, the bottom of
  the grid, so its gate never refuses a two-option question. Seed 43 (tau
  0.55) keeps a working gate and ships. The reason is that calibration-split
  property, not its held-out accuracy.
- **Comparators measured, not quoted.** jev-vlm-4b and OCR → MK-SH-1 were run
  by us on the same rows with pinned revisions.

Every number regenerates from per-case files under `runs/` with the scripts
in `scripts/`. Statistics: episode-grouped paired bootstrap (1,000 resamples,
seed 42); risk intervals exact two-sided Clopper-Pearson with the budget on
the upper end.

## Repository layout

```
docs/        frozen criteria per experiment, licence review
src/mksh_v/  model, generators, splitting, statistics
kaggle/      kernel scripts (self-contained builds via build_kernels.py)
scripts/     generators, verifiers, post-hoc gate reports, HF push
runs/        per-case outputs of every experiment (release bundle)
release/     inference-policy.json and export artifacts
demo/        the interactive demo (Space source)
assets/      mark, charts, film; sources and build scripts in assets/src/
tests/       unit tests incl. every kernel's math under test
```

## Provenance and licences

Code and weights: Apache-2.0. Training data: our synthetic generator
(Apache-2.0) plus CORD v2 train and GUI-Odyssey task_split train (both CC BY
4.0, attribution kept, images byte-identical, labels re-derived from the
sources' ground truth with rules committed before generation). The film's
soundtrack is generated in code; its effects are Kenney (CC0); type is Geist
(SIL OFL 1.1). Full tables: `THIRD_PARTY_NOTICES.md` and
`docs/data-licenses.md`.

Previous release: the E41 checkpoint (seed 42, E42 policy) and its
[launch film](https://cdn.jsdelivr.net/gh/Gjusev/mk-sh-v@main/assets/mkshv-film.mp4),
whose numbers are those of that release.

Sibling text model: [MK-SH-1](https://github.com/Gjusev/MK-SH-1) (Hugging
Face: [Gjusev/MK-SH-1](https://huggingface.co/Gjusev/MK-SH-1)).
