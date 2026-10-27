---
title: MK-SH-V demo
emoji: ⚖️
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 5.49.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: 260M visual decision model with a validated abstention gate
models:
  - Gjusev/MK-SH-V
  - Hcompany/NeoMME-260M
---

# MK-SH-V demo

Upload a document page or a screenshot, describe the state and the question,
list the options, and get calibrated probabilities, a choice, and an explicit
abstention when the top calibrated probability sits below the gate
(tau = 0.55, experiment E48; E47 checkpoint, seed 43).

- Model and policy: [Gjusev/MK-SH-V](https://huggingface.co/Gjusev/MK-SH-V)
- Twelve cases load and decide in one click (train-split images, every
  answer checked against gold; a refusal and a confident wrong answer are
  kept and labelled as failures), and any
  image can be uploaded to test. The decision contract (one tower, one score
  per option, a temperature, then the gate) is documented in the repo README.
- Held-out validation of the gate: 98.6 % coverage with a 95 %
  Clopper-Pearson upper bound on selective risk of 4.1 % on the synthetic
  test (4.8 % on the other seed). On real images the gate keeps most cases
  while 16 to 25 % of the kept answers are wrong; that limitation ships with
  the model and this demo shows it unfiltered.
- Example images: CORD v2 train (Park et al. 2019, NAVER Clova, CC BY 4.0)
  and GUI-Odyssey task_split train (Lu et al. 2024, OpenGVLab, CC BY 4.0),
  resized for the demo.
## Amber Checkpoint (playable tab)

A four-day border-booth game where the model is the decision assistant.
Every check on screen is a measured capability of this checkpoint, played in
its measured framing: document type, form completeness, reading the sender
against the day's manifest, and totals against a 5,000.00 VEL cap with fine
gaps. Below the gate (tau 0.55) the machine refuses to advise and routes the
document to you; the stamps are yours. Documents are synthetic, printed by
our generator; the end screen reports player and model accuracy side by side.
