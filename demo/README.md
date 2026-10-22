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
(tau = 0.55, experiment E42).

- Model and policy: [Gjusev/MK-SH-V](https://huggingface.co/Gjusev/MK-SH-V)
- Four real cases load and decide in one click. The figure below the
  workbench explains the decision contract: one tower, marker pins,
  a probability dial, and the gate. Its three cases are real calibration
  cases with their measured probabilities.
- Held-out validation of the gate: over 86 % coverage with a 95 %
  Clopper-Pearson upper bound on selective risk of 3.2 to 3.4 % on the
  synthetic test (both seeds). On real images the gate either keeps confident
  errors or refuses nearly everything; that limitation ships with the model
  and this demo shows it unfiltered.
- Example images: CORD v2 train (Park et al. 2019, NAVER Clova, CC BY 4.0)
  and GUI-Odyssey task_split train (Lu et al. 2024, OpenGVLab, CC BY 4.0),
  resized for the demo.
