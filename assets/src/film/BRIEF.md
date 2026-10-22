# MK-SH-V launch film: brief

Spec: 1920x1080@60, 26 s (1560 f), 120 BPM (beat 30 f, bar 120 f), pace: crisp,
audio: code-made score (numpy) + Kenney CC0 effects, master -14 LUFS.

Product: MK-SH-V, a 260M visual decision model. Image + state + question +
options in; calibrated probabilities, a choice, or an explicit refusal below a
validated gate (tau 0.55) out. Shown on: GitHub README (muted loop), Hugging
Face card, LinkedIn.

Engine: HTML + one pure render(frame) function (seekable, deterministic),
captured frame by frame with Playwright (local Chrome), encoded with ffmpeg.
cinetic's Remotion/HyperFrames starters were not installed (no npm install in
this environment); its laws, grid, pace and review loop apply unchanged.

## Hard bans (cinetic, verbatim intent)
No eyebrows or kickers; no "Introducing"; no stacked taglines; copy only as
fragments of at most 5 words, one at a time, picture between them; no serif,
no italic; no neon, glow, bloom, purple, multi-hue gradients, glassmorphism;
no emoji, stock icons, sparkles, particles, confetti, lens flares; no
overshoot except true landings; no per-letter tricks on copy.

## Brand-supplied exceptions
- Amber #E8A13C is MK-SH-V's tau: the gate mark in the product, the eyes and
  tide band of the whale shark mark (the family mark, recoloured). It is the accent and means one thing: the gate.
  // cinetic:brand-supplied amber
- Geist is the family typeface (MK-SH-1 uses it); chosen on purpose.

## Truth
Every number on screen is measured: model outputs captured from the release
weights (real_cases.json), accuracy from runs/e41/e41_report.json, risk from
runs/e42-calib/e42_summary.json. Synthetic-only claims are labelled as such.
