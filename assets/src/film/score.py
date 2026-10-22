"""Code-made soundtrack for the MK-SH-V film, on the film's own cues.

    python score.py            # -> work/soundtrack_raw.wav (then ffmpeg loudnorm in mux.sh)

120 BPM, A minor (Am F C G per bar). Music enters on frame 0 under the body,
the beat comes in on bar 3, drops out on the refusal (the stall), rebuilds,
lands its loudest moment on the risk bound, and stops on the bar line where
the lockup forms; one clean hit when the wordmark becomes readable, then a
tail. Effects are Kenney (CC0), one per visible cause, never one per word of
film copy. Every cue frame comes from cues.json (film_build.py).
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
SFX = Path.home() / ".claude/skills/brag/assets/sfx"
SR = 48000
CUES = json.loads((HERE / "cues.json").read_text(encoding="utf-8"))["cue"]
END = CUES["end"] / 60 + 1.6
N = int(END * SR)
BEAT = 0.5
f2s = lambda f: f / 60

rng = np.random.default_rng(42)
mix = np.zeros((N, 2))


def add(sig, t, gain=1.0, pan=0.0):
    i = int(round(t * SR))
    if i >= N or i + len(sig) <= 0:
        return
    sig = sig[: N - i]
    l, r = gain * np.sqrt(0.5 * (1 - pan)), gain * np.sqrt(0.5 * (1 + pan))
    mix[i:i + len(sig), 0] += sig * l
    mix[i:i + len(sig), 1] += sig * r


def env(n, a=0.005, d=0.2):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / d)


def lp(x, hz, order=2):
    return sosfilt(butter(order, hz, "low", fs=SR, output="sos"), x)


def hp(x, hz, order=2):
    return sosfilt(butter(order, hz, "high", fs=SR, output="sos"), x)


NOTE = lambda m: 440 * 2 ** ((m - 69) / 12)
CHORDS = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]  # Am F C G
chord_at = lambda t: CHORDS[int(t // 2) % 4]


def kick(gain=1.0):
    n = int(0.32 * SR); t = np.arange(n) / SR
    f = 45 + 75 * np.exp(-t / 0.035)
    return gain * np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.11)


def hat():
    n = int(0.06 * SR)
    return hp(rng.standard_normal(n), 7000) * env(n, 0.0005, 0.018) * 0.35


def bass(m, dur=0.45, cutoff=420):
    n = int(dur * SR); t = np.arange(n) / SR
    saw = 2 * ((t * NOTE(m)) % 1) - 1
    return lp(saw, cutoff) * env(n, 0.004, 0.16) * 0.5


def pluck(m, dur=0.35, bright=2400):
    n = int(dur * SR); t = np.arange(n) / SR
    tri = 2 * np.abs(2 * ((t * NOTE(m)) % 1) - 1) - 1
    return lp(tri, bright) * env(n, 0.002, 0.09) * 0.3


def pad(ms, dur, cutoff=900):
    n = int(dur * SR); t = np.arange(n) / SR
    s = sum(np.sin(2 * np.pi * NOTE(m) * t * (1 + det)) for m in ms for det in (-0.003, 0.003))
    a = np.minimum(1, t / 0.25) * np.minimum(1, (dur - t) / 0.3)
    return lp(s, cutoff) * a * 0.06


def ogg(name):
    out = WORK / ("sfx_" + name.replace("/", "_") + ".wav")
    if not out.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(SFX / name), "-ac", "1", "-ar", str(SR), str(out)], check=True)
    sr, x = wavfile.read(out)
    return x.astype(np.float64) / 32768.0


STOP = f2s(CUES["iris"])                         # music stops on this bar line
DROP_OUT = (f2s(CUES["stall"]), f2s(CUES["f4"]))  # the refusal: beat drops out
BOUND = f2s(CUES["bound"])                       # loudest moment

# pads under everything until the stop (frame 0 entry, quieter in the first two bars)
for bar in range(int(STOP // 2)):
    t0 = bar * 2.0
    g = 0.55 if t0 < 4 else (1.15 if t0 >= BOUND - 0.01 else 0.9)
    cut = 700 if t0 < 4 else min(2600, 900 + 160 * bar)
    add(pad(chord_at(t0), 2.05, cut), t0, g)

beats = np.arange(0, STOP, BEAT)
for t in beats:
    if DROP_OUT[0] <= t < DROP_OUT[1]:
        continue
    bar = int(t // 2)
    if t >= 4.0:                                      # beat enters on bar 3
        add(kick(1.25 if t >= BOUND - 0.01 else 0.95), t, 0.9)
        add(bass(chord_at(t)[0] - 24, cutoff=380 if t < 14 else 650), t, 0.8)
        add(bass(chord_at(t)[0] - 12, 0.2, cutoff=500), t + 0.25, 0.35)
    if t >= 2.0:                                      # hats from bar 2, offbeats
        add(hat(), t + 0.25, 0.8 if t < BOUND else 1.0, pan=0.3)
    if t >= 14.0:                                     # build: arpeggio from act D
        ch = chord_at(t)
        for k, m in enumerate((ch[0] + 12, ch[1] + 12, ch[2] + 12, ch[1] + 12)[:2]):
            add(pluck(m, bright=1500 + 120 * (t - 14)), t + 0.25 * k, 0.6 + (0.3 if t >= BOUND else 0), pan=-0.25 + 0.5 * k)

# a held low note through the refusal, so the drop-out is a breath, not a hole
add(pad((45, 52), DROP_OUT[1] - DROP_OUT[0] + 0.2, 500), DROP_OUT[0], 1.4)

# ---- effects, each with a visible cause ----
def typed(start, text, cps=30):                       # one key tick per word, at its first char
    pos = 0
    for w in text.split(" "):
        add(ogg("keyboard/" + KEY[pos % len(KEY)]) * 0.5, f2s(start) + pos / cps, 0.5)
        pos += len(w) + 1

KEY = sorted(p.name for p in (SFX / "keyboard").glob("*.wav"))[:6]
typed(CUES["type1"], "Is the goal done?")
typed(CUES["type2"], "total at least 45.662 IDR?")
add(ogg("impact/impactSoft_medium_001.ogg"), f2s(CUES["choice"]), 0.9)           # chip lands
add(ogg("interface/bong_001.ogg"), f2s(CUES["status"] + 28), 0.8)               # "abstain" appears
add(ogg("interface/click_003.ogg"), f2s(CUES["chip"]), 0.9)                      # delta chip lands
add(ogg("impact/impactSoft_heavy_003.ogg"), BOUND, 1.3)                          # the bound lands
for fr in (CUES["f3"] + 38, CUES["f5"] + 38, CUES["f7"] + 38, CUES["f8"] + 38):  # one soft sound per resolved line
    add(ogg("ui/rollover2.ogg"), f2s(fr), 0.35)

# lockup: silence before it, then ONE clean hit as the wordmark becomes readable, then a tail
HIT = f2s(CUES["word"] + 20)
add(ogg("impact/impactSoft_heavy_001.ogg"), HIT, 1.2)
add(pad((45, 57, 64), 2.4, 1200) * np.exp(-np.arange(int(2.4 * SR)) / SR / 0.9), HIT, 2.2)

# gentle fades at the ends (no click), normalise peak; loudness is set in the mux
fade = int(0.02 * SR); mix[:fade] *= np.linspace(0, 1, fade)[:, None]
tail = int(0.8 * SR); mix[-tail:] *= np.linspace(1, 0, tail)[:, None] ** 2
mix /= np.max(np.abs(mix)) * 1.12
wavfile.write(WORK / "soundtrack_raw.wav", SR, (mix * 32767).astype(np.int16))
print("soundtrack_raw.wav", round(N / SR, 2), "s; stop", STOP, "bound", BOUND, "hit", round(HIT, 2))
