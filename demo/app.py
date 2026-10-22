"""MK-SH-V demo (Space Gjusev/MK-SH-V-demo, Gradio, CPU basic).

Canonical source: demo/ in the mk-sh-v repository. The decision path below
(load policy, one encoder pass over image + text, softmax(logits / T), apply
the frozen abstention gate) is the shipped E41/E42 behaviour and must not
change with the skin.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import snapshot_download

from mksh_v.arch_a1v import NeoMMEMarkerScorer, forward_a1v
from mokka_decisions.contracts import DecisionCase, Option

MODEL_ID = "Gjusev/MK-SH-V"
HERE = Path(__file__).parent

torch.set_num_threads(2)  # cpu-basic gives two vCPUs

# local checkout wins (tests and offline runs); the Space uses the Hub repo
PKG = Path(os.environ.get("MKSHV_MODEL_DIR") or snapshot_download(MODEL_ID))
POLICY = json.loads((PKG / "inference-policy.json").read_text(encoding="utf-8"))
T = POLICY["temperature"]["value"]        # fitted on cal_temperature (E42)
TAU = POLICY["abstention"]["threshold"]   # fitted on cal_policy (E42), CP-upper rule

_model = NeoMMEMarkerScorer()
_model.ensure_markers(32)
from safetensors.torch import load_file  # noqa: E402  (after ensure_markers)

_model.load_state_dict(load_file(str(PKG / "model.safetensors")), strict=True)
_model.eval()

FIGURE = (HERE / "figure" / "decision-gate.html").read_text(encoding="utf-8")

EXAMPLES_DIR = HERE / "examples"
EXAMPLES = [
    [str(EXAMPLES_DIR / "gui-notes-fin.png"),
     "Goal: Listen to a lecture on Artificial Intelligence and then jot down the name of the course in the notes app.",
     "Does the screenshot show the goal as completed?", "false: No\ntrue: Yes"],
    [str(EXAMPLES_DIR / "cord-receipt-1.png"),
     "Scanned document page.",
     "Is the total on this document at least 45.662 IDR?", "false: No\ntrue: Yes"],
    [str(EXAMPLES_DIR / "cord-receipt-2.png"),
     "Gescannte Dokumentseite.",
     "Beträgt die Gesamtsumme auf diesem Dokument mindestens 120.000 IDR?", "false: Nein\ntrue: Ja"],
]


def parse_options(text: str) -> tuple[Option, ...]:
    options = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        oid, _, desc = line.partition(":")
        if not _.strip() or not oid.strip():
            raise gr.Error("Each option needs the form 'id: description', one per line.")
        options.append(Option(id=oid.strip(), description=desc.strip()))
    if len(options) < 2:
        raise gr.Error("Give at least two options, one per line: 'id: description'.")
    if len(options) > 8:
        raise gr.Error("At most 8 options (the head has 32 markers; the demo caps at 8).")
    return tuple(options)


def decide(image, state_text: str, question: str, options_text: str):
    if image is None:
        raise gr.Error("Upload a document page or screenshot first.")
    state_text, question = state_text.strip(), question.strip()
    if not state_text or not question:
        raise gr.Error("The state and the question are both required.")
    options = parse_options(options_text)
    case = DecisionCase(
        id="demo", group_id="demo", source="demo", language="en", domain="documents",
        state=state_text, question=question, options=options)
    enc = _model.encode_cases([case], max_state_tokens=700, max_option_tokens=96)
    batch = _model.build_batch(enc, [image], "cpu", max_side=1024)
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = forward_a1v(_model, batch)
        probs = (logits / T).softmax(-1)[0].tolist()
    dt = time.perf_counter() - t0
    top = max(range(len(probs)), key=lambda i: probs[i])
    bars = [(options[i].id, options[i].description, probs[i]) for i in range(len(probs))]
    if probs[top] >= TAU:
        verdict = f"**choice: {options[top].id}** ({probs[top]:.0%} calibrated)"
        abstain = ""
    else:
        verdict = "**abstain**"
        abstain = (f"max calibrated probability {probs[top]:.0%} is below the gate "
                   f"tau = {TAU:.2f}; reason_code `below_validated_threshold`")
    bar_html = "<div class='prob-list'>" + "".join(
        f"<div class='prob-row'><span class='prob-id'>{oid}</span>"
        f"<span class='prob-bar'><span class='prob-fill' style='width:{p*100:.1f}%'></span></span>"
        f"<span class='prob-num'>{p:.1%}</span></div>"
        for oid, _desc, p in bars) + "</div>"
    meta = (f"one encoder pass · {dt:.1f} s on this CPU · T = {T:g}, tau = {TAU:g} "
            "(experiment E42, fitted on calibration splits)")
    return bar_html, verdict, abstain, meta


CSS = """
:root { --ground:#f4f4f2; --surface:#ffffff; --ink:#23272e; --muted:#5c636e;
        --line:#e3e1dc; --acc:#9a6b1f; --acc-ink:#7c5617; }
.gradio-container { max-width: 1060px !important; margin: 0 auto; }
footer { display: none !important; }
#masthead h1 { font-size: 30px; letter-spacing: -0.02em; margin: 0 0 6px; color: var(--ink); }
#masthead p { color: var(--muted); margin: 0; font-size: 15.5px; }
#gate-fig iframe, #gate-fig .html { width: 100%; border: 0; }
.prob-list { display: flex; flex-direction: column; gap: 10px; font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.prob-row { display: grid; grid-template-columns: 88px 1fr 64px; align-items: center; gap: 12px; }
.prob-id { color: var(--ink); font-size: 14px; overflow: hidden; text-overflow: ellipsis; }
.prob-bar { background: var(--line); height: 14px; position: relative; }
.prob-fill { position: absolute; inset: 0 auto 0 0; background: var(--acc-ink); }
.prob-num { color: var(--muted); font-size: 13.5px; text-align: right; }
#verdict { font-size: 22px; letter-spacing: -0.01em; color: var(--ink); }
#verdict .abstain { color: var(--acc-ink); }
#meta { color: var(--muted); font-size: 13px; }
"""


def build() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="MK-SH-V · visual decisions with a gate") as app:
        with gr.Column(elem_id="masthead"):
            gr.HTML("<h1>MK-SH-V</h1>"
                    "<p>A 260M visual decision model: a page or screenshot in, calibrated "
                    "probabilities out, a choice, and a refusal below the gate. "
                    "Every number on this page comes from the shipped policy.</p>")
        gr.HTML(f'<iframe src="/file=figure/decision-gate.html" title="The decision gate" '
                f'style="width:100%;height:560px;border:0;border-radius:8px;" loading="lazy"></iframe>')
        with gr.Row():
            with gr.Column(scale=5):
                image_in = gr.Image(type="pil", label="Image (document page or screenshot)",
                                    height=340)
                state_in = gr.Textbox(label="State", value=EXAMPLES[0][1], lines=3)
                question_in = gr.Textbox(label="Question", value=EXAMPLES[0][2])
                options_in = gr.Textbox(label="Options (one per line: id: description)",
                                        value=EXAMPLES[0][3], lines=3)
                go = gr.Button("Decide", variant="primary")
            with gr.Column(scale=4):
                probs_out = gr.HTML(label="Calibrated probabilities")
                verdict_out = gr.Markdown(elem_id="verdict")
                abstain_out = gr.Markdown()
                meta_out = gr.Markdown(elem_id="meta")
        gr.Examples(examples=EXAMPLES, inputs=[image_in, state_in, question_in, options_in],
                    label="Try a real case")
        gr.Markdown(
            "Weights and policy: [Gjusev/MK-SH-V](https://huggingface.co/Gjusev/MK-SH-V). "
            "The gate was validated on held-out data: over 86 % coverage with a 95 % "
            "Clopper-Pearson upper bound on selective risk of 3.2 to 3.4 % on the synthetic "
            "test (experiment E42); on real images abstention is the declared open problem, "
            "and this demo shows it unfiltered. Example images: CORD (Park et al. 2019, "
            "NAVER Clova) and GUI-Odyssey (Lu et al. 2024, OpenGVLab), both CC BY 4.0.")
        go.click(decide, [image_in, state_in, question_in, options_in],
                 [probs_out, verdict_out, abstain_out, meta_out])
    return app


if __name__ == "__main__":
    gr.set_static_paths([str(HERE / "figure")])
    build().launch()
