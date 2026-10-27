"""MK-SH-V demo (Space Gjusev/MK-SH-V-demo, Gradio, CPU basic).

Canonical source: demo/ in the mk-sh-v repository. The decision path below
(load policy, one encoder pass over image + text, softmax(logits / T), apply
the frozen abstention gate) is the shipped E47/E48 behaviour, including the
E46 input rule (upscale to 1,024 px), and must not change with the skin.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gradio as gr
import torch
from html import escape
from huggingface_hub import snapshot_download

from mksh_v.arch_a1v import NeoMMEMarkerScorer, forward_a1v
from mokka_decisions.contracts import DecisionCase, Option

MODEL_ID = "Gjusev/MK-SH-V"
HERE = Path(__file__).parent

torch.set_num_threads(2)  # cpu-basic gives two vCPUs

# local checkout wins (tests and offline runs); the Space uses the Hub repo
PKG = Path(os.environ.get("MKSHV_MODEL_DIR") or snapshot_download(MODEL_ID))
POLICY = json.loads((PKG / "inference-policy.json").read_text(encoding="utf-8"))
T = POLICY["temperature"]["value"]        # fitted on cal_temperature (E48, E42 protocol)
TAU = POLICY["abstention"]["threshold"]   # fitted on cal_policy (E48), CP-upper rule

_model = NeoMMEMarkerScorer()
_model.ensure_markers(32)
from safetensors.torch import load_file  # noqa: E402  (after ensure_markers)

_model.load_state_dict(load_file(str(PKG / "model.safetensors")), strict=True)
_model.eval()

FIGURE = (HERE / "figure" / "decision-gate.html").read_text(encoding="utf-8")

from game import ACP_CSS, build_tab  # noqa: E402  (the playable Amber Checkpoint tab)

GAME_FONTS = ("<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=Special+Elite&display=swap'>")

EXAMPLES_DIR = HERE / "examples"
# The case set: train-split images picked by assets/src/pick_demo_cases.py and run
# through these weights and policy (favourable answers verified against gold,
# refusals kept for honesty). Regenerate with pick_demo_cases.py + materialize_cases.py.
_CASES_FILE = HERE / "cases.json"
CASES_DB = json.loads(_CASES_FILE.read_text(encoding="utf-8")) if _CASES_FILE.exists() else []
EXAMPLES = [[str(HERE / c["image"]), c["state"], c["question"],
             "\n".join(f"{oid}: {desc}" for oid, desc in c["options"])] for c in CASES_DB]
ANSWERS = [i for i, c in enumerate(CASES_DB) if c["verdict"] == "answer" and c["top"] == c["gold"]]
# failures shown, not hidden: refusals below tau and confident errors above it
FAILURES = [(i, "refused") for i, c in enumerate(CASES_DB) if c["verdict"] == "abstain"] + \
           [(i, "wrong") for i, c in enumerate(CASES_DB) if c["verdict"] == "answer" and c["top"] != c["gold"]]


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
    batch = _model.build_batch(enc, [image], "cpu", max_side=1024, upscale_to=1024)
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = forward_a1v(_model, batch)
        probs = (logits / T).softmax(-1)[0].tolist()
    dt = time.perf_counter() - t0
    top = max(range(len(probs)), key=lambda i: probs[i])
    tick = f"<span class='tau' style='left:{TAU * 100:.1f}%'></span>"
    rows = "".join(
        f"<div class='prob-row{' top' if i == top else ''}'><span class='prob-id'>{escape(o.id)}</span>"
        f"<span class='prob-bar'><span class='prob-fill' style='width:{p * 100:.1f}%'></span>{tick}</span>"
        f"<span class='prob-num'>{p:.1%}</span><span class='prob-desc'>{escape(o.description)}</span></div>"
        for i, (o, p) in enumerate(zip(options, probs)))
    bar_html = (f"<div class='prob-list'><div class='prob-head'><span>calibrated probability</span>"
                f"<span class='tau-key'>&tau; {TAU:g}</span></div>{rows}</div>")
    if probs[top] >= TAU:
        verdict = (f"<div class='verdict'><span class='v-label'>choice:</span>"
                   f"<span class='v-word'>{escape(options[top].id)}</span>"
                   f"<span class='v-note'>{probs[top]:.1%} &ge; &tau; {TAU:g}</span></div>")
        abstain = ""
    else:
        verdict = ("<div class='verdict abstain'><span class='v-word'>abstain</span>"
                   f"<span class='v-note'>top {probs[top]:.1%} &lt; &tau; {TAU:g}</span></div>")
        abstain = ("<div class='reason'>choice = null &middot; reason_code "
                   "<code>below_validated_threshold</code></div>")
    meta = (f"one encoder pass &middot; {dt:.1f} s on this CPU &middot; T = {T:g}, tau = {TAU:g} "
            "(E48, fitted on calibration splits)")
    return bar_html, verdict, abstain, meta


CASES = [  # (thumbnail label, what the model does with it) for the one-click cards
    ("Screenshot", "is the goal done?"),
    ("Receipt", "total at least 45.662 IDR?"),
    ("Beleg, German", "Summe mindestens 120.000 IDR?"),
    ("Receipt, 3 options", "which range is the total in?"),
]

EMPTY = ("<div class='verdict idle'><span class='v-word'>waiting</span>"
         "<span class='v-note'>pick a real case on the left, or upload your own page</span></div>"
         "<div class='prob-list ghost'><div class='prob-head'><span>calibrated probability</span>"
         "<span class='tau-key'>&tau; 0.55</span></div>"
         + "".join("<div class='prob-row'><span class='prob-id'>&nbsp;</span><span class='prob-bar'>"
                   "<span class='tau' style='left:55%'></span></span><span class='prob-num'>&nbsp;</span>"
                   "<span class='prob-desc'>&nbsp;</span></div>" for _ in range(2)) + "</div>")

CSS = """
:root { --ink:#14181F; --muted:#5C6470; --line:#E3E5E9; --tint:#F4F5F7; --navy:#1B2A41; --navy2:#24354F;
        --paper:#F7F6F3; --pale:#B7C2D0; --acc:#E8A13C; --acc-ink:#9A6416; }
.gradio-container { max-width: 1240px !important; margin: 0 auto; }
footer { display: none !important; }
/* the bar: the product's face and its three measured numbers, set as type */
#bar { display:grid; grid-template-columns:auto 1fr auto; align-items:center; gap:18px 22px; padding:10px 0 6px; }
#bar img { width:60px; height:60px; border-radius:14px; display:block; }
#bar h1 { font-size:26px; letter-spacing:-.025em; margin:0; color:var(--ink); line-height:1.1; }
#bar p { margin:3px 0 0; color:var(--muted); font-size:15px; }
#bar .nums { display:flex; gap:30px; }
#bar .num b { display:block; font-size:22px; letter-spacing:-.02em; color:var(--ink); font-variant-numeric:tabular-nums; }
#bar .num span { font-size:12.5px; color:var(--muted); }
#bar .num.gate b { color:var(--acc-ink); }
#bar .links { grid-column:2 / 4; display:flex; gap:20px; font-size:14px; }
#bar .links a { color:var(--ink); text-decoration:none; border-bottom:1px solid var(--line); padding-bottom:1px; }
#bar .links a:hover, #bar .links a:focus-visible { border-color:var(--ink); }
/* one-click real cases */
#cases { display:grid !important; grid-template-columns:1fr 1fr 1fr; gap:10px !important; }
#cases > * { min-width:0 !important; }
#cases button { display:flex !important; flex-direction:row !important; flex-wrap:nowrap !important; align-items:center; width:100%; justify-content:flex-start; text-align:left; gap:12px; padding:10px 12px; min-height:76px;
                background:#fff; border:1px solid var(--line); border-radius:12px; font-weight:500; color:var(--ink);
                white-space:normal; line-height:1.3; font-size:14px; transition:border-color .15s, transform .15s; }
#cases button img { width:52px !important; height:52px !important; max-height:none !important; object-fit:cover; border-radius:8px; flex-shrink:0; }
#cases button:hover { border-color:var(--ink); }
#cases button:active { transform:translateY(1px); }
#cases-label { font-size:13.5px; color:var(--muted); margin:4px 0 -4px; }
.group-label { font-size:13px; font-weight:600; color:var(--ink); margin:10px 0 -2px; }
.caveat { font-size:12.5px; color:var(--muted); margin:2px 0 0; line-height:1.45; }
#cases2 { display:grid !important; grid-template-columns:1fr 1fr; gap:10px !important; }
#cases2 > * { min-width:0 !important; }
#cases2 button { display:flex !important; flex-direction:row !important; align-items:center; width:100%;
                 justify-content:flex-start; text-align:left; gap:12px; padding:8px 12px; min-height:64px;
                 background:#fff; border:1px solid var(--line); border-radius:12px; font-weight:500;
                 color:var(--ink); white-space:normal; line-height:1.3; font-size:14px; }
#cases2 button img { width:44px !important; height:44px !important; max-height:none !important;
                     object-fit:cover; border-radius:8px; flex-shrink:0; }
#cases2 button:hover { border-color:var(--ink); }
/* the readout: the gate, in the navy of the mark's tile */
#decision { background:var(--navy); border-radius:16px; padding:24px 24px 18px; color:var(--paper);
            position:sticky; top:14px; align-self:flex-start; }
#decision .prose, #decision p { color:var(--paper); }
.verdict { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; min-height:52px; }
.v-label { color:var(--pale); font-size:16px; }
.v-word { font-size:48px; font-weight:700; letter-spacing:-.035em; line-height:1; color:var(--paper); }
.verdict.abstain .v-word { color:var(--acc); }
.verdict.idle .v-word { color:#5E6B80; }
.v-note { color:var(--pale); font-size:15px; font-variant-numeric:tabular-nums; }
.reason { margin-top:10px; color:var(--pale); font-size:14px; }
.reason code { background:#2A2113; color:var(--acc); padding:3px 8px; border-radius:6px; }
.prob-list { display:flex; flex-direction:column; gap:14px; margin-top:12px; }
.prob-head { display:flex; justify-content:space-between; color:var(--pale) !important; font-size:13px; }
.tau-key { color:var(--acc); }
.prob-row { display:grid; grid-template-columns:100px 1fr 72px; align-items:center; gap:3px 12px; }
.prob-id { color:var(--paper); font-size:15px; overflow:hidden; text-overflow:ellipsis; font-family:var(--font-mono); }
.prob-desc { grid-column:2 / 4; color:var(--pale); font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.prob-bar { background:var(--navy2); height:16px; border-radius:8px; position:relative; }
.prob-fill { position:absolute; inset:0 auto 0 0; background:#8E9BAD; border-radius:8px; transform-origin:left;
             animation:grow .7s cubic-bezier(.16,1,.3,1) both; }
.prob-row:nth-child(3) .prob-fill { animation-delay:.06s } .prob-row:nth-child(4) .prob-fill { animation-delay:.12s }
.prob-row.top .prob-fill { background:var(--paper); }
@keyframes grow { from { transform:scaleX(0) } to { transform:scaleX(1) } }
.verdict:not(.idle) { animation:rise .45s cubic-bezier(.16,1,.3,1) both .25s; }
@keyframes rise { from { opacity:0; transform:translateY(8px) } to { opacity:1; transform:none } }
@media (prefers-reduced-motion: reduce) { .prob-fill, .verdict { animation:none !important; } }
.tau { position:absolute; top:-6px; bottom:-6px; width:3px; margin-left:-1.5px; background:var(--acc); border-radius:2px; }
.prob-num { color:var(--paper); font-size:15px; text-align:right; font-variant-numeric:tabular-nums; font-family:var(--font-mono); }
.ghost .prob-bar { background:#22324A; }
#decision .prob-head, #decision .prob-head span:first-child { color:var(--pale) !important; }
#decision .tau-key { color:var(--acc) !important; }
#meta, #meta p { color:var(--pale) !important; font-size:12.5px; margin-top:6px; }
@media (max-width: 480px) { /* the game tab's phone gate: keep the shared header inside 390px */
  #bar { grid-template-columns: auto 1fr; gap: 10px 14px; }
  #bar img { width: 42px; height: 42px; }
  #bar h1 { font-size: 19px; }
  #bar p { font-size: 12.5px; }
  #bar .nums { gap: 14px; }
  #bar .num b { font-size: 16px; }
  #bar .num span { font-size: 10px; }
}
@media (max-width: 760px) {
  #bar { grid-template-columns:auto 1fr; }
  #bar .nums { grid-column:1 / 3; gap:22px; flex-wrap:wrap; }
  #bar .links { grid-column:1 / 3; }
  #decision { position:static; }
  .v-word { font-size:40px; }
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Geist"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "monospace"],
).set(button_primary_background_fill="#1B2A41", button_primary_background_fill_hover="#24354F",
      button_primary_text_color="#F7F6F3", block_radius="12px", block_shadow="none",
      body_text_color="#14181F", block_label_text_color="#5C6470")


def build() -> gr.Blocks:
    with gr.Blocks(css=CSS + ACP_CSS, theme=THEME, title="MK-SH-V · visual decisions with a gate") as app:
        gr.HTML("<header id='bar'><img src='/gradio_api/file=static/logo-mark.png' alt='MK-SH-V mark: a whale shark over an amber band'>"
                "<div><h1>MK-SH-V</h1><p>A 260M visual decision model: calibrated probabilities, "
                "a choice, and a refusal below the gate.</p></div>"
                "<div class='nums'><div class='num'><b>0.828</b><span>real screenshots, held out</span></div>"
                "<div class='num gate'><b>&tau; 0.55</b><span>validated gate</span></div>"
                "<div class='num'><b>98.6 %</b><span>coverage, CP95 risk &le; 4.8 %*</span></div></div>"
                "<nav class='links'><a href='https://huggingface.co/Gjusev/MK-SH-V'>Model card</a>"
                "<a href='https://github.com/Gjusev/mk-sh-v'>Code and results</a>"
                "<a href='https://github.com/Gjusev/mk-sh-v#the-decision-contract'>How it decides</a></nav></header>")
        with gr.Tabs():
            with gr.Tab("Decision demo"):
                _demo_tab()
            with gr.Tab("Amber Checkpoint"):
                gr.HTML(GAME_FONTS)
                build_tab(_model)
        with gr.Accordion("* The shipped policy and its held-out validation (E48)", open=False):
            gr.Markdown(
                f"This demo runs the released policy (E47 checkpoint, seed 43): temperature "
                f"**T = {T:g}** and gate **tau = {TAU:g}**, fitted on calibration splits disjoint "
                "from every reported set. Held-out validation, exact two-sided Clopper-Pearson 95 %"
                " (budget: 5 % selective risk):\n\n"
                "| Set | Coverage | Selective risk | CP95 upper |\n"
                "|---|---|---|---|\n"
                "| synthetic test s43 (shipped) | 0.986 | 3.45 % | 4.1 % (held) |\n"
                "| synthetic test s42 | 1.000 | 4.14 % | 4.8 % (held) |\n"
                "| real GUI s43/s42 | 0.973 / 1.000 | 16.1 / 20.0 % | above budget |\n"
                "| real CORD s43/s42 | 0.892 / 1.000 | 25.3 / 26.6 % | above budget |\n"
                "| 3,000 fresh numeric documents s43 | 0.876 | 22.9 % | above budget |\n\n"
                "The budget holds on the synthetic test only. On real images abstention "
                "is the campaign's declared open problem, and this demo shows it unfiltered.")
        gr.Markdown(
            "Example images: CORD v2 (Park et al. 2019, NAVER Clova) and GUI-Odyssey "
            "(Lu et al. 2024, OpenGVLab), both CC BY 4.0. Model: Apache-2.0.")
    return app




def _demo_tab() -> None:
    with gr.Row(equal_height=False):
        with gr.Column(scale=6, min_width=340):
            gr.HTML("<p id='cases-label'>Real cases from the training splits, one click. "
                    "Or drop your own page or screenshot below and press Decide.</p>")
            gr.HTML("<p class='group-label'>The model answers &mdash; confident, and correct "
                    "(checked against each source's gold label)</p>")
            with gr.Row(elem_id="cases"):
                case_btns = [gr.Button(f"{CASES_DB[i]['family'].replace('_', ' ')} · "
                                       f"{CASES_DB[i]['language'].upper()}", icon=EXAMPLES[i][0],
                                       min_width=150, size="sm") for i in ANSWERS]
            gr.HTML("<p class='caveat'>On screenshots the claim is end-of-task screen detection. Checking "
                    "that the goal in the State is the one finished is only partial (measured in E47: paired "
                    "with an unrelated goal it still says yes about 22 % of the time), so it is not claimed.</p>")
            gr.HTML("<p class='group-label'>Where it fails &mdash; a refusal below tau 0.55, and a confident "
                    "wrong answer above it (real receipts: about 1 in 4 kept answers is wrong)</p>")
            with gr.Row(elem_id="cases2"):
                fail_btns = [gr.Button(f"{CASES_DB[i]['family'].replace('_', ' ')} · "
                                       f"{CASES_DB[i]['language'].upper()} · {kind}", icon=EXAMPLES[i][0],
                                       min_width=150, size="sm") for i, kind in FAILURES]
            image_in = gr.Image(type="pil", label="Your image: document page or screenshot", height=300)
            question_in = gr.Textbox(label="Question", value=EXAMPLES[0][2] if EXAMPLES else "")
            with gr.Row():
                state_in = gr.Textbox(label="State", value=EXAMPLES[0][1] if EXAMPLES else "", lines=3, scale=3, min_width=200)
                options_in = gr.Textbox(label="Options, one per line: id: description",
                                        value=EXAMPLES[0][3] if EXAMPLES else "", lines=3, scale=2, min_width=160)
            go = gr.Button("Decide", variant="primary")
        with gr.Column(scale=5, min_width=340, elem_id="decision"):
            verdict_out = gr.HTML(EMPTY)
            abstain_out = gr.HTML()
            probs_out = gr.HTML()
            meta_out = gr.Markdown(elem_id="meta")
    ins, outs = [image_in, state_in, question_in, options_in], [probs_out, verdict_out, abstain_out, meta_out]
    for btn, i in [*zip(case_btns, ANSWERS), *zip(fail_btns, [i for i, _ in FAILURES])]:
        btn.click(lambda i=i: tuple(EXAMPLES[i]), None, ins).then(decide, ins, outs)
    go.click(decide, ins, outs)


if __name__ == "__main__":
    paths = [str(HERE / "figure"), str(HERE / "static"), str(HERE / "game")]
    gr.set_static_paths(paths)
    build().launch(allowed_paths=paths)
