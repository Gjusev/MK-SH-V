"""Amber Checkpoint: the playable demo tab (docs/internal/AMBER_CHECKPOINT_SPEC.md,
art: docs/internal/AMBER_CHECKPOINT_ART.md, reference: demo/game/mock.html).

The player is the inspector; AMBER SYSTEM is the released model. Every check
runs the REAL decide path (one encoder pass, U1024 input rule, softmax / T,
gate tau) with the constants below, sourced from the E48 summary and the
release policy. Gold labels live in days.json and never reach the model.
All styling is scoped under .acp-root.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
from html import escape
from pathlib import Path

HERE = Path(__file__).parent


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)

# ---- The shipped policy, one constants block (release/inference-policy.json, E48) ----
T = 1.124                # temperature, fitted on cal_temperature (E48)
TAU = 0.55               # abstention gate, fitted on cal_policy (E48, CP-upper rule)
COVERAGE = 0.9855        # held-out synthetic test s43
CP95_UPPER = 0.0406      # exact Clopper-Pearson 95% upper bound of selective risk
CHECKPOINT = "a1v_e47_s43"
CHECKPOINT_SHA = "afcbeaa45a4c033b5e7d3f033b85fdda2642c0fef78397fcecc5bef53cd83c29"  # best.pt provenance
WEIGHTS_SHA = "ff4a47599b0f127af6393ff7e74996c9f0e6e47a182fa5ccca4a672251c79491"  # release/model.safetensors
MODEL_ID = "Gjusev/MK-SH-V"
PLATE = (f"tau {TAU:g} · coverage {COVERAGE:.1%} · CP95 ≤ {CP95_UPPER:.1%} · "
         f"E47 s43 / E48 policy")
HONESTY = ("AMBER SYSTEM: E47 seed 43, E48 policy; every check measured in "
           "github.com/Gjusev/mk-sh-v. Documents are synthetic, printed by our generator; "
           "the gate is validated on the synthetic protocol (E48).")

DAYS = json.loads((HERE / "days.json").read_text(encoding="utf-8"))["days"]
ASSETS = "assets"  # static root: demo/game (gr.set_static_paths)
FILE = "/gradio_api/file=" + ASSETS  # gradio file route for <img> srcs

_model = None
_model_lock = threading.Lock()
_cache: dict[str, dict] = {}  # check id -> {p_yes, pred, backed}; deterministic, shared


def load_model():
    """Same weights as the demo tab, loaded once per process."""
    global _model
    if _model is not None:
        return _model
    import torch
    from huggingface_hub import snapshot_download
    from mksh_v.arch_a1v import NeoMMEMarkerScorer
    from safetensors.torch import load_file

    pkg = Path(os.environ.get("MKSHV_MODEL_DIR") or snapshot_download(MODEL_ID))
    import hashlib
    got = hashlib.sha256((pkg / "model.safetensors").read_bytes()).hexdigest()
    if got != WEIGHTS_SHA:
        raise RuntimeError(f"Amber Checkpoint ABORT: weights sha {got} != pinned {WEIGHTS_SHA}")
    m = NeoMMEMarkerScorer()
    m.ensure_markers(32)
    m.load_state_dict(load_file(str(pkg / "model.safetensors")), strict=True)
    m.eval()
    torch.set_num_threads(2)
    _model = m
    return _model


def decide_checks(image, checks: list[dict]) -> list[dict]:
    """The demo's decide() over all checks of one document in a single encoder
    pass (the E42 batch form of the same path): U1024 rule, softmax(logits / T),
    gate tau on the top option. Options come from the check (6-way doc_type,
    3-way form, N names, 2-way threshold: every set is the measured one).
    Cached by check id."""
    import torch
    from mokka_decisions.contracts import DecisionCase, Option

    todo = [c for c in checks if c["id"] not in _cache]
    if todo and image is None:
        raise ValueError("document image missing")
    if todo:
        model = load_model()
        cases = [DecisionCase(
            id=c["id"], group_id=c["id"], source="amber-checkpoint", language="en",
            domain="documents", state=c["state"], question=c["question"],
            options=tuple(Option(id=o["id"], description=o["description"]) for o in c["options"]))
            for c in todo]
        with _model_lock, torch.inference_mode():
            from mksh_v.arch_a1v import forward_a1v
            enc = model.encode_cases(cases, max_state_tokens=700, max_option_tokens=96)
            batch = model.build_batch(enc, [image] * len(cases), "cpu", max_side=1024, upscale_to=1024)
            probs = (forward_a1v(model, batch) / T).softmax(-1).tolist()
        for c, p in zip(todo, probs):
            top = max(range(len(p)), key=lambda i: p[i])
            _cache[c["id"]] = {"p_pred": round(p[top], 4),
                               "pred": c["options"][top]["id"],
                               "pred_desc": c["options"][top]["description"],
                               "backed": p[top] >= TAU}
    return [_cache[c["id"]] for c in checks]


def _prefetch(image, checks: list[dict]) -> None:
    """Warm the next document's checks off the UI thread."""
    if any(c["id"] not in _cache for c in checks):
        threading.Thread(target=decide_checks, args=(image, checks), daemon=True).start()


def _doc_image(day: int, i: int):
    from PIL import Image
    return Image.open(HERE / ASSETS / "docs" / f"d{day}-{i + 1}.png").convert("RGB")


# ------------------------------------------------------------------ session ---
def new_session() -> dict:
    return {"phase": "intro", "day": 1, "doc_i": 0, "stamped": None, "history": []}


def _doc(day_i: int, doc_i: int) -> dict:
    return DAYS[day_i - 1]["docs"][doc_i]


def _checks(day_i: int, doc_i: int) -> list[dict]:
    return _doc(day_i, doc_i)["checks"]


def start_day(s: dict) -> dict:
    s.update(phase="doc", doc_i=0, stamped=None)
    decide_checks(_doc_image(s["day"], 0), _checks(s["day"], 0))
    if len(DAYS[s["day"] - 1]["docs"]) > 1:
        _prefetch(_doc_image(s["day"], 1), _checks(s["day"], 1))
    return s


def stamp_approve(s: dict) -> dict:
    return stamp(s, "approve")


def stamp_deny(s: dict) -> dict:
    return stamp(s, "deny")


def stamp(s: dict, kind: str) -> dict:
    if s["phase"] == "doc" and s["stamped"] is None:
        s["stamped"] = kind
        d = _doc(s["day"], s["doc_i"])
        s["history"].append({"day": s["day"], "doc": d["id"], "player": kind, "gold": d["verdict"],
                             "violations": d["violations"]})
    return s


def file_unstamped(s: dict) -> dict:
    if s["phase"] == "doc" and s["stamped"] is None:
        s["stamped"] = "unstamped"
        d = _doc(s["day"], s["doc_i"])
        s["history"].append({"day": s["day"], "doc": d["id"], "player": "unstamped",
                             "gold": d["verdict"], "violations": d["violations"]})
    return s


def advance(s: dict) -> dict:
    if s["phase"] == "intro":
        return start_day(s)
    if s["phase"] == "doc":
        if s["stamped"] is None:  # advance only adjudicates after a stamp or filing
            return s
        day = DAYS[s["day"] - 1]
        if s["doc_i"] + 1 < len(day["docs"]):
            s["doc_i"] += 1
            s["stamped"] = None
            decide_checks(_doc_image(s["day"], s["doc_i"]), _checks(s["day"], s["doc_i"]))
            nxt = s["doc_i"] + 1
            if nxt < len(day["docs"]):
                _prefetch(_doc_image(s["day"], nxt), _checks(s["day"], nxt))
        else:
            s["phase"] = "day_end"
    elif s["phase"] == "day_end":
        if s["day"] < len(DAYS):
            s["day"] += 1
            s["phase"] = "intro"
        else:
            s["phase"] = "campaign_end"
    elif s["phase"] == "campaign_end":
        s.update(new_session())  # replay identically: day seeds are fixed
    return s


# ------------------------------------------------------------------ scoring ---
def day_scores(s: dict, day: int) -> dict:
    hist = [h for h in s["history"] if h["day"] == day]
    player = [h["player"] == h["gold"] for h in hist]
    docs = DAYS[day - 1]["docs"]
    checks = [c for d in docs for c in d["checks"] if c["id"] in _cache]
    model_ok = [(_cache[c["id"]]["pred"] == c["gold"]) for c in checks]
    answered = [ok for c, ok in zip(checks, model_ok) if _cache[c["id"]]["backed"]]
    return {"docs": len(hist), "player_acc": round(sum(player) / max(1, len(player)), 3),
            "model_acc": round(sum(model_ok) / max(1, len(model_ok)), 3),
            "routed": sum(not _cache[c["id"]]["backed"] for c in checks),
            "model_answered_acc": round(sum(answered) / max(1, len(answered)), 3),
            "checks": len(checks)}


# ------------------------------------------------------------------- render ---
def render_stage(s: dict) -> str:
    if s["phase"] == "intro":
        return _intro_html(s)
    if s["phase"] == "doc":
        return _doc_html(s)
    if s["phase"] == "day_end":
        return _day_end_html(s)
    return _campaign_end_html(s)


def _board_html(day: int) -> str:
    d = DAYS[day - 1]
    last = len(SLIPS_OF[day])
    slips = "".join(
        f"<div class='slip{' today' if i == last - 1 else ''}'><h3>{escape(t)}</h3>"
        f"<p>{escape(x)}</p><span class='d'>{escape(p)}</span></div>"
        for i, (t, x, p) in enumerate(SLIPS_OF[day]))
    return (f"<section class='board' aria-label='Rule board'><div class='board-head'>"
            f"<div class='day-banner'>DAY {day}<small>Rules of entry</small></div>"
            f"<div class='datestamp'>{d['date'].replace(' 19', '<br>19')}</div></div>"
            f"<div class='slips'>{slips}</div></section>")


SLIPS_OF = {d["day"]: [(s["title"], s["text"], s["pinned"]) for s in d["slips"]] for d in DAYS}


def _impression(kind: str, doc_id: str) -> str:
    if kind not in ("approve", "deny"):
        return ""
    r = random.Random(stable_seed("amber", "stamp", doc_id, kind))
    ir, ix, iy = round(r.uniform(-2, 2), 1), r.randint(-3, 3), r.randint(-3, 3)
    img = f"{FILE}/stamp_{'approve' if kind == 'approve' else 'deny'}_160x96_v7.png"
    return (f"<div class='impression {kind}' style='--ir:{ir}deg;--ix:{ix}px;--iy:{iy}px'>"
            f"<img src='{img}' alt='{kind} stamp impression'></div>")


def _verdict_note(s: dict) -> str:
    if s["stamped"] is None:
        return ""
    d = _doc(s["day"], s["doc_i"])
    ok = s["stamped"] == d["verdict"]
    if ok:
        return "<div class='docnote ok'>CORRECT · the file matches the rules</div>"
    if s["stamped"] == "unstamped":
        v = ", ".join(d["violations"]) or "no violation"
        return f"<div class='docnote bad'>FILED UNSTAMPED · gold: {d['verdict'].upper()} ({v})</div>"
    v = ", ".join(d["violations"]) or "no violation"
    return (f"<div class='docnote bad'>WRONG CALL · gold: {d['verdict'].upper()}"
            f"<span class='cite'>citation: {v}</span></div>")


def _doc_html(s: dict) -> str:
    day, i = s["day"], s["doc_i"]
    d = _doc(day, i)
    results = [_cache[c["id"]] for c in d["checks"]]
    any_routed = any(not r["backed"] for r in results)
    checks = "".join(
        _check_html(ck, r, n + 1) for n, (ck, r) in enumerate(zip(d["checks"], results)))
    tag = ("<div class='tag'><div class='clip'></div><div class='tagslip'>"
           "<h4>SECONDARY INSPECTION</h4><p>CHECK "
           + str(next((n + 1 for n, (ck, r) in enumerate(zip(d["checks"], results))
                       if not r["backed"]), 1))
           + " · ROUTED TO YOU</p></div></div>") if any_routed else ""
    stamp_zone = (_impression(s["stamped"], d["id"]) if s["stamped"] in ("approve", "deny")
                  else "<div class='stampzone'>ADJUDICATION AREA</div>")
    return f"""
{_board_html(day)}
<div class='main'>
  <section class='doczone' aria-label='Document under inspection'>
    <div class='tray'></div>
    <div class='docwrap'>{tag}
      <div class='doc tilt-{i % 2}'><img src='{FILE}/docs/{d['id']}.png' alt='document {d["id"]}'>{stamp_zone}</div>
    </div>
    {_verdict_note(s)}
  </section>
  <aside class='right' aria-label='AMBER SYSTEM panel'>
    <div class='panel'>
      <div class='p-head'><div class='lamp{' on' if any_routed else ' off'}'></div>
        <h2>AMBER SYSTEM<small>BOOTH 04 · MODEL MK-SH-V</small></h2></div>
      {checks}
      <div class='plate'><span>{escape(PLATE.split(' · E47')[0])}</span><span class='gate'>E47 s43 · E48 policy</span></div>
    </div>
  </aside>
</div>"""


def _check_html(ck: dict, r: dict, n: int) -> str:
    q = escape(ck["question"])
    p = r["p_pred"]
    # one bar per check (mock structure): the top option carries the signal
    label = r["pred"] if ck["kind"] != "manifest" else r["pred_desc"]
    label = {"true": "yes", "false": "no"}.get(label, label)
    machine = label if ck["kind"] != "manifest" else escape(r["pred_desc"])
    if r["backed"]:
        verdict = (f"<div class='verdictline'><span class='backed'>● BACKED · p ≥ τ</span>"
                   f"<span class='backed'>machine: {machine}</span></div>")
    else:
        verdict = (f"<div class='verdictline'><span class='routed'>● SECONDARY INSPECTION</span>"
                   f"<span class='routed'>{p:.1%} &lt; τ {TAU:g}</span></div>")
    return (f"<div class='check{' routed-row' if not r['backed'] else ''}' id='check-{n}'>"
            f"<div class='q'>{q}</div>"
            f"<div class='prob-row{' routed-row' if not r['backed'] else ''} top'>"
            f"<span class='prob-id'>{escape(label)[:10]}</span>"
            f"<span class='prob-bar'><span class='prob-fill' style='width:{p * 100:.1f}%'></span>"
            f"<span class='tau'></span></span>"
            f"<span class='prob-num'>{p:.1%}</span></div>{verdict}</div>")


def _intro_html(s: dict) -> str:
    day = s["day"]
    first = day == 1
    head = ("AMBER CHECKPOINT" if first else f"DAY {day}")
    sub = ("The week begins. You hold the stamps; the machine holds the probabilities."
           if first else "The board grows. The rules of every prior day still apply.")
    return f"""
{_board_html(day)}
<div class='main intro'>
  <section class='introcard'>
    <div class='i-kicker'>REPUBLIC OF VELDANIA · BORDER BUREAU · BOOTH 04</div>
    <h1>{head}</h1>
    <p class='i-sub'>{sub}</p>
    <p class='i-body'>Eight documents reach your desk today. AMBER SYSTEM prints one probability
      line per check; below the gate tau {TAU:g} it refuses to advise and the case is routed to you.
      Your stamp is the decision of record.</p>
    <p class='i-foot'>{escape(HONESTY)}</p>
  </section>
</div>"""


def _ledger(s: dict, day: int) -> str:
    hist = [h for h in s["history"] if h["day"] == day]
    cells = []
    for h in hist:
        ok = h["player"] == h["gold"]
        mark = h["player"] if h["player"] in ("approve", "deny") else "unstamped"
        cells.append(
            f"<div class='led{' ok' if ok else ' bad'}'>"
            f"<img src='{FILE}/docs/{h['doc']}.png' alt=''>"
            f"<span class='led-mark'>{mark}</span>"
            f"<span class='led-gold'>gold: {h['gold']}</span></div>")
    return "".join(cells)


def _scores_html(sc: dict) -> str:
    return (f"<div class='scores'>"
            f"<div class='score'><b>{sc['player_acc']:.0%}</b><span>player accuracy</span></div>"
            f"<div class='score'><b>{sc['model_acc']:.0%}</b><span>AMBER alone (argmax)</span></div>"
            f"<div class='score amber'><b>{sc['routed']}</b><span>checks routed to you</span></div>"
            f"<div class='score'><b>{sc['model_answered_acc']:.0%}</b><span>AMBER when backed</span></div>"
            f"</div>")


def _day_end_html(s: dict) -> str:
    day = s["day"]
    sc = day_scores(s, day)
    return f"""
{_board_html(day)}
<div class='main'>
  <section class='endscreen'>
    <div class='i-kicker'>END OF DAY · {escape(DAYS[day - 1]['date'])}</div>
    <h1>DAY {day} LEDGER</h1>
    {_scores_html(sc)}
    <div class='ledger'>{_ledger(s, day)}</div>
    <p class='i-foot'>{escape(HONESTY)}</p>
  </section>
</div>"""


def _campaign_end_html(s: dict) -> str:
    total = {k: 0.0 for k in ("player_acc", "model_acc", "model_answered_acc")}
    routed = checks = 0
    n = len(DAYS)
    for day in range(1, n + 1):
        sc = day_scores(s, day)
        for k in total:
            total[k] += sc[k]
        routed += sc["routed"]
        checks += sc["checks"]
    total = {k: round(v / n, 3) for k, v in total.items()}
    sc = {**total, "routed": routed, "checks": checks}
    return f"""
{_board_html(n)}
<div class='main'>
  <section class='endscreen'>
    <div class='i-kicker'>END OF CAMPAIGN · BOOTH 04 CLOSED</div>
    <h1>THE WEEK IN INK</h1>
    {_scores_html(sc)}
    <p class='i-note'>You played inside the model's validated distribution: {checks} checks,
      every one a measured capability of the released checkpoint.</p>
    <p class='i-foot'>{escape(HONESTY)}</p>
  </section>
</div>"""


# ---------------------------------------------------------------------- CSS ---
ACP_CSS = """
/* Amber Checkpoint, scoped under .acp-root. Tokens: AMBER_CHECKPOINT_ART.md.
   One lamp above the desk: shadows are hard +2/+2 offsets. No gradients, no glow. */
.acp-root{--ink:#23251D;--paper:#EFE7D2;--kraft:#D9CBA8;--olive:#6B6A45;--navy:#1B2A41;
 --track:#24354F;--fpaper:#F7F6F3;--pale:#B7C2D0;--amber:#E8A13C;--amberink:#9A6416;
 --approve:#2F6B47;--deny:#9E3B32;--seal:#2B3A55;--desk:#403D2F;--desk2:#332F22;--muted:#4A4534;
 --grain:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)' opacity='0.55'/%3E%3C/svg%3E");
 font-family:'IBM Plex Sans',ui-sans-serif,sans-serif;color:var(--ink);padding:0 0 6px}
.acp-root *{box-sizing:border-box}
.acp-stage{max-width:1200px;margin:0 auto;background:var(--desk);padding:16px;display:flex;
 flex-direction:column;gap:20px;position:relative;overflow:hidden;min-height:640px;border-radius:4px}
.acp-stage::after{content:"";position:absolute;inset:0;background-image:var(--grain);opacity:.05;pointer-events:none}

/* TOP - RULE BOARD */
.acp-root .board{height:148px;background:var(--kraft);border-radius:6px;position:relative;
 box-shadow:3px 3px 0 rgba(0,0,0,.28);display:flex;align-items:stretch;padding:12px 16px;gap:14px;flex:none}
.acp-root .board::before{content:"";position:absolute;left:0;top:0;bottom:0;width:7px;background:var(--desk2);border-radius:6px 0 0 6px}
.acp-root .board-head{width:158px;flex:none;display:flex;flex-direction:column;justify-content:space-between;padding-right:6px}
.acp-root .day-banner{font-family:'Special Elite',serif;font-size:24px;color:var(--olive);line-height:1.05;letter-spacing:.5px}
.acp-root .day-banner small{display:block;font-size:16px;color:var(--ink);margin-top:2px}
.acp-root .datestamp{width:78px;height:44px;border:2px solid var(--ink);border-radius:50%/40%;display:flex;align-items:center;justify-content:center;
 font-family:'IBM Plex Mono',monospace;font-size:10.5px;font-weight:600;letter-spacing:.5px;color:var(--ink);transform:rotate(-3deg);line-height:1.25;text-align:center}
.acp-root .slips{display:flex;gap:14px;align-items:flex-start;flex:1;padding-top:2px;min-width:0}
.acp-root .slip{flex:1 1 0;min-width:0;background:var(--paper);box-shadow:3px 3px 0 rgba(0,0,0,.22);
 padding:10px 10px 8px;position:relative;transform:rotate(-.5deg)}
.acp-root .slip:nth-child(2){transform:rotate(1.6deg)}
.acp-root .slip:nth-child(3){transform:rotate(-2.2deg);margin-top:4px}
.acp-root .slip:nth-child(4){transform:rotate(.4deg);margin-top:3px}
.acp-root .slip:nth-child(n+5){transform:rotate(-1.4deg);margin-top:5px}
.acp-root .slip::before{content:"";position:absolute;top:-4px;left:50%;width:9px;height:9px;border-radius:50%;
 background:var(--deny);box-shadow:inset -2px -2px 0 rgba(0,0,0,.35), 2px 2px 0 rgba(0,0,0,.18)}
.acp-root .slip h3{font-family:'Special Elite',serif;font-size:16px;font-weight:400;margin:0 0 4px}
.acp-root .slip p{font-size:12.5px;line-height:1.35;margin:0}
.acp-root .slip .d{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--muted);display:block;margin-top:5px;letter-spacing:.5px}
.acp-root .slip.today h3{border-bottom:2px solid var(--amberink);display:inline-block;padding-bottom:1px}

/* MAIN */
.acp-root .main{flex:1;display:flex;gap:20px;min-height:0}

/* CENTER - DOCUMENT */
.acp-root .doczone{flex:1;position:relative;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;min-width:0}
.acp-root .tray{position:absolute;width:512px;height:560px;max-width:100%;background:#C7B58F;transform:rotate(.8deg);
 box-shadow:5px 5px 0 rgba(0,0,0,.34);border-radius:4px}
.acp-root .tray::after{content:"";position:absolute;inset:0;background-image:var(--grain);opacity:.06}
.acp-root .docwrap{position:relative;width:min(472px,100%);filter:drop-shadow(6px 8px 0 rgba(0,0,0,.38))}
.acp-root .doc{position:relative;padding:0;transform:rotate(-.4deg)}
.acp-root .doc.tilt-1{transform:rotate(.5deg)}
.acp-root .doc img{display:block;width:100%;height:auto}
.acp-root .doc::after{content:"";position:absolute;inset:0;background-image:var(--grain);opacity:.045;pointer-events:none;
 clip-path:polygon(0 5px,3px 3px,5px 0,calc(100% - 6px) 0,100% 4px,100% calc(100% - 5px),calc(100% - 4px) 100%,4px 100%,0 calc(100% - 6px))}
.acp-root .stampzone{position:absolute;right:22px;bottom:18px;width:172px;height:64px;border:2px dashed rgba(35,37,29,.42);
 display:flex;align-items:center;justify-content:center;font-size:9.5px;letter-spacing:1.6px;font-weight:600;color:rgba(35,37,29,.62)}
.acp-root .impression{position:absolute;right:14px;bottom:8px;width:160px;height:96px;
 transform:rotate(var(--ir)) translate(var(--ix),var(--iy));animation:inkset .2s ease-out both}
.acp-root .impression img{width:160px;height:96px;image-rendering:pixelated}
@keyframes inkset{0%{transform:rotate(var(--ir)) translate(var(--ix),calc(var(--iy) - 6px)) scale(1.06);opacity:0}
 60%{opacity:1}100%{transform:rotate(var(--ir)) translate(var(--ix),var(--iy)) scale(1)}}
.acp-root .tag{position:absolute;top:-14px;right:-16px;transform:rotate(3deg);z-index:3}
.acp-root .clip{width:14px;height:30px;border:2.5px solid var(--muted);border-radius:7px;margin:0 auto -6px;position:relative}
.acp-root .clip::after{content:"";position:absolute;left:2px;top:2px;bottom:2px;width:5px;border:1.5px solid var(--muted);border-radius:4px}
.acp-root .tagslip{background:var(--paper);box-shadow:2px 2px 0 rgba(0,0,0,.25);padding:7px 10px;border-top:2px solid var(--amberink)}
.acp-root .tagslip h4{font-family:'Special Elite',serif;font-size:13px;font-weight:400;margin:0}
.acp-root .tagslip p{font-size:10px;letter-spacing:.8px;font-weight:600;color:var(--deny);margin:1px 0 0}
.acp-root .docnote{background:var(--paper);box-shadow:3px 3px 0 rgba(0,0,0,.25);padding:8px 14px;
 font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.6px;max-width:472px}
.acp-root .docnote.ok{border-left:4px solid var(--approve)}
.acp-root .docnote.bad{border-left:4px solid var(--deny)}
.acp-root .docnote .cite{display:block;color:var(--muted);font-size:10.5px;margin-top:2px}

/* RIGHT - AMBER SYSTEM */
.acp-root .right{width:360px;max-width:44%;flex:none;display:flex;flex-direction:column}
.acp-root .panel{background:var(--navy);border-radius:16px;padding:18px 20px 14px;color:var(--fpaper);
 position:relative;box-shadow:4px 4px 0 rgba(0,0,0,.3)}
.acp-root .panel::before{content:"";position:absolute;inset:0;border-radius:16px;background-image:var(--grain);opacity:.03;pointer-events:none}
.acp-root .p-head{display:flex;align-items:center;gap:10px;border-bottom:1px solid #2A3B58;padding-bottom:10px;margin-bottom:10px}
.acp-root .lamp{width:32px;height:32px;flex:none;border-radius:50%;background-size:100% 100%;image-rendering:pixelated}
.acp-root .lamp.on{background-image:var(--lamp-on);animation:lampblink .32s steps(1) 1}
.acp-root .lamp.off{background-image:var(--lamp-off)}
@keyframes lampblink{0%,50%{filter:brightness(.4)}25%,75%,100%{filter:none}}
.acp-root .p-head h2{font-size:15.5px;font-weight:600;letter-spacing:2.5px;margin:0}
.acp-root .p-head small{display:block;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--pale);letter-spacing:1.2px;margin-top:2px;font-weight:400}
.acp-root .check{padding:8px 0 9px;border-bottom:1px solid #22324A}
.acp-root .check:last-of-type{border-bottom:none}
.acp-root .q{font-size:12.5px;color:var(--pale);line-height:1.35;margin-bottom:6px}
.acp-root .prob-row{display:grid;grid-template-columns:44px 1fr 52px;align-items:center;gap:3px 10px}
.acp-root .prob-id{font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--fpaper)}
.acp-root .prob-bar{background:var(--track);height:16px;border-radius:8px;position:relative}
.acp-root .prob-fill{position:absolute;inset:0 auto 0 0;border-radius:8px;background:#8E9BAD;height:16px;
 animation:acp-grow .7s cubic-bezier(.16,1,.3,1) both;transform-origin:left}
.acp-root .prob-row.top .prob-fill{background:var(--fpaper)}
.acp-root .check:nth-of-type(2) .prob-fill{animation-delay:.06s}
.acp-root .check:nth-of-type(3) .prob-fill{animation-delay:.12s}
@keyframes acp-grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
.acp-root .tau{position:absolute;top:-4px;bottom:-4px;width:3px;margin-left:-1.5px;background:var(--amber);border-radius:2px;left:55%}
.acp-root .prob-num{font-family:'IBM Plex Mono',monospace;font-size:13px;text-align:right;color:var(--fpaper);font-variant-numeric:tabular-nums}
.acp-root .verdictline{display:flex;justify-content:space-between;margin-top:5px;font-size:11.5px;font-weight:600;letter-spacing:1.2px}
.acp-root .backed{color:var(--pale)}
.acp-root .routed{color:var(--amber)}
.acp-root .check.routed-row .prob-bar{outline:1.5px solid var(--amber);outline-offset:2px;border-radius:8px}
.acp-root .check.routed-row .prob-num{color:var(--amber)}
.acp-root .plate{margin-top:12px;border-top:1px solid #2A3B58;padding-top:9px;display:flex;justify-content:space-between;
 font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--fpaper)}
.acp-root .plate .gate{color:var(--amber)}

/* INTRO / END SCREENS */
.acp-root .introcard,.acp-root .endscreen{flex:1;background:var(--kraft);border-radius:6px;padding:34px 40px;
 box-shadow:3px 3px 0 rgba(0,0,0,.28);max-width:760px;margin:0 auto;align-self:stretch;display:flex;flex-direction:column;justify-content:center}
.acp-root .i-kicker{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:2px;color:var(--muted)}
.acp-root .introcard h1,.acp-root .endscreen h1{font-family:'Special Elite',serif;font-size:44px;font-weight:400;color:var(--olive);margin:6px 0 10px;line-height:1}
.acp-root .i-sub{font-size:16px;margin:0 0 14px;max-width:56ch}
.acp-root .i-body{font-size:14px;line-height:1.55;max-width:62ch;margin:0}
.acp-root .i-foot{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);line-height:1.5;
 border-top:1px solid rgba(35,37,29,.3);padding-top:10px;margin-top:18px;max-width:70ch}
.acp-root .scores{display:flex;gap:26px;flex-wrap:wrap;margin:6px 0 14px}
.acp-root .score b{display:block;font-family:'IBM Plex Mono',monospace;font-size:30px;font-weight:600;font-variant-numeric:tabular-nums}
.acp-root .score.amber b{color:var(--amberink)}
.acp-root .score span{font-size:12px;color:var(--muted)}
.acp-root .ledger{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:4px 0 8px}
.acp-root .led{position:relative;background:#C7B58F;padding:6px;box-shadow:2px 2px 0 rgba(0,0,0,.25)}
.acp-root .led img{display:block;width:100%;height:auto;filter:saturate(.85)}
.acp-root .led .led-mark{position:absolute;right:6px;bottom:6px;font-family:'IBM Plex Mono',monospace;font-size:10px;
 font-weight:600;letter-spacing:1px;padding:2px 6px;color:var(--fpaper);background:rgba(27,42,65,.85)}
.acp-root .led.bad .led-mark{background:rgba(158,59,50,.9)}
.acp-root .led .led-gold{position:absolute;left:6px;top:6px;font-family:'IBM Plex Mono',monospace;font-size:9px;
 letter-spacing:.5px;color:var(--fpaper);background:rgba(35,37,29,.8);padding:1px 4px}
.acp-root .i-note{font-size:13px;line-height:1.5;max-width:60ch;margin:0 0 6px}

/* DESK - stamps live in a gr.Row styled as the ledge */
.acp-root .desk{background:var(--desk);border-radius:4px;padding:10px 16px 14px;display:flex;gap:28px;
 align-items:center;justify-content:center;flex-wrap:wrap;box-shadow:0 3px 0 rgba(0,0,0,.35);border-top:2px solid #5A543C;max-width:1200px;margin:0 auto;width:100%}
.acp-root .acp-stamp{background:none;border:none;padding:0;cursor:pointer;line-height:0;position:relative;
 filter:drop-shadow(3px 4px 0 rgba(0,0,0,.38));transition:transform .12s ease, filter .12s ease;border-radius:0}
.acp-root .acp-stamp img{width:150px;height:90px;image-rendering:pixelated;display:block}
.acp-root .acp-stamp .sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.acp-root #acp-approve{transform:rotate(-1.5deg)}
.acp-root #acp-deny{transform:rotate(1.2deg)}
.acp-root .acp-stamp:hover{transform:translateY(-2px);filter:drop-shadow(5px 6px 0 rgba(0,0,0,.38))}
.acp-root .acp-stamp:active{transform:translateY(4px);filter:drop-shadow(1px 2px 0 rgba(0,0,0,.38))}
.acp-root .acp-stamp:disabled{opacity:.45;cursor:default;transform:none;filter:grayscale(.6)}
.acp-root .acp-stamp:focus-visible{outline:2px solid var(--amber);outline-offset:3px}
.acp-root .acp-plain{background:var(--paper);color:var(--ink);border:2px solid var(--ink);border-radius:6px;
 font-family:'IBM Plex Mono',monospace;font-size:12.5px;letter-spacing:.8px;padding:10px 16px;box-shadow:2px 2px 0 rgba(0,0,0,.3)}
.acp-root .acp-plain:hover{background:#F5EDD8}
.acp-root .acp-advance{background:var(--navy);color:var(--fpaper);border:none;border-radius:8px;
 font-family:'IBM Plex Sans',sans-serif;font-size:14px;font-weight:600;letter-spacing:1px;padding:12px 22px;
 box-shadow:2px 2px 0 rgba(0,0,0,.3)}
.acp-root .acp-advance:hover{background:var(--track)}

@media (max-width:820px){
 .acp-root .main{flex-direction:column}
 .acp-root .right{width:100%;max-width:100%}
 .acp-root .board{height:auto;flex-wrap:wrap}
 .acp-root .slips{flex-wrap:wrap}
 .acp-root .slip{flex:1 1 150px}
 .acp-root .tray{display:none}
 .acp-root .ledger{grid-template-columns:repeat(2,1fr)}
 .acp-root .desk{gap:16px}
 .acp-root .acp-stamp img{width:120px;height:72px}
}
@media (prefers-reduced-motion: reduce){
 .acp-root .prob-fill,.acp-root .impression,.acp-root .lamp.on{animation:none !important}
 .acp-root .acp-stamp:hover,.acp-root .acp-stamp:active{transform:none}
}
"""

_LAMP = "--lamp-on:url(/gradio_api/file=assets/lamp_on_16_v3.png);--lamp-off:url(/gradio_api/file=assets/lamp_off_16_v3.png)"


def build_tab(model=None):
    """The Amber Checkpoint tab: stage HTML + the desk (stamps, file, advance)."""
    import gradio as gr


    global _model
    if model is not None:
        _model = model  # share the demo tab's loaded weights, never a second copy

    s = gr.State(new_session())
    with gr.Column(elem_classes=["acp-root"]):
        stage = gr.HTML(f"<div class='acp-stage' style='{_LAMP}'>{render_stage(new_session())}</div>")
        with gr.Row(elem_classes=["desk"]):
            approve = gr.Button("", elem_id="acp-approve", icon=str(HERE / f"{ASSETS}/stamp_approve_80x48_v7.png"),
                                interactive=False)
            deny = gr.Button("", elem_id="acp-deny", icon=str(HERE / f"{ASSETS}/stamp_deny_80x48_v7.png"),
                             interactive=False)
            file_plain = gr.Button("FILE WITHOUT STAMP", elem_classes=["acp-plain"], interactive=False)
            advance_btn = gr.Button("BEGIN DAY 1", elem_classes=["acp-advance"])

    def paint(sess):
        stage_html = f"<div class='acp-stage' style='{_LAMP}'>{render_stage(sess)}</div>"
        in_doc = sess["phase"] == "doc" and sess["stamped"] is None
        if sess["phase"] == "intro":
            lbl = f"BEGIN DAY {sess['day']}"
        elif sess["phase"] == "doc":
            lbl = "NEXT DOCUMENT" if sess["stamped"] is not None else "AWAITING YOUR STAMP"
        elif sess["phase"] == "day_end":
            lbl = f"BEGIN DAY {sess['day'] + 1}" if sess["day"] < len(DAYS) else "CLOSE THE WEEK"
        else:
            lbl = "PLAY AGAIN"
        inter = sess["phase"] != "doc" or sess["stamped"] is not None
        return stage_html, gr.update(value=lbl, interactive=inter), \
            gr.update(interactive=in_doc), gr.update(interactive=in_doc), gr.update(interactive=in_doc)

    outs = [s, stage, advance_btn, approve, deny, file_plain]

    def run(fn):
        def h(st):
            st = fn(st)
            return (st, *paint(st))
        return h

    approve.click(run(stamp_approve), [s], outs)
    deny.click(run(stamp_deny), [s], outs)
    file_plain.click(run(file_unstamped), [s], outs)
    advance_btn.click(run(advance), [s], outs)
    return stage
